"""
Walk-Forward Analysis Framework
================================

This module provides the core walk-forward testing infrastructure for backtesting.
Ensures strict point-in-time data slicing to prevent lookahead bias.

Key Principles:
- Each decision point only sees data BEFORE that date
- Training/validation window handling
- Systematic iteration through history

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .config_loader import AdaptiveRotationConfig
from .data_preprocessor import DataPreprocessor
from .utils.calendar_utils import get_week_end_dates, is_trading_day


# ============================================================================
# Data Classes for Walk-Forward Results
# ============================================================================

@dataclass
class WalkForwardPeriod:
    """
    Single walk-forward period (one decision point)
    
    Attributes:
        decision_date: Date of decision
        train_start: Start of training window
        train_end: End of training window (= decision_date)
        test_end: End of test/validation period (next rebalance date)
        train_data: Training data (as_of decision_date)
        is_valid: Whether this period has sufficient history
        metadata: Additional metadata
    """
    decision_date: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_end: Optional[pd.Timestamp] = None
    train_data: Optional[Dict[str, pd.DataFrame]] = None
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (
            f"WalkForwardPeriod(decision={self.decision_date.date()}, "
            f"train=[{self.train_start.date()} to {self.train_end.date()}], "
            f"valid={self.is_valid})"
        )


@dataclass
class WalkForwardResult:
    """
    Complete walk-forward analysis result
    
    Attributes:
        periods: List of walk-forward periods
        start_date: Start date of analysis
        end_date: End date of analysis
        total_periods: Total number of periods
        valid_periods: Number of valid periods (sufficient history)
        invalid_periods: Number of invalid periods
        config_hash: Configuration hash for validation
    """
    periods: List[WalkForwardPeriod]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    total_periods: int
    valid_periods: int
    invalid_periods: int
    config_hash: str
    
    def __repr__(self):
        return (
            f"WalkForwardResult({self.total_periods} periods, "
            f"{self.valid_periods} valid, "
            f"dates=[{self.start_date.date()} to {self.end_date.date()}])"
        )
    
    def get_decision_dates(self, valid_only: bool = True) -> List[pd.Timestamp]:
        """Get list of decision dates"""
        if valid_only:
            return [p.decision_date for p in self.periods if p.is_valid]
        return [p.decision_date for p in self.periods]
    
    def get_period_by_date(self, date: pd.Timestamp) -> Optional[WalkForwardPeriod]:
        """Get period for a specific decision date"""
        for period in self.periods:
            if period.decision_date == date:
                return period
        return None
    
    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            "Walk-Forward Analysis Summary",
            "=" * 50,
            f"Start Date: {self.start_date.date()}",
            f"End Date: {self.end_date.date()}",
            f"Total Periods: {self.total_periods}",
            f"Valid Periods: {self.valid_periods} ({self.valid_periods/self.total_periods*100:.1f}%)",
            f"Invalid Periods: {self.invalid_periods}",
            f"Config Hash: {self.config_hash[:16]}...",
        ]
        
        if self.valid_periods > 0:
            first_valid = next(p for p in self.periods if p.is_valid)
            lines.extend([
                "",
                "First Valid Period:",
                f"  Decision Date: {first_valid.decision_date.date()}",
                f"  Training Window: {first_valid.train_start.date()} to {first_valid.train_end.date()}",
                f"  Training Periods: {(first_valid.train_end - first_valid.train_start).days // 7} weeks",
            ])
        
        return "\n".join(lines)


# ============================================================================
# Walk-Forward Generator
# ============================================================================

class WalkForwardAnalyzer:
    """
    Walk-forward analysis framework
    
    Handles systematic iteration through historical data with proper
    point-in-time data slicing.
    
    Two main modes:
    1. Expanding window: Train window grows over time
    2. Rolling window: Train window size stays constant
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> analyzer = WalkForwardAnalyzer(config)
        >>> 
        >>> # Generate walk-forward periods
        >>> result = analyzer.generate_periods(
        ...     start_date="2020-01-01",
        ...     end_date="2023-12-31",
        ...     min_train_periods=26
        ... )
        >>> 
        >>> # Iterate through periods
        >>> for period in result.periods:
        ...     if period.is_valid:
        ...         # Use period.train_data for strategy decisions
        ...         weights = strategy.run(period.train_data)
    """
    
    def __init__(
        self,
        config: AdaptiveRotationConfig,
        preprocessor: Optional[DataPreprocessor] = None,
    ):
        """
        Initialize walk-forward analyzer
        
        Args:
            config: Strategy configuration
            preprocessor: Optional data preprocessor (will create if None)
        """
        self.config = config
        self.preprocessor = preprocessor
        
        if self.preprocessor is None:
            self.preprocessor = DataPreprocessor(config)
    
    def generate_periods(
        self,
        start_date: str,
        end_date: str,
        min_train_periods: Optional[int] = None,
        window_type: str = "expanding",
        rolling_window_size: Optional[int] = None,
        rebalance_frequency: str = "weekly",
    ) -> WalkForwardResult:
        """
        Generate walk-forward periods
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            min_train_periods: Minimum training periods required
                (defaults to config.history.minimum_history_weeks)
            window_type: "expanding" or "rolling"
            rolling_window_size: Size of rolling window (if window_type="rolling")
            rebalance_frequency: Rebalancing frequency ("weekly", "monthly")
        
        Returns:
            WalkForwardResult object
        
        Raises:
            ValueError: If invalid parameters
        
        Examples:
            >>> # Expanding window (train window grows)
            >>> result = analyzer.generate_periods(
            ...     "2020-01-01", "2023-12-31",
            ...     window_type="expanding"
            ... )
            
            >>> # Rolling window (52-week train window)
            >>> result = analyzer.generate_periods(
            ...     "2020-01-01", "2023-12-31",
            ...     window_type="rolling",
            ...     rolling_window_size=52
            ... )
        """
        # Validate parameters
        if window_type not in ["expanding", "rolling"]:
            raise ValueError(f"Invalid window_type: {window_type}")
        
        if window_type == "rolling" and rolling_window_size is None:
            raise ValueError("rolling_window_size required for rolling window")
        
        min_train_periods = min_train_periods or self.config.history.minimum_history_weeks
        
        # Ensure data is loaded
        if self.preprocessor.weekly_data is None:
            print("[WalkForward] Loading data...")
            self.preprocessor.load_and_prepare()
        
        # Get rebalance dates
        if rebalance_frequency == "weekly":
            rebalance_dates = get_week_end_dates(start_date, end_date)
        elif rebalance_frequency == "monthly":
            # Get month-end dates
            all_weeks = get_week_end_dates(start_date, end_date)
            rebalance_dates = pd.DatetimeIndex([
                d for d in all_weeks if d.is_month_end or 
                (d + pd.Timedelta(days=7)).month != d.month
            ])
        else:
            raise ValueError(f"Invalid rebalance_frequency: {rebalance_frequency}")
        
        print(f"[WalkForward] Generating {len(rebalance_dates)} periods")
        print(f"[WalkForward] Window type: {window_type}")
        print(f"[WalkForward] Min train periods: {min_train_periods}")
        
        # Generate periods
        periods = []
        data_start, data_end = self.preprocessor.get_available_date_range()
        
        for i, decision_date in enumerate(rebalance_dates):
            # Skip if decision_date is before data availability
            if decision_date < data_start:
                continue
            
            # Skip if decision_date is after data end
            if decision_date > data_end:
                break
            
            # Determine training window
            if window_type == "expanding":
                # Expanding: use all data from start to decision_date
                train_start = data_start
                train_end = decision_date
                
            elif window_type == "rolling":
                # Rolling: use fixed-size window
                # Go back rolling_window_size periods from decision_date
                available_dates = self.preprocessor.common_dates[
                    self.preprocessor.common_dates <= decision_date
                ]
                
                if len(available_dates) >= rolling_window_size:
                    train_start = available_dates[-rolling_window_size]
                    train_end = decision_date
                else:
                    # Not enough data for rolling window
                    train_start = available_dates[0] if len(available_dates) > 0 else decision_date
                    train_end = decision_date
            
            # Determine test end (next rebalance date)
            if i + 1 < len(rebalance_dates):
                test_end = rebalance_dates[i + 1]
            else:
                test_end = None
            
            # Check if sufficient history
            available_periods = (
                self.preprocessor.common_dates[
                    self.preprocessor.common_dates <= decision_date
                ]
            ).size
            
            is_valid = available_periods >= min_train_periods
            
            # Create period
            period = WalkForwardPeriod(
                decision_date=decision_date,
                train_start=train_start,
                train_end=train_end,
                test_end=test_end,
                is_valid=is_valid,
                metadata={
                    "available_periods": available_periods,
                    "min_required_periods": min_train_periods,
                    "window_type": window_type,
                }
            )
            
            periods.append(period)
        
        # Create result
        result = WalkForwardResult(
            periods=periods,
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            total_periods=len(periods),
            valid_periods=sum(1 for p in periods if p.is_valid),
            invalid_periods=sum(1 for p in periods if not p.is_valid),
            config_hash=self.config.compute_config_hash(),
        )
        
        print(f"[WalkForward] Generated {result.total_periods} periods")
        print(f"[WalkForward] Valid: {result.valid_periods}, Invalid: {result.invalid_periods}")
        
        return result
    
    def load_period_data(
        self,
        period: WalkForwardPeriod,
        lookback_periods: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load training data for a specific period
        
        Args:
            period: WalkForwardPeriod object
            lookback_periods: Optional lookback limit
                (defaults to all data up to decision_date)
        
        Returns:
            Dict mapping symbol → DataFrame (training data)
        
        Examples:
            >>> period = result.periods[10]
            >>> train_data = analyzer.load_period_data(period)
            >>> 
            >>> # Or with lookback limit
            >>> train_data = analyzer.load_period_data(period, lookback_periods=52)
        """
        if not period.is_valid:
            raise ValueError(f"Period {period.decision_date} is not valid (insufficient history)")
        
        # Get data as of decision_date
        train_data = self.preprocessor.get_data_as_of(
            as_of_date=period.decision_date,
            lookback_periods=lookback_periods,
        )
        
        return train_data
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        strategy_func: Callable[[WalkForwardPeriod, Dict[str, pd.DataFrame]], Any],
        min_train_periods: Optional[int] = None,
        window_type: str = "expanding",
        rolling_window_size: Optional[int] = None,
        rebalance_frequency: str = "weekly",
        verbose: bool = True,
    ) -> Tuple[WalkForwardResult, List[Any]]:
        """
        Run full walk-forward backtest with strategy function
        
        Args:
            start_date: Start date
            end_date: End date
            strategy_func: Strategy function that takes (period, data) and returns result
            min_train_periods: Minimum training periods
            window_type: "expanding" or "rolling"
            rolling_window_size: Size of rolling window
            rebalance_frequency: Rebalancing frequency
            verbose: Print progress
        
        Returns:
            (walk_forward_result, strategy_results)
        
        Examples:
            >>> def my_strategy(period, data):
            ...     # Run strategy logic
            ...     weights = {"AAPL": 0.5, "MSFT": 0.5}
            ...     return weights
            >>> 
            >>> wf_result, strategy_results = analyzer.run_backtest(
            ...     "2020-01-01", "2023-12-31",
            ...     strategy_func=my_strategy
            ... )
        """
        # Generate periods
        result = self.generate_periods(
            start_date=start_date,
            end_date=end_date,
            min_train_periods=min_train_periods,
            window_type=window_type,
            rolling_window_size=rolling_window_size,
            rebalance_frequency=rebalance_frequency,
        )
        
        # Run strategy for each valid period
        strategy_results = []
        
        for i, period in enumerate(result.periods):
            if not period.is_valid:
                strategy_results.append(None)
                continue
            
            if verbose and i % 10 == 0:
                print(f"[WalkForward] Processing period {i+1}/{len(result.periods)}: {period.decision_date.date()}")
            
            # Load data
            train_data = self.load_period_data(period)
            
            # Run strategy
            try:
                strategy_result = strategy_func(period, train_data)
                strategy_results.append(strategy_result)
            except Exception as e:
                if verbose:
                    print(f"[WalkForward] Error in period {period.decision_date.date()}: {e}")
                strategy_results.append(None)
        
        if verbose:
            valid_results = sum(1 for r in strategy_results if r is not None)
            print(f"[WalkForward] Completed: {valid_results}/{result.valid_periods} successful")
        
        return result, strategy_results


# ============================================================================
# Utility Functions
# ============================================================================

def get_train_test_split(
    data: Dict[str, pd.DataFrame],
    split_date: pd.Timestamp,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Split data into train and test sets at a specific date
    
    Args:
        data: Dict mapping symbol → DataFrame
        split_date: Date to split at (train includes split_date, test after)
    
    Returns:
        (train_data, test_data)
    
    Examples:
        >>> train, test = get_train_test_split(data, pd.Timestamp("2023-06-30"))
    """
    train_data = {}
    test_data = {}
    
    for symbol, df in data.items():
        train_data[symbol] = df[df.index <= split_date].copy()
        test_data[symbol] = df[df.index > split_date].copy()
    
    return train_data, test_data


def validate_no_lookahead(
    decision_date: pd.Timestamp,
    data: Dict[str, pd.DataFrame],
) -> bool:
    """
    Validate that data has no lookahead bias
    
    Args:
        decision_date: Decision date
        data: Data to validate
    
    Returns:
        True if no lookahead detected
    
    Raises:
        ValueError: If lookahead bias detected
    
    Examples:
        >>> validate_no_lookahead(pd.Timestamp("2023-06-30"), data)
    """
    for symbol, df in data.items():
        if df.index.max() > decision_date:
            raise ValueError(
                f"Lookahead bias detected for {symbol}: "
                f"data contains dates after {decision_date.date()}"
            )
    
    return True


if __name__ == "__main__":
    """Quick test of walk-forward framework"""
    
    print("Testing Walk-Forward Framework")
    print("=" * 60)
    
    from config_loader import load_config
    
    try:
        # Load config
        config = load_config("src/strategies/AdaptiveRotationConf_v1.2.1.yaml")
        print(f"[OK] Config loaded")
        
        # Initialize analyzer
        analyzer = WalkForwardAnalyzer(config)
        print(f"[OK] Analyzer initialized")
        
        # Generate periods
        print("\nGenerating walk-forward periods...")
        result = analyzer.generate_periods(
            start_date="2020-01-01",
            end_date="2020-12-31",
            window_type="expanding",
            min_train_periods=26,
        )
        
        # Print summary
        print(f"\n{result.summary()}")
        
        # Test loading data for a period
        if result.valid_periods > 0:
            first_valid = next(p for p in result.periods if p.is_valid)
            print(f"\nTesting data load for: {first_valid.decision_date.date()}")
            
            train_data = analyzer.load_period_data(first_valid)
            print(f"[OK] Loaded training data for {len(train_data)} symbols")
            
            # Validate no lookahead
            validate_no_lookahead(first_valid.decision_date, train_data)
            print(f"[OK] No lookahead bias detected")
        
        print(f"\n{'='*60}")
        print("[PASS] Walk-forward framework test complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
