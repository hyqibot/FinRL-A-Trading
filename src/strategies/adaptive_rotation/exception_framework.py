"""
Exception Framework
===================

Detects exceptional assets that qualify for special treatment in portfolio.

Key Concepts:
- M/K Persistence: M triggers in last K weeks (original rule)
- Strong Signal: Single trigger with high Z-score AND strong returns (new rule)
- Exception assets get priority slots and weight multiplier

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from .config_loader import AdaptiveRotationConfig


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExceptionCandidate:
    """Exception detection result for a single asset"""
    symbol: str
    trigger_count: int          # Number of triggers in lookback window
    lookback_weeks: int         # Window size (K)
    recent_zscores: List[float] # Z-scores in lookback window
    trigger_dates: List[str]    # Dates when threshold was exceeded
    qualifies: bool             # Whether asset qualifies as exception
    
    # Thresholds
    z_threshold: float = 2.5
    min_trigger_count: int = 2  # M
    
    # Latest metrics
    latest_zscore: float = 0.0
    
    # Strong signal rule (optional)
    strong_signal_qualified: bool = False
    strong_signal_zscore: Optional[float] = None
    strong_signal_return: Optional[float] = None
    strong_signal_benchmark_return: Optional[float] = None
    strong_signal_reason: Optional[str] = None
    
    def __post_init__(self):
        """Validate candidate"""
        # Qualifies if either persistence rule OR strong signal rule is met
        persistence_qualified = self.trigger_count >= self.min_trigger_count
        self.qualifies = persistence_qualified or self.strong_signal_qualified


@dataclass
class ExceptionDetectionResult:
    """Complete exception detection result"""
    exceptions: List[ExceptionCandidate]    # Qualified exceptions
    candidates: Dict[str, ExceptionCandidate]  # All checked assets
    as_of_date: pd.Timestamp
    
    # Detection parameters
    z_threshold: float
    lookback_weeks: int
    min_trigger_count: int
    
    def get_qualified_symbols(self) -> List[str]:
        """Get list of qualified exception symbols"""
        return [exc.symbol for exc in self.exceptions]
    
    def get_candidate(self, symbol: str) -> Optional[ExceptionCandidate]:
        """Get candidate info for a specific symbol"""
        return self.candidates.get(symbol)
    
    def has_exceptions(self) -> bool:
        """Check if any exceptions were found"""
        return len(self.exceptions) > 0


# ============================================================================
# Exception Detection Logic
# ============================================================================

def check_strong_signal_rule(
    symbol: str,
    latest_zscore: float,
    asset_prices: pd.Series,
    benchmark_prices: pd.Series,
    z_threshold: float = 3.5,
    return_multiplier: float = 1.5,
    return_lookback_weeks: int = 12,
    require_positive_return: bool = True,
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Check if asset qualifies under strong signal rule
    
    Strong signal rule requires:
    1. Current Z-score > z_threshold (e.g., 3.5)
    2. 12-week return > multiplier * benchmark 12-week return
    3. Optionally: 12-week return > 0 (positive absolute return)
    
    Args:
        symbol: Asset symbol
        latest_zscore: Current Z-score
        asset_prices: Price series for the asset
        benchmark_prices: Price series for benchmark (e.g., QQQ)
        z_threshold: Z-score threshold (default 3.5)
        return_multiplier: Return multiplier (default 1.5)
        return_lookback_weeks: Lookback period in weeks (default 12)
        require_positive_return: Require positive absolute return (default True)
    
    Returns:
        Tuple of (qualified, asset_return, benchmark_return, reason)
    """
    # Check Z-score threshold
    if latest_zscore < z_threshold:
        return False, None, None, f"Z-score {latest_zscore:.2f} < {z_threshold}"
    
    # Check if sufficient data
    if len(asset_prices) < return_lookback_weeks or len(benchmark_prices) < return_lookback_weeks:
        return False, None, None, "Insufficient price data"
    
    # Calculate returns
    try:
        asset_current = asset_prices.iloc[-1]
        asset_past = asset_prices.iloc[-return_lookback_weeks]
        asset_return = (asset_current - asset_past) / asset_past
        
        benchmark_current = benchmark_prices.iloc[-1]
        benchmark_past = benchmark_prices.iloc[-return_lookback_weeks]
        benchmark_return = (benchmark_current - benchmark_past) / benchmark_past
    except (IndexError, ZeroDivisionError) as e:
        return False, None, None, f"Return calculation error: {e}"
    
    # Check positive return requirement
    if require_positive_return and asset_return <= 0:
        return False, asset_return, benchmark_return, f"Negative return {asset_return:.2%}"
    
    # Check return multiplier
    required_return = return_multiplier * benchmark_return
    if asset_return <= required_return:
        return False, asset_return, benchmark_return, \
            f"Return {asset_return:.2%} <= {return_multiplier}x benchmark {benchmark_return:.2%}"
    
    # Qualified!
    reason = f"Z={latest_zscore:.2f}, Return={asset_return:.2%} > {return_multiplier}x{benchmark_return:.2%}"
    return True, asset_return, benchmark_return, reason


def count_triggers_in_window(
    zscores: pd.Series,
    threshold: float,
    lookback_periods: int,
) -> Tuple[int, List[pd.Timestamp]]:
    """
    Count how many times Z-score exceeded threshold in lookback window
    
    Args:
        zscores: Z-score time series
        threshold: Threshold value
        lookback_periods: Lookback window size
    
    Returns:
        Tuple of (trigger_count, trigger_dates)
    
    Examples:
        >>> zscores = pd.Series([1.0, 2.6, 2.7, 1.5], index=dates)
        >>> count, dates = count_triggers_in_window(zscores, 2.5, 4)
        >>> print(f"Triggers: {count}")
    """
    # Get last N periods
    window = zscores.tail(lookback_periods)
    
    # Find where threshold was exceeded
    triggers = window[window > threshold]
    
    trigger_count = len(triggers)
    trigger_dates = triggers.index.tolist()
    
    return trigger_count, trigger_dates


def check_mk_persistence(
    zscores: pd.Series,
    threshold: float,
    lookback_periods: int,
    min_trigger_count: int,
) -> bool:
    """
    Check M/K persistence rule
    
    Args:
        zscores: Z-score time series
        threshold: Threshold value
        lookback_periods: K (window size)
        min_trigger_count: M (minimum triggers)
    
    Returns:
        True if M/K rule is satisfied
    
    Examples:
        >>> # 2 out of 4 rule
        >>> qualifies = check_mk_persistence(zscores, 2.5, 4, 2)
    """
    trigger_count, _ = count_triggers_in_window(
        zscores, threshold, lookback_periods
    )
    
    return trigger_count >= min_trigger_count


def check_asset_exception(
    symbol: str,
    zscores: pd.Series,
    z_threshold: float,
    lookback_weeks: int,
    min_trigger_count: int,
    as_of_date: Optional[pd.Timestamp] = None,
    # Strong signal parameters (optional)
    asset_prices: Optional[pd.Series] = None,
    benchmark_prices: Optional[pd.Series] = None,
    strong_signal_enabled: bool = False,
    strong_signal_z_threshold: float = 3.5,
    strong_signal_return_multiplier: float = 1.5,
    strong_signal_return_lookback: int = 12,
    strong_signal_require_positive: bool = True,
) -> ExceptionCandidate:
    """
    Check if a single asset qualifies as exception
    
    Checks both:
    1. Original M/K persistence rule
    2. New strong signal rule (if enabled)
    
    Args:
        symbol: Asset symbol
        zscores: Historical Z-score series
        z_threshold: Z-score threshold for persistence rule
        lookback_weeks: K (window size) for persistence rule
        min_trigger_count: M (minimum triggers) for persistence rule
        as_of_date: Current date (uses last date if None)
        
        # Strong signal parameters
        asset_prices: Price series for asset (required for strong signal)
        benchmark_prices: Price series for benchmark (required for strong signal)
        strong_signal_enabled: Enable strong signal rule
        strong_signal_z_threshold: Z-score threshold for strong signal (default 3.5)
        strong_signal_return_multiplier: Return multiplier (default 1.5)
        strong_signal_return_lookback: Return lookback weeks (default 12)
        strong_signal_require_positive: Require positive return (default True)
    
    Returns:
        ExceptionCandidate object
    
    Examples:
        >>> candidate = check_asset_exception(
        ...     "AAPL",
        ...     aapl_zscores,
        ...     z_threshold=2.5,
        ...     lookback_weeks=4,
        ...     min_trigger_count=2
        ... )
        >>> print(f"Qualifies: {candidate.qualifies}")
    """
    # Filter to as_of_date if provided
    if as_of_date is not None:
        zscores = zscores[zscores.index <= as_of_date]
        if asset_prices is not None:
            asset_prices = asset_prices[asset_prices.index <= as_of_date]
        if benchmark_prices is not None:
            benchmark_prices = benchmark_prices[benchmark_prices.index <= as_of_date]
    
    # Get latest z-score (needed for both persistence and strong signal rules)
    latest_zscore = float(zscores.iloc[-1]) if len(zscores) > 0 else 0.0
    
    # Check persistence rule (requires sufficient Z-score history)
    trigger_count = 0
    trigger_dates = []
    trigger_date_strs = []
    recent_zscores = []
    
    if len(zscores) >= lookback_weeks:
        # Sufficient data for persistence rule
        window = zscores.tail(lookback_weeks)
        
        # Count triggers
        trigger_count, trigger_dates = count_triggers_in_window(
            zscores, z_threshold, lookback_weeks
        )
        
        # Get recent z-scores as list
        recent_zscores = window.tolist()
        
        # Format trigger dates as strings
        trigger_date_strs = [d.strftime("%Y-%m-%d") for d in trigger_dates]
    
    # Check strong signal rule (independent of Z-score history)
    # Strong signal only needs: current Z-score + price history
    strong_qualified = False
    strong_return = None
    strong_benchmark_return = None
    strong_reason = None
    
    if strong_signal_enabled and asset_prices is not None and benchmark_prices is not None:
        strong_qualified, strong_return, strong_benchmark_return, strong_reason = \
            check_strong_signal_rule(
                symbol,
                latest_zscore,
                asset_prices,
                benchmark_prices,
                z_threshold=strong_signal_z_threshold,
                return_multiplier=strong_signal_return_multiplier,
                return_lookback_weeks=strong_signal_return_lookback,
                require_positive_return=strong_signal_require_positive,
            )
    
    return ExceptionCandidate(
        symbol=symbol,
        trigger_count=trigger_count,
        lookback_weeks=lookback_weeks,
        recent_zscores=recent_zscores,
        trigger_dates=trigger_date_strs,
        qualifies=(trigger_count >= min_trigger_count) or strong_qualified,
        z_threshold=z_threshold,
        min_trigger_count=min_trigger_count,
        latest_zscore=latest_zscore,
        strong_signal_qualified=strong_qualified,
        strong_signal_zscore=latest_zscore if strong_qualified else None,
        strong_signal_return=strong_return,
        strong_signal_benchmark_return=strong_benchmark_return,
        strong_signal_reason=strong_reason,
    )


# ============================================================================
# Batch Exception Detection
# ============================================================================

def find_exceptions_in_pool(
    asset_zscores: Dict[str, pd.Series],
    z_threshold: float,
    lookback_weeks: int,
    min_trigger_count: int,
    as_of_date: Optional[pd.Timestamp] = None,
    candidate_pool: Optional[List[str]] = None,
) -> List[ExceptionCandidate]:
    """
    Scan multiple assets for exceptions
    
    Args:
        asset_zscores: Dict mapping symbol → Z-score series
        z_threshold: Z-score threshold
        lookback_weeks: K (window size)
        min_trigger_count: M (minimum triggers)
        as_of_date: Current date
        candidate_pool: Optional list of symbols to check (checks all if None)
    
    Returns:
        List of ExceptionCandidate objects (only qualified ones)
    
    Examples:
        >>> exceptions = find_exceptions_in_pool(
        ...     {"AAPL": aapl_z, "MSFT": msft_z},
        ...     z_threshold=2.5,
        ...     lookback_weeks=4,
        ...     min_trigger_count=2
        ... )
        >>> print(f"Found {len(exceptions)} exceptions")
    """
    exceptions = []
    
    # Determine which symbols to check
    if candidate_pool is not None:
        symbols_to_check = candidate_pool
    else:
        symbols_to_check = list(asset_zscores.keys())
    
    # Check each symbol
    for symbol in symbols_to_check:
        if symbol not in asset_zscores:
            continue
        
        zscores = asset_zscores[symbol]
        
        candidate = check_asset_exception(
            symbol,
            zscores,
            z_threshold,
            lookback_weeks,
            min_trigger_count,
            as_of_date,
        )
        
        # Only include qualified exceptions
        if candidate.qualifies:
            exceptions.append(candidate)
    
    # Sort by latest Z-score (strongest first)
    exceptions.sort(key=lambda x: x.latest_zscore, reverse=True)
    
    return exceptions


def check_all_candidates(
    asset_zscores: Dict[str, pd.Series],
    z_threshold: float,
    lookback_weeks: int,
    min_trigger_count: int,
    as_of_date: Optional[pd.Timestamp] = None,
    candidate_pool: Optional[List[str]] = None,
    # Strong signal parameters
    asset_prices: Optional[Dict[str, pd.Series]] = None,
    benchmark_prices: Optional[pd.Series] = None,
    strong_signal_enabled: bool = False,
    strong_signal_z_threshold: float = 3.5,
    strong_signal_return_multiplier: float = 1.5,
    strong_signal_return_lookback: int = 12,
    strong_signal_require_positive: bool = True,
) -> Dict[str, ExceptionCandidate]:
    """
    Check all candidates and return full results (qualified and non-qualified)
    
    Args:
        asset_zscores: Dict mapping symbol → Z-score series
        z_threshold: Z-score threshold
        lookback_weeks: K (window size)
        min_trigger_count: M (minimum triggers)
        as_of_date: Current date
        candidate_pool: Optional list of symbols to check
        
        # Strong signal parameters
        asset_prices: Dict mapping symbol → price series
        benchmark_prices: Benchmark price series (e.g., QQQ)
        strong_signal_enabled: Enable strong signal rule
        strong_signal_z_threshold: Z-score threshold for strong signal
        strong_signal_return_multiplier: Return multiplier
        strong_signal_return_lookback: Return lookback weeks
        strong_signal_require_positive: Require positive return
    
    Returns:
        Dict mapping symbol → ExceptionCandidate
    """
    candidates = {}
    
    # Determine which symbols to check
    if candidate_pool is not None:
        symbols_to_check = candidate_pool
    else:
        symbols_to_check = list(asset_zscores.keys())
    
    # Check each symbol
    for symbol in symbols_to_check:
        if symbol not in asset_zscores:
            continue
        
        zscores = asset_zscores[symbol]
        
        # Get asset prices if available
        symbol_prices = None
        if asset_prices is not None and symbol in asset_prices:
            symbol_prices = asset_prices[symbol]
        
        candidate = check_asset_exception(
            symbol,
            zscores,
            z_threshold,
            lookback_weeks,
            min_trigger_count,
            as_of_date,
            asset_prices=symbol_prices,
            benchmark_prices=benchmark_prices,
            strong_signal_enabled=strong_signal_enabled,
            strong_signal_z_threshold=strong_signal_z_threshold,
            strong_signal_return_multiplier=strong_signal_return_multiplier,
            strong_signal_return_lookback=strong_signal_return_lookback,
            strong_signal_require_positive=strong_signal_require_positive,
        )
        
        candidates[symbol] = candidate
    
    return candidates


# ============================================================================
# High-Level API
# ============================================================================

class ExceptionDetector:
    """
    Detects exceptional assets using M/K persistence rule
    
    Examples:
        >>> detector = ExceptionDetector(
        ...     z_threshold=2.5,
        ...     lookback_weeks=4,
        ...     min_trigger_count=2
        ... )
        >>> 
        >>> result = detector.detect_exceptions(
        ...     asset_zscores=zscores_dict,
        ...     as_of_date=pd.Timestamp("2024-02-01")
        ... )
        >>> 
        >>> print(f"Found {len(result.exceptions)} exceptions")
        >>> for exc in result.exceptions:
        ...     print(f"  {exc.symbol}: {exc.trigger_count}/{exc.lookback_weeks}")
    """
    
    def __init__(
        self,
        z_threshold: float = 2.5,
        lookback_weeks: int = 4,
        min_trigger_count: int = 2,
        # Strong signal parameters
        strong_signal_enabled: bool = False,
        strong_signal_z_threshold: float = 3.5,
        strong_signal_return_multiplier: float = 1.5,
        strong_signal_return_lookback: int = 12,
        strong_signal_require_positive: bool = True,
    ):
        """
        Initialize exception detector
        
        Args:
            z_threshold: Z-score threshold for persistence rule
            lookback_weeks: K (window size) for persistence rule
            min_trigger_count: M (minimum triggers to qualify) for persistence rule
            
            # Strong signal parameters
            strong_signal_enabled: Enable strong signal single-trigger rule
            strong_signal_z_threshold: Z-score threshold for strong signal (default 3.5)
            strong_signal_return_multiplier: Return multiplier (default 1.5)
            strong_signal_return_lookback: Return lookback weeks (default 12)
            strong_signal_require_positive: Require positive absolute return (default True)
        """
        # Persistence rule parameters
        self.z_threshold = z_threshold
        self.lookback_weeks = lookback_weeks
        self.min_trigger_count = min_trigger_count
        
        # Strong signal parameters
        self.strong_signal_enabled = strong_signal_enabled
        self.strong_signal_z_threshold = strong_signal_z_threshold
        self.strong_signal_return_multiplier = strong_signal_return_multiplier
        self.strong_signal_return_lookback = strong_signal_return_lookback
        self.strong_signal_require_positive = strong_signal_require_positive
    
    def detect_exceptions(
        self,
        asset_zscores: Dict[str, pd.Series],
        as_of_date: pd.Timestamp,
        candidate_pool: Optional[List[str]] = None,
        # Strong signal data (optional)
        asset_prices: Optional[Dict[str, pd.Series]] = None,
        benchmark_prices: Optional[pd.Series] = None,
    ) -> ExceptionDetectionResult:
        """
        Detect all exceptional assets
        
        Checks both:
        1. M/K persistence rule (original)
        2. Strong signal rule (if enabled and price data provided)
        
        Args:
            asset_zscores: Dict mapping symbol → Z-score series
            as_of_date: Current decision date
            candidate_pool: Optional list of symbols to check
            asset_prices: Dict mapping symbol → price series (for strong signal)
            benchmark_prices: Benchmark price series (for strong signal, e.g., QQQ)
        
        Returns:
            ExceptionDetectionResult object
        """
        # Check all candidates
        all_candidates = check_all_candidates(
            asset_zscores,
            self.z_threshold,
            self.lookback_weeks,
            self.min_trigger_count,
            as_of_date,
            candidate_pool,
            # Strong signal parameters
            asset_prices=asset_prices,
            benchmark_prices=benchmark_prices,
            strong_signal_enabled=self.strong_signal_enabled,
            strong_signal_z_threshold=self.strong_signal_z_threshold,
            strong_signal_return_multiplier=self.strong_signal_return_multiplier,
            strong_signal_return_lookback=self.strong_signal_return_lookback,
            strong_signal_require_positive=self.strong_signal_require_positive,
        )
        
        # Filter to qualified exceptions
        exceptions = [
            candidate for candidate in all_candidates.values()
            if candidate.qualifies
        ]
        
        # Sort by latest Z-score
        exceptions.sort(key=lambda x: x.latest_zscore, reverse=True)
        
        return ExceptionDetectionResult(
            exceptions=exceptions,
            candidates=all_candidates,
            as_of_date=as_of_date,
            z_threshold=self.z_threshold,
            lookback_weeks=self.lookback_weeks,
            min_trigger_count=self.min_trigger_count,
        )
    
    @classmethod
    def from_config(cls, config: AdaptiveRotationConfig) -> "ExceptionDetector":
        """
        Create detector from configuration
        
        Args:
            config: Strategy configuration
        
        Returns:
            ExceptionDetector instance
        """
        # Get strong signal config (with defaults if not present)
        strong_signal_cfg = getattr(config.exception, 'strong_signal', None)
        
        if strong_signal_cfg is not None:
            strong_signal_enabled = getattr(strong_signal_cfg, 'enabled', False)
            strong_signal_z_threshold = getattr(strong_signal_cfg, 'z_threshold', 3.5)
            strong_signal_return_multiplier = getattr(strong_signal_cfg, 'return_multiplier', 1.5)
            strong_signal_return_lookback = getattr(strong_signal_cfg, 'return_lookback_weeks', 12)
            strong_signal_require_positive = getattr(strong_signal_cfg, 'require_positive_return', True)
        else:
            # Default: disabled
            strong_signal_enabled = False
            strong_signal_z_threshold = 3.5
            strong_signal_return_multiplier = 1.5
            strong_signal_return_lookback = 12
            strong_signal_require_positive = True
        
        return cls(
            z_threshold=config.exception.z_threshold,
            lookback_weeks=config.exception.lookback_weeks,
            min_trigger_count=config.exception.min_trigger_count,
            strong_signal_enabled=strong_signal_enabled,
            strong_signal_z_threshold=strong_signal_z_threshold,
            strong_signal_return_multiplier=strong_signal_return_multiplier,
            strong_signal_return_lookback=strong_signal_return_lookback,
            strong_signal_require_positive=strong_signal_require_positive,
        )


if __name__ == "__main__":
    """Quick test of exception detection"""
    
    print("Testing Exception Detection")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="W-FRI")
    
    # Asset with consistent high Z-scores (should qualify)
    strong_asset = pd.Series(
        [2.6, 2.7, 2.8, 2.9, 2.7, 2.6] + [2.0] * (len(dates) - 6),
        index=dates
    )
    
    # Asset with sporadic high Z-scores (should NOT qualify - only 1 trigger)
    sporadic_asset = pd.Series(
        [2.6, 2.0, 2.1, 2.2] + [1.5] * (len(dates) - 4),
        index=dates
    )
    
    # Asset with moderate Z-scores (should NOT qualify)
    moderate_asset = pd.Series(
        [2.0, 2.1, 2.2, 2.3] + [1.8] * (len(dates) - 4),
        index=dates
    )
    
    zscores_dict = {
        "STRONG": strong_asset,
        "SPORADIC": sporadic_asset,
        "MODERATE": moderate_asset,
    }
    
    # Test detection
    detector = ExceptionDetector(
        z_threshold=2.5,
        lookback_weeks=4,
        min_trigger_count=2
    )
    
    test_date = pd.Timestamp("2024-02-01")
    
    result = detector.detect_exceptions(
        asset_zscores=zscores_dict,
        as_of_date=test_date
    )
    
    print(f"\n[Test Date: {test_date.date()}]")
    print(f"\nDetection Parameters:")
    print(f"  Z-threshold: {result.z_threshold}")
    print(f"  Lookback: {result.lookback_weeks} weeks")
    print(f"  Min triggers: {result.min_trigger_count}")
    
    print(f"\nQualified Exceptions ({len(result.exceptions)}):")
    for exc in result.exceptions:
        print(f"  {exc.symbol}")
        print(f"    Triggers: {exc.trigger_count}/{exc.lookback_weeks}")
        print(f"    Latest Z-score: {exc.latest_zscore:.2f}")
        print(f"    Trigger Dates: {exc.trigger_dates}")
    
    print(f"\nAll Candidates:")
    for symbol, candidate in result.candidates.items():
        status = "✓ QUALIFIED" if candidate.qualifies else "✗ Not Qualified"
        print(f"  {symbol}: {candidate.trigger_count}/{candidate.lookback_weeks} {status}")
    
    print(f"\n{'='*60}")
    print("[PASS] Exception detection test complete!")
