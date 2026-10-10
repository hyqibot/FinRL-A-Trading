"""
Data Preprocessor for Adaptive Multi-Asset Rotation Strategy
=============================================================

This module handles loading, cleaning, and preprocessing of financial data:
- Load daily OHLCV data from CSV files
- Aggregate daily → weekly data
- Align all symbols to common trading calendar
- Point-in-time data slicing for walk-forward analysis

Key Principle: STRICT point-in-time integrity - no lookahead bias allowed!

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings

from .utils.calendar_utils import (
    get_trading_calendar,
    get_week_end_dates,
    is_trading_day,
    align_to_trading_day,
)
from .config_loader import AdaptiveRotationConfig


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_daily_csv(
    symbol: str,
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load daily OHLCV data for a single symbol from CSV
    
    Args:
        symbol: Ticker symbol (e.g., "AAPL", "^GSPC")
        data_dir: Directory containing CSV files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns [date, open, high, low, close, volume]
        Index is DatetimeIndex
    
    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If data loading fails
    
    Examples:
        >>> df = load_daily_csv("AAPL", "./data/fmp_daily")
        >>> df = load_daily_csv("AAPL", "./data", "2020-01-01", "2023-12-31")
    """
    # Construct file path
    csv_path = Path(data_dir) / f"{symbol}_daily.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {symbol} data")
        
        # Parse dates and set as index
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        
        # Filter date range if specified
        if start_date is not None:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.Timestamp(end_date)]
        
        # Validate data
        if df.empty:
            raise ValueError(f"No data for {symbol} in specified date range")
        
        # Check for negative prices/volume (data quality)
        if (df[["open", "high", "low", "close"]] < 0).any().any():
            warnings.warn(f"Negative prices detected in {symbol}, setting to NaN")
            df.loc[(df[["open", "high", "low", "close"]] < 0).any(axis=1)] = np.nan
        
        return df
        
    except Exception as e:
        raise ValueError(f"Failed to load {symbol}: {e}")


def load_multiple_symbols(
    symbols: List[str],
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    required: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load daily data for multiple symbols
    
    Args:
        symbols: List of ticker symbols
        data_dir: Directory containing CSV files
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        required: If True, raise error if any symbol fails to load
    
    Returns:
        Dict mapping symbol → DataFrame
    
    Raises:
        ValueError: If required=True and any symbol fails to load
    
    Examples:
        >>> data = load_multiple_symbols(["AAPL", "MSFT"], "./data")
        >>> aapl_df = data["AAPL"]
    """
    data = {}
    failed = []
    
    for symbol in symbols:
        try:
            df = load_daily_csv(symbol, data_dir, start_date, end_date)
            data[symbol] = df
        except Exception as e:
            if required:
                failed.append((symbol, str(e)))
            else:
                warnings.warn(f"Skipping {symbol}: {e}")
    
    if required and failed:
        error_msg = "\n".join([f"  - {sym}: {err}" for sym, err in failed])
        raise ValueError(f"Failed to load required symbols:\n{error_msg}")
    
    return data


# ============================================================================
# Weekly Aggregation Functions
# ============================================================================

def aggregate_daily_to_weekly(
    daily_df: pd.DataFrame,
    week_end_dates: pd.DatetimeIndex,
    method: str = "last_available",
) -> pd.DataFrame:
    """
    Aggregate daily OHLCV data to weekly frequency
    
    Aggregation rules:
    - Open: First day's open
    - High: Max high across week
    - Low: Min low across week
    - Close: Last day's close
    - Volume: Sum of daily volumes
    
    Args:
        daily_df: Daily OHLCV data with DatetimeIndex
        week_end_dates: Target week-end dates (usually Fridays)
        method: Aggregation method
            - "last_available": Use last available data before week end
            - "strict": Only use data exactly on week end dates
    
    Returns:
        Weekly OHLCV DataFrame indexed by week_end_dates
    
    Note:
        Uses point-in-time semantics - only data BEFORE week end is used
    
    Examples:
        >>> week_ends = get_week_end_dates("2020-01-01", "2020-12-31")
        >>> weekly = aggregate_daily_to_weekly(daily_df, week_ends)
    """
    if method not in ["last_available", "strict"]:
        raise ValueError(f"Invalid method: {method}")
    
    # Initialize result with NaN
    weekly_df = pd.DataFrame(
        index=week_end_dates,
        columns=["open", "high", "low", "close", "volume"]
    )
    
    if method == "last_available":
        # For each week end, aggregate data from previous week
        for i, week_end in enumerate(week_end_dates):
            # Determine start of week
            if i == 0:
                # First week: use 7 days before
                week_start = week_end - pd.Timedelta(days=7)
            else:
                # Subsequent weeks: use previous week end + 1 day
                week_start = week_end_dates[i-1] + pd.Timedelta(days=1)
            
            # Get data for this week (up to and including week_end)
            week_data = daily_df[(daily_df.index > week_start) & 
                                 (daily_df.index <= week_end)]
            
            if not week_data.empty:
                weekly_df.loc[week_end, "open"] = week_data.iloc[0]["open"]
                weekly_df.loc[week_end, "high"] = week_data["high"].max()
                weekly_df.loc[week_end, "low"] = week_data["low"].min()
                weekly_df.loc[week_end, "close"] = week_data.iloc[-1]["close"]
                weekly_df.loc[week_end, "volume"] = week_data["volume"].sum()
    
    elif method == "strict":
        # Only use data points that exactly match week end dates
        for week_end in week_end_dates:
            if week_end in daily_df.index:
                weekly_df.loc[week_end] = daily_df.loc[week_end]
    
    return weekly_df.astype(float)


def aggregate_multiple_symbols_to_weekly(
    daily_data: Dict[str, pd.DataFrame],
    week_end_dates: pd.DatetimeIndex,
    method: str = "last_available",
) -> Dict[str, pd.DataFrame]:
    """
    Aggregate multiple symbols' daily data to weekly
    
    Args:
        daily_data: Dict mapping symbol → daily DataFrame
        week_end_dates: Target week-end dates
        method: Aggregation method ("last_available" or "strict")
    
    Returns:
        Dict mapping symbol → weekly DataFrame
    
    Examples:
        >>> weekly_data = aggregate_multiple_symbols_to_weekly(
        ...     daily_data, week_ends
        ... )
    """
    weekly_data = {}
    
    for symbol, daily_df in daily_data.items():
        try:
            weekly_df = aggregate_daily_to_weekly(daily_df, week_end_dates, method)
            weekly_data[symbol] = weekly_df
        except Exception as e:
            warnings.warn(f"Failed to aggregate {symbol}: {e}")
    
    return weekly_data


# ============================================================================
# Data Alignment Functions
# ============================================================================

def align_symbols_to_common_dates(
    data: Dict[str, pd.DataFrame],
    fill_method: str = "forward",
    max_fill_gaps: int = 2,
) -> Tuple[Dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """
    Align all symbols to common date index
    
    This ensures all symbols have data for the same dates, which is required
    for correlation calculations, portfolio construction, etc.
    
    Args:
        data: Dict mapping symbol → DataFrame (with DatetimeIndex)
        fill_method: How to handle missing data
            - "forward": Forward fill (use last available)
            - "drop": Drop dates with any missing data
            - "none": Keep NaN values
        max_fill_gaps: Maximum consecutive gaps to forward fill (0 = no limit)
    
    Returns:
        (aligned_data, common_dates)
        - aligned_data: Dict mapping symbol → aligned DataFrame
        - common_dates: DatetimeIndex of common dates
    
    Raises:
        ValueError: If no common dates found
    
    Examples:
        >>> aligned, dates = align_symbols_to_common_dates(weekly_data)
    """
    if not data:
        raise ValueError("No data provided")
    
    # Find union of all dates
    all_dates = pd.DatetimeIndex([])
    for df in data.values():
        all_dates = all_dates.union(df.index)
    
    all_dates = all_dates.sort_values()
    
    if all_dates.empty:
        raise ValueError("No common dates found")
    
    # Reindex all symbols to common dates
    aligned_data = {}
    
    for symbol, df in data.items():
        # Reindex to all_dates
        aligned_df = df.reindex(all_dates)
        
        # Apply fill method
        if fill_method == "forward":
            if max_fill_gaps > 0:
                # Limit forward fill to max_fill_gaps
                aligned_df = aligned_df.ffill(limit=max_fill_gaps)
            else:
                # No limit on forward fill
                aligned_df = aligned_df.ffill()
        elif fill_method == "drop":
            # Will drop dates with NaN later
            pass
        elif fill_method == "none":
            # Keep NaN as is
            pass
        else:
            raise ValueError(f"Invalid fill_method: {fill_method}")
        
        aligned_data[symbol] = aligned_df
    
    # If drop method, find dates where all symbols have data
    if fill_method == "drop":
        # Find dates with complete data across all symbols
        valid_mask = pd.Series(True, index=all_dates)
        for df in aligned_data.values():
            valid_mask &= df.notna().all(axis=1)
        
        common_dates = all_dates[valid_mask]
        
        # Filter all DataFrames to common dates
        aligned_data = {
            symbol: df.loc[common_dates]
            for symbol, df in aligned_data.items()
        }
    else:
        common_dates = all_dates
    
    return aligned_data, common_dates


# ============================================================================
# Point-in-Time Data Slicing (Critical for Walk-Forward)
# ============================================================================

def get_data_as_of_date(
    data: Dict[str, pd.DataFrame],
    as_of_date: Union[str, pd.Timestamp],
    lookback_periods: Optional[int] = None,
    include_as_of_date: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Get data strictly as of a specific date (point-in-time slice)
    
    This is CRITICAL for walk-forward analysis to prevent lookahead bias.
    
    Args:
        data: Dict mapping symbol → DataFrame with DatetimeIndex
        as_of_date: Cut-off date (YYYY-MM-DD or Timestamp)
        lookback_periods: Optional number of periods to include
            - If None, include all data up to as_of_date
            - If int, include last N periods up to as_of_date
        include_as_of_date: If True, include as_of_date in result
    
    Returns:
        Dict mapping symbol → sliced DataFrame
    
    Examples:
        >>> # Get all data up to 2024-02-01 (inclusive)
        >>> data_slice = get_data_as_of_date(data, "2024-02-01")
        
        >>> # Get last 26 weeks of data up to 2024-02-01
        >>> data_slice = get_data_as_of_date(data, "2024-02-01", lookback_periods=26)
    """
    as_of_date = pd.Timestamp(as_of_date)
    
    sliced_data = {}
    
    for symbol, df in data.items():
        # Filter to dates <= as_of_date
        if include_as_of_date:
            mask = df.index <= as_of_date
        else:
            mask = df.index < as_of_date
        
        sliced_df = df[mask].copy()
        
        # Apply lookback if specified
        if lookback_periods is not None and lookback_periods > 0:
            sliced_df = sliced_df.tail(lookback_periods)
        
        sliced_data[symbol] = sliced_df
    
    return sliced_data


def validate_sufficient_history(
    data: Dict[str, pd.DataFrame],
    min_periods: int,
    as_of_date: Union[str, pd.Timestamp],
) -> Tuple[bool, List[str]]:
    """
    Validate that all symbols have sufficient history as of date
    
    Args:
        data: Dict mapping symbol → DataFrame
        min_periods: Minimum required periods
        as_of_date: Date to check as of
    
    Returns:
        (is_valid, insufficient_symbols)
        - is_valid: True if all symbols have sufficient history
        - insufficient_symbols: List of symbols with insufficient data
    
    Examples:
        >>> is_valid, missing = validate_sufficient_history(
        ...     data, min_periods=26, as_of_date="2024-02-01"
        ... )
        >>> if not is_valid:
        ...     print(f"Insufficient data for: {missing}")
    """
    as_of_date = pd.Timestamp(as_of_date)
    insufficient = []
    
    for symbol, df in data.items():
        # Count periods up to as_of_date
        available_periods = (df.index <= as_of_date).sum()
        
        if available_periods < min_periods:
            insufficient.append(symbol)
    
    is_valid = len(insufficient) == 0
    
    return is_valid, insufficient


# ============================================================================
# High-Level Data Loader
# ============================================================================

class DataPreprocessor:
    """
    High-level data preprocessing pipeline
    
    Handles complete workflow:
    1. Load daily CSVs for all required symbols
    2. Aggregate to weekly frequency
    3. Align all symbols to common calendar
    4. Provide point-in-time slicing for walk-forward
    
    Examples:
        >>> config = load_config("config.yaml")
        >>> preprocessor = DataPreprocessor(config)
        >>> 
        >>> # Load and prepare all data
        >>> preprocessor.load_and_prepare()
        >>> 
        >>> # Get data as of specific date
        >>> data_slice = preprocessor.get_data_as_of("2024-02-01")
        >>> 
        >>> # Check if sufficient history
        >>> is_valid = preprocessor.has_sufficient_history("2024-02-01", min_weeks=26)
    """
    
    def __init__(self, config: AdaptiveRotationConfig):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.daily_data: Optional[Dict[str, pd.DataFrame]] = None
        self.weekly_data: Optional[Dict[str, pd.DataFrame]] = None
        self.week_end_dates: Optional[pd.DatetimeIndex] = None
        self.common_dates: Optional[pd.DatetimeIndex] = None
    
    def load_and_prepare(
        self,
        data_dir: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Load and prepare all data
        
        Args:
            data_dir: Data directory (defaults to config.paths.data_root)
            start_date: Start date (defaults to config.dates.start_date)
            end_date: End date (defaults to config.dates.end_date)
        
        Raises:
            ValueError: If data loading or processing fails
        """
        # Use config defaults if not specified
        if data_dir is None:
            data_dir = self.config.paths.data_root
            # Convert to absolute path if relative
            if not Path(data_dir).is_absolute():
                # Assume relative to project root (3 levels up from this file)
                module_dir = Path(__file__).parent
                project_root = module_dir.parent.parent.parent
                data_dir = str((project_root / data_dir).resolve())
        
        start_date = start_date or self.config.dates.start_date
        end_date = end_date or self.config.dates.end_date
        
        # Get required symbols
        symbols = self.config.get_required_symbols()
        
        print(f"[DataPreprocessor] Loading {len(symbols)} symbols from {data_dir}")
        
        # Load daily data
        self.daily_data = load_multiple_symbols(
            symbols=symbols,
            data_dir=data_dir,
            start_date=start_date,
            end_date=end_date,
            required=True,
        )
        
        print(f"[DataPreprocessor] Loaded daily data for {len(self.daily_data)} symbols")
        
        # Determine week-end dates
        # Use earliest start and latest end across all symbols
        all_start_dates = [df.index.min() for df in self.daily_data.values()]
        all_end_dates = [df.index.max() for df in self.daily_data.values()]
        
        data_start = min(all_start_dates)
        data_end = max(all_end_dates)
        
        self.week_end_dates = get_week_end_dates(
            start=data_start.strftime("%Y-%m-%d"),
            end=data_end.strftime("%Y-%m-%d"),
        )
        
        print(f"[DataPreprocessor] Week-end dates: {len(self.week_end_dates)} weeks")
        
        # Aggregate to weekly
        self.weekly_data = aggregate_multiple_symbols_to_weekly(
            daily_data=self.daily_data,
            week_end_dates=self.week_end_dates,
            method="last_available",
        )
        
        print(f"[DataPreprocessor] Aggregated to weekly for {len(self.weekly_data)} symbols")
        
        # Align symbols
        self.weekly_data, self.common_dates = align_symbols_to_common_dates(
            data=self.weekly_data,
            fill_method="forward",
            max_fill_gaps=2,
        )
        
        print(f"[DataPreprocessor] Aligned to {len(self.common_dates)} common dates")
        print(f"[DataPreprocessor] Date range: {self.common_dates[0]} to {self.common_dates[-1]}")
    
    def get_data_as_of(
        self,
        as_of_date: Union[str, pd.Timestamp],
        lookback_periods: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get point-in-time data slice (WEEKLY data)
        
        Args:
            as_of_date: Cut-off date
            lookback_periods: Optional lookback window
        
        Returns:
            Dict mapping symbol → sliced DataFrame
        
        Raises:
            RuntimeError: If data not loaded yet
        """
        if self.weekly_data is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare() first.")
        
        return get_data_as_of_date(
            data=self.weekly_data,
            as_of_date=as_of_date,
            lookback_periods=lookback_periods,
            include_as_of_date=True,
        )
    
    def get_daily_data_as_of(
        self,
        as_of_date: Union[str, pd.Timestamp],
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.Series]:
        """
        Get point-in-time DAILY close prices (for Fast Risk-Off detection)
        
        This method returns daily close prices up to as_of_date for specified symbols.
        This is critical for Fast Risk-Off detection which requires daily data
        to detect rapid market shocks.
        
        Args:
            as_of_date: Cut-off date
            symbols: List of symbols to extract (e.g., ['^GSPC', '^VIX', 'QQQ'])
                    If None, returns all symbols
        
        Returns:
            Dict mapping symbol → daily close price Series
        
        Raises:
            RuntimeError: If daily data not loaded yet
        
        Examples:
            >>> preprocessor.load_and_prepare()
            >>> daily_prices = preprocessor.get_daily_data_as_of(
            ...     as_of_date='2024-02-01',
            ...     symbols=['^GSPC', '^VIX', 'QQQ']
            ... )
            >>> vix_daily = daily_prices['^VIX']
        """
        if self.daily_data is None:
            raise RuntimeError("Daily data not loaded. Call load_and_prepare() first.")
        
        as_of_ts = pd.Timestamp(as_of_date)
        
        # Get symbols to extract
        symbols_to_extract = symbols if symbols is not None else list(self.daily_data.keys())
        
        # Extract daily close prices
        daily_prices = {}
        for symbol in symbols_to_extract:
            if symbol not in self.daily_data:
                continue
            
            df = self.daily_data[symbol]
            # Filter up to as_of_date
            df_slice = df[df.index <= as_of_ts]
            
            if not df_slice.empty and 'close' in df_slice.columns:
                daily_prices[symbol] = df_slice['close']
        
        return daily_prices
    
    def has_sufficient_history(
        self,
        as_of_date: Union[str, pd.Timestamp],
        min_weeks: Optional[int] = None,
    ) -> bool:
        """
        Check if sufficient history exists as of date
        
        Args:
            as_of_date: Date to check
            min_weeks: Minimum weeks required (defaults to config setting)
        
        Returns:
            True if all symbols have sufficient history
        
        Raises:
            RuntimeError: If data not loaded yet
        """
        if self.weekly_data is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare() first.")
        
        min_weeks = min_weeks or self.config.history.minimum_history_weeks
        
        is_valid, _ = validate_sufficient_history(
            data=self.weekly_data,
            min_periods=min_weeks,
            as_of_date=as_of_date,
        )
        
        return is_valid
    
    def get_available_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get available date range after alignment
        
        Returns:
            (start_date, end_date)
        
        Raises:
            RuntimeError: If data not loaded yet
        """
        if self.common_dates is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare() first.")
        
        return self.common_dates[0], self.common_dates[-1]
    
    def get_weekly_returns(
        self,
        as_of_date: Optional[Union[str, pd.Timestamp]] = None,
        lookback_periods: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """
        Calculate weekly returns for all symbols
        
        Args:
            as_of_date: Optional cut-off date
            lookback_periods: Optional lookback window
        
        Returns:
            Dict mapping symbol → returns Series
        
        Raises:
            RuntimeError: If data not loaded yet
        """
        if self.weekly_data is None:
            raise RuntimeError("Data not loaded. Call load_and_prepare() first.")
        
        # Get data slice
        if as_of_date is not None:
            data = self.get_data_as_of(as_of_date, lookback_periods)
        else:
            data = self.weekly_data
        
        # Calculate returns
        returns = {}
        for symbol, df in data.items():
            returns[symbol] = df["close"].pct_change()
        
        return returns


if __name__ == "__main__":
    """Quick test of data preprocessor"""
    
    print("Testing DataPreprocessor module...")
    print("=" * 60)
    
    # Test with minimal setup
    from config_loader import load_config
    
    try:
        # Load config
        config = load_config("src/strategies/AdaptiveRotationConf_v1.2.1.yaml")
        print(f"[OK] Config loaded: {config.strategy.name}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config)
        print(f"[OK] Preprocessor initialized")
        
        # Load and prepare data
        print("\nLoading and preparing data...")
        preprocessor.load_and_prepare()
        
        # Get date range
        start, end = preprocessor.get_available_date_range()
        print(f"\n[OK] Data range: {start.date()} to {end.date()}")
        
        # Test point-in-time slice
        test_date = "2024-02-01"
        data_slice = preprocessor.get_data_as_of(test_date, lookback_periods=26)
        print(f"\n[OK] Point-in-time slice as of {test_date}:")
        for symbol in list(data_slice.keys())[:3]:
            df = data_slice[symbol]
            print(f"  - {symbol}: {len(df)} periods, last date {df.index[-1].date()}")
        
        # Check sufficient history
        has_history = preprocessor.has_sufficient_history(test_date)
        print(f"\n[OK] Sufficient history as of {test_date}: {has_history}")
        
        print(f"\n{'='*60}")
        print("[PASS] DataPreprocessor test complete!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
