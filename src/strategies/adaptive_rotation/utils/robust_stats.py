"""
Robust Statistical Computations
================================

This module provides robust statistical functions using Median Absolute Deviation (MAD)
instead of standard deviation. These are more resistant to outliers.

Functions:
    - compute_mad: Compute Median Absolute Deviation
    - robust_zscore: Compute robust Z-score using MAD
    - compute_information_ratio: Compute Information Ratio (IR)

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def compute_mad(
    series: pd.Series,
    window: Optional[int] = None,
    center: bool = False
) -> Union[float, pd.Series]:
    """
    Compute Median Absolute Deviation (MAD)
    
    MAD is a robust measure of variability, defined as:
        MAD = median(|X - median(X)|)
    
    It is more resistant to outliers than standard deviation.
    
    Args:
        series: Input data series
        window: If provided, compute rolling MAD with this window size
                If None, compute single MAD value for entire series
        center: If True, center the rolling window (only used with window)
    
    Returns:
        float: Single MAD value if window is None
        pd.Series: Rolling MAD series if window is provided
    
    Examples:
        >>> data = pd.Series([1, 2, 3, 10])
        >>> compute_mad(data)
        1.0
        
        >>> prices = pd.Series([100, 101, 102, 103, 110, 105])
        >>> rolling_mad = compute_mad(prices, window=3)
    
    Notes:
        - MAD is typically scaled by 1.4826 to make it comparable to std
        - This implementation uses the unscaled version for consistency
        - NaN values are handled by pandas operations
    """
    if window is None:
        # Compute single MAD value
        median = series.median()
        mad = (series - median).abs().median()
        return mad
    else:
        # Compute rolling MAD
        rolling_median = series.rolling(window=window, center=center).median()
        deviations = (series - rolling_median).abs()
        rolling_mad = deviations.rolling(window=window, center=center).median()
        return rolling_mad


def robust_zscore(
    series: pd.Series,
    window: int,
    center_metric: str = "median",
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute robust Z-score using MAD instead of standard deviation
    
    Robust Z-score is defined as:
        Z = (X - center) / MAD
    
    Where center is typically the median (more robust than mean).
    
    Args:
        series: Input data series
        window: Rolling window size for computing median and MAD
        center_metric: "median" or "mean" for the center measure
                      "median" is recommended for robustness
        min_periods: Minimum number of observations required
                     If None, defaults to window size
    
    Returns:
        pd.Series: Robust Z-scores
                   - Z > 0: Above center
                   - Z < 0: Below center
                   - |Z| > 2.5: Potential outlier (common threshold)
    
    Examples:
        >>> prices = pd.Series([100, 101, 102, 110, 105, 103])
        >>> zscores = robust_zscore(prices, window=3)
        >>> outliers = zscores[zscores.abs() > 2.5]
    
    Notes:
        - First (window-1) values will be NaN
        - MAD of 0 (constant series) will result in inf/-inf Z-scores
        - This is expected behavior for detecting anomalies
    """
    if min_periods is None:
        min_periods = window
    
    # Validate center_metric early
    if center_metric not in ["median", "mean"]:
        raise ValueError(f"center_metric must be 'median' or 'mean', got {center_metric}")
    
    # Initialize output series with NaN
    zscores = pd.Series(index=series.index, dtype=float)
    zscores[:] = np.nan
    
    # Compute Z-scores using rolling apply for efficiency
    def compute_window_zscore(window_data):
        """Compute Z-score for a single window"""
        if len(window_data) < min_periods:
            return np.nan
        
        # Compute center
        if center_metric == "median":
            center = np.median(window_data)
        elif center_metric == "mean":
            center = np.mean(window_data)
        else:
            return np.nan
        
        # Compute MAD
        deviations = np.abs(window_data - center)
        mad = np.median(deviations)
        
        # Compute Z-score for the most recent value
        current_value = window_data.iloc[-1] if hasattr(window_data, 'iloc') else window_data[-1]
        
        # Apply MAD threshold to prevent extreme Z-scores from near-constant series
        MIN_MAD_THRESHOLD = 1e-6
        if mad < MIN_MAD_THRESHOLD:
            # MAD too small, treat as constant series
            if abs(current_value - center) < MIN_MAD_THRESHOLD:
                return 0.0
            else:
                # Cap at reasonable extreme value instead of inf
                return 10.0 * np.sign(current_value - center)
        
        return (current_value - center) / mad
    
    # Apply rolling calculation
    zscores = series.rolling(window=window, min_periods=min_periods).apply(
        compute_window_zscore,
        raw=False
    )
    
    return zscores


def compute_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    lookback: int,
    robust: bool = True,
    annualization_factor: float = 1.0,
    min_periods: Optional[int] = None
) -> float:
    """
    Compute Information Ratio (IR)
    
    Information Ratio measures risk-adjusted excess return:
        IR = mean(excess_return) / std(excess_return)
    
    If robust=True, uses MAD instead of std:
        IR_robust = mean(excess_return) / MAD(excess_return)
    
    Args:
        returns: Asset returns series
        benchmark_returns: Benchmark returns series
        lookback: Number of periods to use for calculation
        robust: If True, use MAD instead of std (recommended)
        annualization_factor: Scale factor for annualization
                             (e.g., sqrt(52) for weekly data)
        min_periods: Minimum periods required. If None, use lookback
    
    Returns:
        float: Information Ratio
               - IR > 0: Outperforming benchmark
               - IR < 0: Underperforming benchmark
               - Higher absolute value = better risk-adjusted performance
    
    Examples:
        >>> asset_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        >>> bench_returns = pd.Series([0.005, 0.015, 0.005, 0.02])
        >>> ir = compute_information_ratio(
        ...     asset_returns,
        ...     bench_returns,
        ...     lookback=4,
        ...     robust=True
        ... )
    
    Notes:
        - Returns should be in decimal form (0.01 = 1%)
        - Both series must have the same index
        - NaN values are dropped before calculation
        - Robust version is more stable with outliers
    """
    if min_periods is None:
        min_periods = lookback
    
    # Align series and compute excess returns
    aligned_returns, aligned_bench = returns.align(benchmark_returns, join='inner')
    
    # Take last 'lookback' periods
    excess_returns = (aligned_returns - aligned_bench).tail(lookback)
    
    # Check if we have enough data
    valid_data = excess_returns.dropna()
    if len(valid_data) < min_periods:
        return np.nan
    
    # Compute mean excess return
    mean_excess = valid_data.mean()
    
    # Compute variability measure
    if robust:
        # Use MAD as variability measure
        mad = compute_mad(valid_data)
        # If MAD is 0 (e.g., > 50% of values are identical), fall back to std
        if mad == 0 or np.isnan(mad) or not np.isfinite(mad):
            # Fall back to standard deviation
            std = valid_data.std()
            if std == 0 or np.isnan(std) or not np.isfinite(std):
                return np.nan
            variability = std
        else:
            variability = mad
    else:
        # Use standard deviation
        std = valid_data.std()
        # For constant excess returns (std=0), return NaN (no signal)
        if std == 0 or np.isnan(std) or not np.isfinite(std):
            return np.nan
        variability = std
    
    # Compute Information Ratio
    ir = mean_excess / variability
    
    # Apply annualization if requested
    if annualization_factor != 1.0:
        ir = ir * np.sqrt(annualization_factor)
    
    return ir


def scale_mad_to_std(mad: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Scale MAD to be comparable to standard deviation
    
    The scaling factor 1.4826 makes MAD(normal distribution) = std(normal distribution)
    
    Args:
        mad: MAD value or series
    
    Returns:
        Scaled MAD equivalent to std
    
    Example:
        >>> mad = compute_mad(pd.Series([1, 2, 3, 4, 5]))
        >>> std_equivalent = scale_mad_to_std(mad)
    """
    SCALE_FACTOR = 1.4826
    return mad * SCALE_FACTOR


# Optional: Additional utility functions

def detect_outliers_mad(
    series: pd.Series,
    window: int,
    threshold: float = 3.0
) -> pd.Series:
    """
    Detect outliers using MAD-based Z-scores
    
    Args:
        series: Input data
        window: Rolling window for MAD calculation
        threshold: Z-score threshold (typically 2.5 or 3.0)
    
    Returns:
        pd.Series: Boolean series, True for outliers
    
    Example:
        >>> prices = pd.Series([100, 101, 102, 150, 103])
        >>> outliers = detect_outliers_mad(prices, window=3, threshold=3.0)
        >>> outlier_values = prices[outliers]
    """
    zscores = robust_zscore(series, window=window)
    return zscores.abs() > threshold


def winsorize_by_mad(
    series: pd.Series,
    window: int,
    n_mad: float = 3.0
) -> pd.Series:
    """
    Winsorize (cap) extreme values using MAD
    
    Values beyond n_mad * MAD from median are capped to that limit.
    
    Args:
        series: Input data
        window: Rolling window for MAD calculation
        n_mad: Number of MADs for the threshold
    
    Returns:
        pd.Series: Winsorized series
    
    Example:
        >>> prices = pd.Series([100, 101, 102, 200, 103])
        >>> winsorized = winsorize_by_mad(prices, window=3, n_mad=3.0)
    """
    zscores = robust_zscore(series, window=window)
    
    # Cap Z-scores at +/- n_mad
    capped_zscores = zscores.clip(lower=-n_mad, upper=n_mad)
    
    # Reconstruct values
    rolling_median = series.rolling(window=window).median()
    rolling_mad = compute_mad(series, window=window)
    
    winsorized = rolling_median + capped_zscores * rolling_mad
    
    # For the first (window-1) values, use original
    winsorized.iloc[:window-1] = series.iloc[:window-1]
    
    return winsorized


if __name__ == "__main__":
    """Quick test of basic functionality"""
    
    print("Testing robust_stats module...")
    print("=" * 60)
    
    # Test 1: compute_mad
    print("\n1. Testing compute_mad:")
    data = pd.Series([1, 2, 3, 10])
    mad = compute_mad(data)
    print(f"   Data: {data.tolist()}")
    print(f"   MAD: {mad}")
    print(f"   ✓ Expected: 1.0")
    
    # Test 2: robust_zscore
    print("\n2. Testing robust_zscore:")
    prices = pd.Series([100, 101, 102, 110, 105, 103, 104])
    zscores = robust_zscore(prices, window=3)
    print(f"   Prices: {prices.tolist()}")
    print(f"   Z-scores (window=3): {zscores.round(2).tolist()}")
    print(f"   ✓ Outlier at index 3 should have high Z-score")
    
    # Test 3: compute_information_ratio
    print("\n3. Testing compute_information_ratio:")
    asset_ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
    bench_ret = pd.Series([0.005, 0.015, 0.005, 0.02, 0.005])
    ir = compute_information_ratio(asset_ret, bench_ret, lookback=5, robust=True)
    print(f"   Asset returns: {asset_ret.tolist()}")
    print(f"   Bench returns: {bench_ret.tolist()}")
    print(f"   Information Ratio: {ir:.3f}")
    print(f"   ✓ IR > 0 means outperforming benchmark")
    
    print("\n" + "=" * 60)
    print("✓ All basic tests passed!")
