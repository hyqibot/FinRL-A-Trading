"""
Group Strength Computation
===========================

Computes strength metrics for asset groups to determine which groups
should be activated in the portfolio.

Key Metrics:
- Excess return vs benchmark (QQQ)
- Robust Information Ratio (using MAD)
- Group ranking

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .utils.robust_stats import compute_information_ratio
from .config_loader import AdaptiveRotationConfig


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GroupStrengthMetrics:
    """Strength metrics for a single group"""
    group_name: str
    excess_return: float  # vs benchmark
    information_ratio: float  # robust IR
    is_valid: bool  # Whether sufficient data exists
    rank: Optional[int] = None  # Rank among all groups (1=strongest)
    
    # Raw metrics for debugging
    group_return: float = 0.0
    benchmark_return: float = 0.0
    lookback_periods: int = 0


@dataclass
class GroupStrengthResult:
    """Complete group strength analysis result"""
    groups: Dict[str, GroupStrengthMetrics]  # group_name → metrics
    ranked_groups: List[str]  # Groups sorted by strength (strongest first)
    active_groups: List[str]  # Top N groups to activate
    as_of_date: pd.Timestamp
    benchmark_symbol: str
    
    def get_group_metrics(self, group_name: str) -> Optional[GroupStrengthMetrics]:
        """Get metrics for a specific group"""
        return self.groups.get(group_name)
    
    def get_top_n_groups(self, n: int) -> List[str]:
        """Get top N strongest groups"""
        return self.ranked_groups[:n]


# ============================================================================
# Group Return Computation
# ============================================================================

def compute_group_returns(
    prices: Dict[str, pd.Series],
    group_symbols: List[str],
    lookback_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute equal-weighted group returns
    
    Args:
        prices: Dict mapping symbol → price series
        group_symbols: List of symbols in this group
        lookback_periods: Optional lookback window
    
    Returns:
        Series of group returns
    
    Note:
        Returns are computed as simple average of constituent returns.
        Missing data is handled by computing returns only from available symbols.
    """
    # Filter to group symbols
    group_prices = {sym: prices[sym] for sym in group_symbols if sym in prices}
    
    if not group_prices:
        return pd.Series(dtype=float)
    
    # Apply lookback
    if lookback_periods is not None:
        group_prices = {
            sym: series.tail(lookback_periods)
            for sym, series in group_prices.items()
        }
    
    # Compute returns for each symbol
    returns_dict = {}
    for sym, series in group_prices.items():
        returns_dict[sym] = series.pct_change()
    
    # Convert to DataFrame for easy averaging
    returns_df = pd.DataFrame(returns_dict)
    
    # Equal-weighted average (ignoring NaN)
    group_returns = returns_df.mean(axis=1)
    
    return group_returns


def compute_excess_returns(
    group_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """
    Compute excess returns vs benchmark
    
    Args:
        group_returns: Group return series
        benchmark_returns: Benchmark return series
    
    Returns:
        Series of excess returns (group - benchmark)
    
    Note:
        Returns are aligned by index. If dates don't match exactly,
        only common dates are used.
    """
    # Align indices
    common_dates = group_returns.index.intersection(benchmark_returns.index)
    
    if len(common_dates) == 0:
        return pd.Series(dtype=float)
    
    excess = group_returns.loc[common_dates] - benchmark_returns.loc[common_dates]
    
    return excess


# ============================================================================
# Group Strength Computation
# ============================================================================

def compute_group_strength(
    prices: Dict[str, pd.Series],
    group_name: str,
    group_symbols: List[str],
    benchmark_symbol: str,
    lookback_periods: int,
    robust: bool = True,
) -> GroupStrengthMetrics:
    """
    Compute strength metrics for a single group
    
    Args:
        prices: Dict mapping symbol → price series
        group_name: Name of the group
        group_symbols: List of symbols in this group
        benchmark_symbol: Benchmark symbol (e.g., "QQQ")
        lookback_periods: Lookback window (in periods)
        robust: Whether to use robust IR (MAD-based)
    
    Returns:
        GroupStrengthMetrics object
    
    Examples:
        >>> metrics = compute_group_strength(
        ...     prices, "group_a", ["AAPL", "MSFT"], "QQQ", 12
        ... )
        >>> print(f"IR: {metrics.information_ratio:.2f}")
    """
    # Check if benchmark exists
    if benchmark_symbol not in prices:
        return GroupStrengthMetrics(
            group_name=group_name,
            excess_return=0.0,
            information_ratio=0.0,
            is_valid=False,
            lookback_periods=0,
        )
    
    # Compute group returns
    group_returns = compute_group_returns(prices, group_symbols, lookback_periods)
    
    # Check if sufficient data
    if len(group_returns) < lookback_periods or group_returns.isna().all():
        return GroupStrengthMetrics(
            group_name=group_name,
            excess_return=0.0,
            information_ratio=0.0,
            is_valid=False,
            lookback_periods=len(group_returns),
        )
    
    # Get benchmark returns
    benchmark_series = prices[benchmark_symbol]
    if lookback_periods is not None:
        benchmark_series = benchmark_series.tail(lookback_periods)
    
    benchmark_returns = benchmark_series.pct_change()
    
    # Compute excess returns
    excess_returns = compute_excess_returns(group_returns, benchmark_returns)
    
    # Check if sufficient data after alignment
    valid_excess = excess_returns.dropna()
    if len(valid_excess) < lookback_periods * 0.8:  # Require 80% data availability
        return GroupStrengthMetrics(
            group_name=group_name,
            excess_return=0.0,
            information_ratio=0.0,
            is_valid=False,
            lookback_periods=len(valid_excess),
        )
    
    # Compute Information Ratio
    # IR = mean(excess_return) / std(excess_return)
    # For robust version, use MAD instead of std
    ir = compute_information_ratio(
        returns=group_returns.loc[valid_excess.index],
        benchmark_returns=benchmark_returns.loc[valid_excess.index],
        lookback=len(valid_excess),
        robust=robust,
    )
    
    # Compute cumulative returns
    group_cum_return = (1 + group_returns.loc[valid_excess.index]).prod() - 1
    benchmark_cum_return = (1 + benchmark_returns.loc[valid_excess.index]).prod() - 1
    excess_cum_return = group_cum_return - benchmark_cum_return
    
    return GroupStrengthMetrics(
        group_name=group_name,
        excess_return=float(excess_cum_return),
        information_ratio=float(ir) if not pd.isna(ir) else 0.0,
        is_valid=True,
        group_return=float(group_cum_return),
        benchmark_return=float(benchmark_cum_return),
        lookback_periods=len(valid_excess),
    )


def rank_groups_by_strength(
    group_metrics: Dict[str, GroupStrengthMetrics],
    ranking_metric: str = "information_ratio",
) -> List[str]:
    """
    Rank groups by strength metric
    
    Args:
        group_metrics: Dict mapping group_name → metrics
        ranking_metric: Metric to rank by ("information_ratio" or "excess_return")
    
    Returns:
        List of group names sorted by strength (strongest first)
    
    Examples:
        >>> ranked = rank_groups_by_strength(metrics, "information_ratio")
        >>> print(f"Strongest group: {ranked[0]}")
    """
    # Filter to valid groups only
    valid_groups = {
        name: metrics
        for name, metrics in group_metrics.items()
        if metrics.is_valid
    }
    
    if not valid_groups:
        return []
    
    # Sort by metric (descending)
    if ranking_metric == "information_ratio":
        sorted_groups = sorted(
            valid_groups.items(),
            key=lambda x: x[1].information_ratio,
            reverse=True
        )
    elif ranking_metric == "excess_return":
        sorted_groups = sorted(
            valid_groups.items(),
            key=lambda x: x[1].excess_return,
            reverse=True
        )
    else:
        raise ValueError(f"Invalid ranking_metric: {ranking_metric}")
    
    # Extract names
    ranked_names = [name for name, _ in sorted_groups]
    
    # Assign ranks
    for rank, name in enumerate(ranked_names, start=1):
        group_metrics[name].rank = rank
    
    return ranked_names


def select_active_groups(
    ranked_groups: List[str],
    max_active_groups: int,
    group_metrics: Dict[str, GroupStrengthMetrics],
    trend_filter: bool = True,
) -> List[str]:
    """
    Select which groups to activate
    
    Args:
        ranked_groups: Groups sorted by strength
        max_active_groups: Maximum number of groups to activate
        group_metrics: Dict mapping group_name → metrics
        trend_filter: If True, only activate groups with positive excess return
    
    Returns:
        List of groups to activate
    
    Examples:
        >>> active = select_active_groups(ranked, max_groups=2, metrics)
    """
    active_groups = []
    
    for group_name in ranked_groups:
        metrics = group_metrics[group_name]
        
        # Apply trend filter if enabled
        if trend_filter and metrics.excess_return <= 0:
            continue
        
        active_groups.append(group_name)
        
        # Stop if we've reached max
        if len(active_groups) >= max_active_groups:
            break
    
    return active_groups


# ============================================================================
# High-Level API
# ============================================================================

def analyze_group_strength(
    prices: Dict[str, pd.Series],
    config: AdaptiveRotationConfig,
    as_of_date: pd.Timestamp,
) -> GroupStrengthResult:
    """
    Analyze strength of all asset groups
    
    Args:
        prices: Dict mapping symbol → price series (should be point-in-time)
        config: Strategy configuration
        as_of_date: Current decision date
    
    Returns:
        GroupStrengthResult object
    
    Examples:
        >>> result = analyze_group_strength(weekly_prices, config, date)
        >>> 
        >>> # Get strongest groups
        >>> print(f"Active groups: {result.active_groups}")
        >>> 
        >>> # Get metrics for a specific group
        >>> metrics = result.get_group_metrics("group_a_growth_tech")
        >>> print(f"IR: {metrics.information_ratio:.2f}")
    """
    # Get configuration
    benchmark_symbol = config.benchmark.excess_return_benchmark
    lookback_periods = config.group_strength.lookback_weeks
    ranking_metric = config.group_strength.metric
    trend_filter = config.group_strength.trend_filter
    max_active_groups = config.portfolio.max_active_groups
    
    # Compute metrics for each group
    group_metrics = {}
    
    for group_name, group_config in config.asset_groups.items():
        metrics = compute_group_strength(
            prices=prices,
            group_name=group_name,
            group_symbols=group_config.symbols,
            benchmark_symbol=benchmark_symbol,
            lookback_periods=lookback_periods,
            robust=(config.ranking.robust if hasattr(config, 'ranking') else True),
        )
        
        group_metrics[group_name] = metrics
    
    # Rank groups
    # Map config metric names to internal names
    if ranking_metric == "risk_adjusted_return":
        ranking_key = "information_ratio"
    elif ranking_metric == "return":
        ranking_key = "excess_return"
    else:
        ranking_key = "information_ratio"
    
    ranked_groups = rank_groups_by_strength(group_metrics, ranking_key)
    
    # Select active groups
    active_groups = select_active_groups(
        ranked_groups,
        max_active_groups,
        group_metrics,
        trend_filter,
    )
    
    return GroupStrengthResult(
        groups=group_metrics,
        ranked_groups=ranked_groups,
        active_groups=active_groups,
        as_of_date=as_of_date,
        benchmark_symbol=benchmark_symbol,
    )


if __name__ == "__main__":
    """Quick test of group strength computation"""
    
    print("Testing Group Strength Computation")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="W-FRI")
    
    # Simulate different group behaviors
    prices = {}
    
    # Strong tech stocks
    for sym in ["AAPL", "MSFT", "NVDA"]:
        prices[sym] = pd.Series(
            100 + np.arange(len(dates)) * 2 + np.random.randn(len(dates)) * 5,
            index=dates
        )
    
    # Weak commodities
    for sym in ["GLD", "SLV"]:
        prices[sym] = pd.Series(
            100 + np.arange(len(dates)) * 0.5 + np.random.randn(len(dates)) * 5,
            index=dates
        )
    
    # Benchmark
    prices["QQQ"] = pd.Series(
        100 + np.arange(len(dates)) * 1.5 + np.random.randn(len(dates)) * 5,
        index=dates
    )
    
    # Load config
    from config_loader import load_config
    config = load_config("src/strategies/AdaptiveRotationConf_v1.2.1.yaml")
    
    # Test
    test_date = pd.Timestamp("2024-06-30")
    
    # Get data as of date
    prices_as_of = {
        sym: series[series.index <= test_date]
        for sym, series in prices.items()
    }
    
    result = analyze_group_strength(prices_as_of, config, test_date)
    
    print(f"\n[Test Date: {test_date.date()}]")
    print(f"\nGroup Rankings:")
    for rank, group_name in enumerate(result.ranked_groups, 1):
        metrics = result.groups[group_name]
        print(f"  {rank}. {group_name}")
        print(f"     IR: {metrics.information_ratio:.3f}")
        print(f"     Excess Return: {metrics.excess_return:.2%}")
        print(f"     Valid: {metrics.is_valid}")
    
    print(f"\nActive Groups (max={config.portfolio.max_active_groups}):")
    for group in result.active_groups:
        print(f"  - {group}")
    
    print(f"\n{'='*60}")
    print("[PASS] Group strength test complete!")
