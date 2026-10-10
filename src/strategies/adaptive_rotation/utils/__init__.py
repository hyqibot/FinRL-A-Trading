"""
Utility modules for Adaptive Rotation Strategy
"""

from .robust_stats import (
    compute_mad,
    robust_zscore,
    compute_information_ratio,
    scale_mad_to_std,
    detect_outliers_mad,
    winsorize_by_mad
)

from .calendar_utils import (
    get_trading_calendar,
    get_week_end_dates,
    is_trading_day,
    trading_days_between,
    get_next_trading_day,
    get_previous_trading_day,
    get_available_exchanges,
    align_to_trading_day
)

__all__ = [
    # Robust stats
    "compute_mad",
    "robust_zscore",
    "compute_information_ratio",
    "scale_mad_to_std",
    "detect_outliers_mad",
    "winsorize_by_mad",
    # Calendar utils
    "get_trading_calendar",
    "get_week_end_dates",
    "is_trading_day",
    "trading_days_between",
    "get_next_trading_day",
    "get_previous_trading_day",
    "get_available_exchanges",
    "align_to_trading_day",
]
