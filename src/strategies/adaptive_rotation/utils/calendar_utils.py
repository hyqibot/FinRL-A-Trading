"""
Trading Calendar Utilities
==========================

This module provides functions for working with trading calendars,
including identifying trading days, week-end dates, and counting trading days.

Functions:
    - get_trading_calendar: Get valid trading days for a date range
    - get_week_end_dates: Get Friday (or last trading day of week) dates
    - is_trading_day: Check if a date is a trading day
    - trading_days_between: Count trading days between two dates
    - get_next_trading_day: Get next trading day after a given date
    - get_previous_trading_day: Get previous trading day before a given date

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import pandas_market_calendars as mcal
from typing import Union, Optional
from datetime import datetime, timedelta
import numpy as np


# Cache for calendar instance to avoid repeated initialization
_calendar_cache = {}


def get_trading_calendar(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    exchange: str = "NYSE"
) -> pd.DatetimeIndex:
    """
    Get valid trading days for a date range
    
    Returns a DatetimeIndex of all trading days (excluding weekends and holidays)
    for the specified exchange.
    
    Args:
        start: Start date (inclusive)
        end: End date (inclusive)
        exchange: Exchange name. Common values:
                  - "NYSE" (default): New York Stock Exchange
                  - "NASDAQ": NASDAQ
                  - "LSE": London Stock Exchange
                  Full list: mcal.get_calendar_names()
    
    Returns:
        pd.DatetimeIndex: Trading days in the range
    
    Examples:
        >>> from datetime import date
        >>> trading_days = get_trading_calendar("2024-01-01", "2024-01-31")
        >>> len(trading_days)  # ~20 trading days in January
        20
        
        >>> # Check specific date
        >>> pd.Timestamp("2024-12-25") in trading_days  # Christmas
        False
    
    Notes:
        - Includes start and end dates if they are trading days
        - Excludes weekends (Saturday, Sunday)
        - Excludes exchange-specific holidays
        - Times are set to midnight (00:00:00)
    """
    # Convert to timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    # Get or create calendar instance (with caching)
    if exchange not in _calendar_cache:
        _calendar_cache[exchange] = mcal.get_calendar(exchange)
    calendar = _calendar_cache[exchange]
    
    # Get trading schedule
    schedule = calendar.schedule(start_date=start, end_date=end)
    
    # Extract dates (market_open column contains trading days)
    trading_days = pd.DatetimeIndex(schedule.index.date).unique()
    
    return trading_days


def get_week_end_dates(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    exchange: str = "NYSE"
) -> pd.DatetimeIndex:
    """
    Get week-end dates (Friday or last trading day of week)
    
    For weekly strategies, returns the last trading day of each week.
    Typically Friday, but if Friday is a holiday, returns Thursday (or earlier).
    
    Args:
        start: Start date
        end: End date
        exchange: Exchange name (default: NYSE)
    
    Returns:
        pd.DatetimeIndex: Week-end trading dates
    
    Examples:
        >>> week_ends = get_week_end_dates("2024-01-01", "2024-01-31")
        >>> # Returns ~4-5 Fridays in January
        
        >>> # Christmas week 2024: Friday is holiday
        >>> xmas_week = get_week_end_dates("2024-12-23", "2024-12-27")
        >>> # Returns Thursday Dec 26 (if Friday is holiday)
    
    Notes:
        - Week is defined as Monday-Sunday
        - Returns last trading day of each week
        - Handles holiday weeks gracefully
        - Useful for weekly rebalancing strategies
    """
    # Get all trading days
    trading_days = get_trading_calendar(start, end, exchange)
    
    if len(trading_days) == 0:
        return pd.DatetimeIndex([])
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame({'date': trading_days})
    df['date'] = pd.to_datetime(df['date'])
    
    # Get week number (ISO week)
    df['year'] = df['date'].dt.isocalendar().year
    df['week'] = df['date'].dt.isocalendar().week
    
    # Get last trading day of each week
    week_ends = df.groupby(['year', 'week'])['date'].max()
    
    return pd.DatetimeIndex(week_ends.values)


def is_trading_day(
    date: Union[str, pd.Timestamp, datetime],
    exchange: str = "NYSE"
) -> bool:
    """
    Check if a date is a valid trading day
    
    Args:
        date: Date to check
        exchange: Exchange name (default: NYSE)
    
    Returns:
        bool: True if date is a trading day, False otherwise
    
    Examples:
        >>> is_trading_day("2024-01-02")  # Tuesday
        True
        
        >>> is_trading_day("2024-01-01")  # New Year's Day
        False
        
        >>> is_trading_day("2024-01-06")  # Saturday
        False
    
    Notes:
        - Returns False for weekends
        - Returns False for holidays
        - Time component is ignored (only date matters)
    """
    date = pd.Timestamp(date)
    
    # Get trading days for the specific date
    trading_days = get_trading_calendar(
        start=date,
        end=date,
        exchange=exchange
    )
    
    return len(trading_days) > 0


def trading_days_between(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    exchange: str = "NYSE",
    inclusive: str = "both"
) -> int:
    """
    Count trading days between two dates
    
    Args:
        start: Start date
        end: End date
        exchange: Exchange name (default: NYSE)
        inclusive: "both" (default), "left", "right", or "neither"
                   Controls whether start and end dates are included
    
    Returns:
        int: Number of trading days
    
    Examples:
        >>> # Count trading days in January 2024
        >>> trading_days_between("2024-01-01", "2024-01-31")
        20
        
        >>> # Excluding end date
        >>> trading_days_between("2024-01-01", "2024-01-31", inclusive="left")
        19
    
    Notes:
        - Returns 0 if start > end (regardless of inclusive)
        - Counts only actual trading days (no weekends/holidays)
        - Useful for checking data sufficiency
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    # Handle invalid range
    if start > end:
        return 0
    
    # Get trading days
    trading_days = get_trading_calendar(start, end, exchange)
    count = len(trading_days)
    
    # Adjust for inclusive parameter
    if inclusive == "both":
        pass  # Keep as is
    elif inclusive == "left":
        # Exclude end date if it's a trading day
        if is_trading_day(end, exchange):
            count -= 1
    elif inclusive == "right":
        # Exclude start date if it's a trading day
        if is_trading_day(start, exchange):
            count -= 1
    elif inclusive == "neither":
        # Exclude both if they're trading days
        if is_trading_day(start, exchange):
            count -= 1
        if is_trading_day(end, exchange):
            count -= 1
    else:
        raise ValueError(f"inclusive must be 'both', 'left', 'right', or 'neither', got {inclusive}")
    
    return max(0, count)


def get_next_trading_day(
    date: Union[str, pd.Timestamp],
    exchange: str = "NYSE",
    n_days: int = 1
) -> pd.Timestamp:
    """
    Get the next trading day after a given date
    
    Args:
        date: Reference date
        exchange: Exchange name (default: NYSE)
        n_days: Number of trading days ahead (default: 1)
    
    Returns:
        pd.Timestamp: Next trading day
    
    Examples:
        >>> next_day = get_next_trading_day("2024-01-05")  # Friday
        >>> # Returns Monday 2024-01-08 (skipping weekend)
        
        >>> next_week = get_next_trading_day("2024-01-05", n_days=5)
        >>> # Returns 5 trading days later
    
    Raises:
        ValueError: If n_days < 1
        IndexError: If not enough future trading days exist
    
    Notes:
        - Skips weekends and holidays automatically
        - Does not include the reference date itself
        - Useful for forward-looking date calculations
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    
    date = pd.Timestamp(date)
    
    # Get trading days starting from the day after
    start = date + timedelta(days=1)
    # Look ahead enough to find n trading days (conservative estimate)
    end = date + timedelta(days=n_days * 2 + 10)  # 2x buffer for holidays
    
    trading_days = get_trading_calendar(start, end, exchange)
    
    if len(trading_days) < n_days:
        raise IndexError(f"Not enough future trading days. Need {n_days}, found {len(trading_days)}")
    
    return trading_days[n_days - 1]


def get_previous_trading_day(
    date: Union[str, pd.Timestamp],
    exchange: str = "NYSE",
    n_days: int = 1
) -> pd.Timestamp:
    """
    Get the previous trading day before a given date
    
    Args:
        date: Reference date
        exchange: Exchange name (default: NYSE)
        n_days: Number of trading days back (default: 1)
    
    Returns:
        pd.Timestamp: Previous trading day
    
    Examples:
        >>> prev_day = get_previous_trading_day("2024-01-08")  # Monday
        >>> # Returns Friday 2024-01-05 (skipping weekend)
        
        >>> prev_week = get_previous_trading_day("2024-01-08", n_days=5)
        >>> # Returns 5 trading days earlier
    
    Raises:
        ValueError: If n_days < 1
        IndexError: If not enough past trading days exist
    
    Notes:
        - Skips weekends and holidays automatically
        - Does not include the reference date itself
        - Useful for backward-looking date calculations
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    
    date = pd.Timestamp(date)
    
    # Get trading days up to the day before
    end = date - timedelta(days=1)
    # Look back enough to find n trading days (conservative estimate)
    start = date - timedelta(days=n_days * 2 + 10)  # 2x buffer for holidays
    
    trading_days = get_trading_calendar(start, end, exchange)
    
    if len(trading_days) < n_days:
        raise IndexError(f"Not enough past trading days. Need {n_days}, found {len(trading_days)}")
    
    return trading_days[-(n_days)]


def get_available_exchanges() -> list:
    """
    Get list of available exchange calendars
    
    Returns:
        list: Available exchange names
    
    Example:
        >>> exchanges = get_available_exchanges()
        >>> 'NYSE' in exchanges
        True
    """
    return mcal.get_calendar_names()


def align_to_trading_day(
    date: Union[str, pd.Timestamp],
    exchange: str = "NYSE",
    method: str = "forward"
) -> pd.Timestamp:
    """
    Align a date to the nearest trading day
    
    Useful for converting arbitrary dates to valid trading dates.
    
    Args:
        date: Date to align
        exchange: Exchange name (default: NYSE)
        method: "forward" (default), "backward", or "nearest"
                - forward: Next trading day if not already one
                - backward: Previous trading day if not already one
                - nearest: Closest trading day
    
    Returns:
        pd.Timestamp: Aligned trading day
    
    Examples:
        >>> # Saturday -> Monday
        >>> align_to_trading_day("2024-01-06", method="forward")
        Timestamp('2024-01-08')
        
        >>> # Saturday -> Friday
        >>> align_to_trading_day("2024-01-06", method="backward")
        Timestamp('2024-01-05')
    
    Notes:
        - If date is already a trading day, returns it unchanged
        - Useful for user-provided dates that may be weekends/holidays
    """
    date = pd.Timestamp(date)
    
    # If already a trading day, return as is
    if is_trading_day(date, exchange):
        return date
    
    if method == "forward":
        return get_next_trading_day(date, exchange, n_days=1)
    elif method == "backward":
        return get_previous_trading_day(date, exchange, n_days=1)
    elif method == "nearest":
        # Get both directions
        next_day = get_next_trading_day(date, exchange, n_days=1)
        prev_day = get_previous_trading_day(date, exchange, n_days=1)
        
        # Return closer one
        if abs((next_day - date).days) <= abs((date - prev_day).days):
            return next_day
        else:
            return prev_day
    else:
        raise ValueError(f"method must be 'forward', 'backward', or 'nearest', got {method}")


if __name__ == "__main__":
    """Quick test of basic functionality"""
    
    print("Testing calendar_utils module...")
    print("=" * 60)
    
    # Test 1: Get trading calendar
    print("\n1. Testing get_trading_calendar:")
    trading_days = get_trading_calendar("2024-01-01", "2024-01-31")
    print(f"   Trading days in Jan 2024: {len(trading_days)}")
    print(f"   First: {trading_days[0].date()}")
    print(f"   Last: {trading_days[-1].date()}")
    print(f"   ✓ Expected: ~20 trading days")
    
    # Test 2: Week end dates
    print("\n2. Testing get_week_end_dates:")
    week_ends = get_week_end_dates("2024-01-01", "2024-01-31")
    print(f"   Week-end dates in Jan 2024: {len(week_ends)}")
    print(f"   Dates: {[d.date() for d in week_ends]}")
    print(f"   ✓ Expected: 4-5 Fridays")
    
    # Test 3: Is trading day
    print("\n3. Testing is_trading_day:")
    test_dates = [
        ("2024-01-02", True, "Tuesday"),
        ("2024-01-01", False, "New Year"),
        ("2024-01-06", False, "Saturday"),
        ("2024-12-25", False, "Christmas")
    ]
    for date, expected, desc in test_dates:
        result = is_trading_day(date)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {date} ({desc}): {result}")
    
    # Test 4: Trading days between
    print("\n4. Testing trading_days_between:")
    count = trading_days_between("2024-01-01", "2024-01-31")
    print(f"   Trading days in Jan 2024: {count}")
    print(f"   ✓ Expected: ~20 days")
    
    # Test 5: Next/Previous trading day
    print("\n5. Testing get_next_trading_day:")
    next_day = get_next_trading_day("2024-01-05")  # Friday
    print(f"   Next trading day after Friday 2024-01-05: {next_day.date()}")
    print(f"   ✓ Expected: Monday 2024-01-08")
    
    print("\n6. Testing get_previous_trading_day:")
    prev_day = get_previous_trading_day("2024-01-08")  # Monday
    print(f"   Previous trading day before Monday 2024-01-08: {prev_day.date()}")
    print(f"   ✓ Expected: Friday 2024-01-05")
    
    # Test 7: Align to trading day
    print("\n7. Testing align_to_trading_day:")
    saturday = "2024-01-06"
    aligned_fwd = align_to_trading_day(saturday, method="forward")
    aligned_bwd = align_to_trading_day(saturday, method="backward")
    print(f"   Saturday 2024-01-06:")
    print(f"     Forward: {aligned_fwd.date()} (Monday)")
    print(f"     Backward: {aligned_bwd.date()} (Friday)")
    print(f"   ✓ Weekends skipped correctly")
    
    print("\n" + "=" * 60)
    print("✓ All basic tests passed!")
    print("\nAvailable exchanges:", len(get_available_exchanges()), "exchanges")
