"""
Trading Calendar Module
=======================

Provides functions to get trading days and check if dates are trading days.
Uses pandas_market_calendars or exchange_calendars if available, with fallback to basic logic.
"""

import logging
from typing import List, Set
from datetime import datetime, date
import pandas as pd
from functools import lru_cache
import tzlocal

logger = logging.getLogger(__name__)

# Try to import market calendar libraries
import pandas_market_calendars as mcal
_calendar_lib = 'pandas_market_calendars'
_calendar_lib = None
_nyse_calendar = None

try:
    
    _nyse_calendar = mcal.get_calendar('NYSE')
    logger.info("Using pandas_market_calendars for trading calendar")
except ImportError:
    try:
        import exchange_calendars as xcals
        _calendar_lib = 'exchange_calendars'
        _nyse_calendar = xcals.get_calendar('XNYS')  # NYSE calendar
        logger.info("Using exchange_calendars for trading calendar")
    except ImportError:
        logger.warning("Neither pandas_market_calendars nor exchange_calendars found. Using basic calendar logic.")
        logger.warning("Install with: pip install pandas-market-calendars  OR  pip install exchange-calendars")


@lru_cache(maxsize=32)
def _get_calendar_cached(exchange: str):
    return mcal.get_calendar(exchange)


@lru_cache(maxsize=256)
def _cached_trading_days(exchange: str, start_date: str, end_date: str) -> tuple:
    cal = _get_calendar_cached(exchange)
    # INSERT_YOUR_CODE
    # Get current timezone name as tz parameter for calendar schedule
    tzinfo = None
    try:
        # pandas >= 1.1: dt.now().astimezone().tzinfo.zone (linux/mac), or tzlocal.get_localzone_name() if installed
        tzinfo = tzlocal.get_localzone_name()
    except Exception:
        try:
            # Fallback to .tzname(), not always ideal
            tzinfo = datetime.now().astimezone().tzname()
        except Exception:
            tzinfo = "UTC"
    schedule = cal.schedule(start_date=start_date, end_date=end_date, tz=tzinfo)
    return tuple(schedule.index.strftime('%Y-%m-%d').tolist())


def get_trading_days(start_date: str, end_date: str, exchange: str = 'NYSE') -> pd.DatetimeIndex:
    """
    Get trading days between start_date and end_date.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        exchange: Exchange name (default: NYSE)
        
    Returns:
        DatetimeIndex of trading days
    """
    # Use cached trading days for repeated calls with same parameters
    days = _cached_trading_days(exchange, start_date, end_date)
    return pd.DatetimeIndex(pd.to_datetime(list(days)))
    

def is_trading_day(check_date: str, exchange: str = 'NYSE') -> bool:
    """
    Check if a date is a trading day.
    
    Args:
        check_date: Date to check (YYYY-MM-DD)
        exchange: Exchange name (default: NYSE)
        
    Returns:
        True if it's a trading day, False otherwise
    """
    # Reuse cached trading days via set to avoid repeated schedule queries
    return check_date in get_trading_days_set(check_date, check_date, exchange)
    
    


def get_trading_days_set(start_date: str, end_date: str, exchange: str = 'NYSE') -> Set[str]:
    """
    Get a set of trading days (as strings) between start_date and end_date.
    Useful for fast membership testing.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        exchange: Exchange name (default: NYSE)
        
    Returns:
        Set of trading days in YYYY-MM-DD format
    """
    # Directly use cached days to avoid conversions and repeated lookups
    return set(_cached_trading_days(exchange, start_date, end_date))


def filter_trading_days(dates: List[str], exchange: str = 'NYSE') -> List[str]:
    """
    Filter a list of dates to only include trading days.
    
    Args:
        dates: List of dates (YYYY-MM-DD)
        exchange: Exchange name (default: NYSE)
        
    Returns:
        List of dates that are trading days
    """
    if not dates:
        return []
    
    # Get trading days for the range
    min_date = min(dates)
    max_date = max(dates)
    trading_days_set = get_trading_days_set(min_date, max_date, exchange)
    
    return [d for d in dates if d in trading_days_set]


def get_missing_trading_days(existing_dates: List[str], start_date: str, end_date: str, 
                             exchange: str = 'NYSE') -> List[str]:
    """
    Get missing trading days from a list of existing dates.

    If end_date is today or later (in the exchange's timezone), adjust end_date to the latest date
    for which there is already price data (i.e., the latest in existing_dates).

    Args:
        existing_dates: List of existing dates (YYYY-MM-DD)
        start_date: Start date of range (YYYY-MM-DD)
        end_date: End date of range (YYYY-MM-DD)
        exchange: Exchange name (default: NYSE)

    Returns:
        List of missing trading days
    """
    import pytz
    from datetime import datetime

    # Determine exchange timezone
    exchange_tz_map = {
        'NYSE': 'America/New_York',
        'NASDAQ': 'America/New_York',
        'AMEX': 'America/New_York',
        'LSE': 'Europe/London',
        'JPX': 'Asia/Tokyo',
        # Add more exchanges as needed
    }
    tz_str = exchange_tz_map.get(exchange.upper(), 'America/New_York')
    exchange_tz = pytz.timezone(tz_str)

    # Get current time in exchange timezone
    now_exchange = datetime.now(exchange_tz).date()

    # If end_date is today or later in exchange timezone, adjust to latest existing date <= end_date
    end_date_dt = pd.to_datetime(end_date).date()
    if end_date_dt >= now_exchange and existing_dates:
        # Only consider existing_dates <= end_date
        existing_dates_in_range = [d for d in existing_dates if pd.to_datetime(d).date() <= end_date_dt]
        if existing_dates_in_range:
            # Use the latest available date
            latest_existing = max(existing_dates_in_range)
            end_date = latest_existing

    # Get all trading days in the range
    all_trading_days = get_trading_days_set(start_date, end_date, exchange)

    # Convert existing dates to set
    existing_set = set(existing_dates)

    # Find missing trading days
    missing = sorted(all_trading_days - existing_set)

    return missing


def consolidate_date_ranges(dates: List[str]) -> List[tuple]:
    """
    Consolidate a list of dates into continuous ranges.
    
    Args:
        dates: Sorted list of dates (YYYY-MM-DD)
        
    Returns:
        List of (start_date, end_date) tuples
    """
    if not dates:
        return []
    
    # Sort dates
    sorted_dates = sorted(dates)
    date_objs = [pd.to_datetime(d) for d in sorted_dates]
    
    ranges = []
    range_start = date_objs[0]
    range_end = date_objs[0]
    
    for i in range(1, len(date_objs)):
        current = date_objs[i]
        # Check if current is consecutive (within 1 day)
        if (current - range_end).days <= 1:
            range_end = current
        else:
            # Save current range and start new one
            ranges.append((
                range_start.strftime('%Y-%m-%d'),
                range_end.strftime('%Y-%m-%d')
            ))
            range_start = current
            range_end = current
    
    # Add last range
    ranges.append((
        range_start.strftime('%Y-%m-%d'),
        range_end.strftime('%Y-%m-%d')
    ))
    
    return ranges


if __name__ == "__main__":
    # Test the trading calendar
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Trading Calendar Test")
    print("=" * 70)
    
    # Test get_trading_days
    print("\nTest 1: Get trading days for January 2024")
    trading_days = get_trading_days('2024-01-01', '2024-01-31')
    print(f"Trading days count: {len(trading_days)}")
    print(f"First 5 days: {trading_days[:5].strftime('%Y-%m-%d').tolist()}")
    print(f"Last 5 days: {trading_days[-5:].strftime('%Y-%m-%d').tolist()}")
    
    # Test is_trading_day
    print("\nTest 2: Check specific dates")
    test_dates = [
        '2024-01-01',  # New Year (holiday)
        '2024-01-02',  # Tuesday
        '2024-01-06',  # Saturday
        '2024-01-08',  # Monday
    ]
    for d in test_dates:
        is_trading = is_trading_day(d)
        weekday = pd.to_datetime(d).strftime('%A')
        print(f"  {d} ({weekday}): {'? Trading day' if is_trading else '? Non-trading day'}")
    
    # Test get_missing_trading_days
    print("\nTest 3: Find missing trading days")
    existing = ['2024-01-02', '2024-01-03', '2024-01-08', '2024-01-09']
    missing = get_missing_trading_days(existing, '2024-01-02', '2024-01-10')
    print(f"Existing dates: {existing}")
    print(f"Missing trading days: {missing}")
    
    # Test consolidate_date_ranges
    print("\nTest 4: Consolidate date ranges")
    dates_to_consolidate = ['2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10']
    ranges = consolidate_date_ranges(dates_to_consolidate)
    print(f"Dates: {dates_to_consolidate}")
    print(f"Consolidated ranges: {ranges}")

