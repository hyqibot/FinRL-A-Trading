#!/usr/bin/env python3
"""Backfill historical SP500 fundamental data (2016-2025).

Identifies ex-SP500 members missing from the fundamental_data table
using sp500_historical_constituents.csv, fetches their fundamentals
from FMP, and fills tradedate / actual_tradedate / trade_price /
y_return so they are fully usable for survivorship-bias-free analysis.

Usage:
    python3 src/data/backfill_historical_sp500.py
"""

import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

from data.data_fetcher import fetch_fundamental_data, get_all_historical_sp500_tickers
from data.data_store import get_data_store

DB_PATH = os.path.join(project_root, "data", "finrl_trading.db")
CSV_PATH = os.path.join(project_root, "data", "sp500_historical_constituents.csv")

# Quarter-end dates we care about
QUARTER_ENDS = [
    f"{y}-{m}" for y in range(2016, 2026)
    for m in ("03-31", "06-30", "09-30", "12-31")
    if not (y == 2025 and m in ("06-30", "09-30", "12-31"))
]
# Trim: 2016-Q1 through 2025-Q1 = 37 quarters
QUARTER_ENDS = [q for q in QUARTER_ENDS if q <= "2025-03-31"]

# Map datadate → tradedate (first day of next-next month)
DATADATE_TO_TRADEDATE = {
    "12-31": ("03-01", 1),   # Dec 31 → Mar 1 (next year)
    "03-31": ("06-01", 0),   # Mar 31 → Jun 1 (same year)
    "06-30": ("09-01", 0),   # Jun 30 → Sep 1 (same year)
    "09-30": ("12-01", 0),   # Sep 30 → Dec 1 (same year)
}


_csv_cache = {}

def get_sp500_members_for_quarter(csv_path: str, quarter_end: str) -> set:
    """Return the set of SP500 tickers active during a given quarter."""
    if csv_path not in _csv_cache:
        _csv_cache[csv_path] = pd.read_csv(csv_path)
        _csv_cache[csv_path]["date"] = pd.to_datetime(_csv_cache[csv_path]["date"])
    df = _csv_cache[csv_path]
    qe = pd.to_datetime(quarter_end)
    # Quarter start is ~3 months before quarter end
    qs = qe - pd.DateOffset(months=3)
    # Find rows within this quarter
    mask = (df["date"] >= qs) & (df["date"] <= qe)
    members = set()
    for tickers_str in df.loc[mask, "tickers"]:
        members.update(t.strip() for t in tickers_str.split(","))
    return members


def identify_missing_pairs(conn: sqlite3.Connection) -> tuple:
    """Find (ticker, datadate) pairs that are in SP500 history but missing from DB.

    Returns:
        (missing_tickers: set, missing_pairs: list of (ticker, datadate),
         quarter_expected: dict of datadate -> set of expected tickers)
    """
    # Load existing (ticker, datadate) pairs from DB
    existing = pd.read_sql(
        "SELECT ticker, datadate FROM fundamental_data", conn
    )
    existing_set = set(zip(existing["ticker"], existing["datadate"]))

    missing_tickers = set()
    missing_pairs = []
    quarter_expected = {}

    for qe in QUARTER_ENDS:
        members = get_sp500_members_for_quarter(CSV_PATH, qe)
        quarter_expected[qe] = members
        for tic in members:
            if (tic, qe) not in existing_set:
                missing_tickers.add(tic)
                missing_pairs.append((tic, qe))

    return missing_tickers, missing_pairs, quarter_expected


def fetch_and_store_fundamentals(missing_tickers: set):
    """Fetch fundamentals for missing tickers via existing pipeline and save to DB."""
    tickers_list = sorted(missing_tickers)
    print(f"\n=== Step 2: Fetching fundamentals for {len(tickers_list)} missing tickers ===")

    # Process in batches to manage memory and provide progress
    batch_size = 50
    store = get_data_store()
    total_saved = 0

    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers_list) - 1) // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} tickers "
              f"({batch[0]}..{batch[-1]})")

        try:
            tickers_df = pd.DataFrame({
                "tickers": batch,
                "sectors": [""] * len(batch),
            })
            df = fetch_fundamental_data(
                tickers_df,
                start_date="2015-06-01",
                end_date="2026-04-01",
                preferred_source="FMP",
            )
            if not df.empty:
                n = store.save_fundamental_data(df)
                total_saved += n
                print(f"    Fetched {len(df)} records, saved {n}")
            else:
                print(f"    No data returned")
        except Exception as e:
            print(f"    ERROR: {e}")

        # Brief pause to respect FMP rate limits
        time.sleep(1)

    print(f"  Total saved: {total_saved}")
    return total_saved


def compute_tradedate(datadate_str: str) -> str:
    """Map datadate to tradedate: 12/31→3/1, 3/31→6/1, 6/30→9/1, 9/30→12/1."""
    mm_dd = datadate_str[5:]  # e.g. "03-31"
    year = int(datadate_str[:4])
    mapping = DATADATE_TO_TRADEDATE.get(mm_dd)
    if mapping is None:
        return ""
    target_mmdd, year_add = mapping
    return f"{year + year_add}-{target_mmdd}"


def fill_tradedate_and_prices(conn: sqlite3.Connection):
    """Fill tradedate, actual_tradedate, trade_price for records where they are NULL."""
    print("\n=== Step 3: Filling tradedate, actual_tradedate, trade_price ===")

    # Find records with NULL tradedate
    null_records = pd.read_sql(
        "SELECT ticker, datadate FROM fundamental_data WHERE tradedate IS NULL",
        conn,
    )
    if null_records.empty:
        print("  No records with NULL tradedate — nothing to fill.")
        return

    print(f"  Records with NULL tradedate: {len(null_records)}")

    # Compute tradedate for each record
    null_records["tradedate"] = null_records["datadate"].apply(compute_tradedate)
    # Drop any that couldn't be mapped (non-standard quarter ends)
    null_records = null_records[null_records["tradedate"] != ""].copy()

    # Group by ticker to fetch prices efficiently
    tickers = sorted(null_records["ticker"].unique())
    print(f"  Unique tickers needing price data: {len(tickers)}")

    # Collect all needed tradedates
    all_tradedates = sorted(null_records["tradedate"].unique())
    print(f"  Unique tradedates to look up: {len(all_tradedates)}")

    # Get FMP API key
    try:
        from src.config.settings import get_config
        config = get_config()
        api_key = config.fmp.api_key.get_secret_value()
    except Exception as e:
        print(f"  ERROR: Cannot get FMP API key: {e}")
        print("  Setting tradedate only (without prices).")
        api_key = None

    # Use NYSE calendar to find actual trading days
    import pandas_market_calendars as mcal
    nyse = mcal.get_calendar("NYSE")

    # For each tradedate, find the actual first trading day on/after it
    tradedate_to_actual = {}
    for td_str in all_tradedates:
        td = pd.to_datetime(td_str)
        # Search in a 10-day window
        schedule = nyse.schedule(
            start_date=td.strftime("%Y-%m-%d"),
            end_date=(td + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        )
        if not schedule.empty:
            actual = schedule.index[0].strftime("%Y-%m-%d")
        else:
            actual = td_str  # fallback
        tradedate_to_actual[td_str] = actual

    # Now fetch prices for each (ticker, actual_tradedate) via FMP
    # Build a mapping: ticker -> {actual_tradedate: price}
    ticker_prices = {}

    if api_key:
        # Collect all unique (ticker, actual_tradedate) pairs
        price_lookups = []
        for _, row in null_records.iterrows():
            td = row["tradedate"]
            actual_td = tradedate_to_actual.get(td, td)
            price_lookups.append((row["ticker"], actual_td))

        # Group by ticker for batch price fetches
        from collections import defaultdict
        ticker_dates = defaultdict(set)
        for tic, dt in price_lookups:
            ticker_dates[tic].add(dt)

        # Fetch prices using FMP historical-price-eod
        base_url = "https://financialmodelingprep.com/stable"
        print(f"  Fetching prices for {len(ticker_dates)} tickers ...")

        batch_count = 0
        for tic, dates in ticker_dates.items():
            fmp_tic = tic.replace(".", "-")
            # Get min/max date range needed
            min_dt = min(dates)
            max_dt = max(dates)
            # Extend by a few days to ensure coverage
            from_dt = (pd.to_datetime(min_dt) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
            to_dt = (pd.to_datetime(max_dt) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")

            url = (f"{base_url}/historical-price-eod/full?"
                   f"symbol={fmp_tic}&from={from_dt}&to={to_dt}&apikey={api_key}")
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()

                historical = []
                if isinstance(data, dict) and "historical" in data:
                    historical = data["historical"] or []
                elif isinstance(data, list):
                    historical = data

                # Index by date
                price_by_date = {}
                for item in historical:
                    price_by_date[item["date"]] = item.get("adjClose", item.get("close"))

                ticker_prices[tic] = price_by_date

            except Exception as e:
                pass  # Skip tickers with no price data (bankrupt etc.)

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"    {batch_count}/{len(ticker_dates)} tickers done ...")
                time.sleep(0.5)

        print(f"  Got price data for {len(ticker_prices)} / {len(ticker_dates)} tickers")

    # Apply updates
    cursor = conn.cursor()
    updates = 0
    for _, row in null_records.iterrows():
        tic = row["ticker"]
        datadate = row["datadate"]
        td = row["tradedate"]
        actual_td = tradedate_to_actual.get(td, td)

        # Look up price
        price = None
        if tic in ticker_prices:
            price_map = ticker_prices[tic]
            # Try exact actual_tradedate, then nearby dates
            price = price_map.get(actual_td)
            if price is None:
                # Try a few days around
                for offset in range(1, 6):
                    for sign in [1, -1]:
                        dt_try = (pd.to_datetime(actual_td) + pd.Timedelta(days=offset * sign)).strftime("%Y-%m-%d")
                        if dt_try in price_map:
                            price = price_map[dt_try]
                            break
                    if price is not None:
                        break

        cursor.execute(
            """UPDATE fundamental_data
               SET tradedate = ?, actual_tradedate = ?, trade_price = ?
               WHERE ticker = ? AND datadate = ?""",
            (td, actual_td, price, tic, datadate),
        )
        updates += 1

    conn.commit()
    print(f"  Updated tradedate/actual_tradedate/trade_price for {updates} records")


def recompute_y_return(conn: sqlite3.Connection, tickers: set):
    """Recompute y_return = ln(next_trade_price / this_trade_price) for given tickers."""
    print(f"\n=== Step 3b: Recomputing y_return for {len(tickers)} tickers ===")

    if not tickers:
        return

    placeholders = ",".join(["?"] * len(tickers))
    df = pd.read_sql(
        f"""SELECT ticker, datadate, trade_price
            FROM fundamental_data
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, datadate""",
        conn,
        params=list(tickers),
    )
    df["trade_price"] = pd.to_numeric(df["trade_price"], errors="coerce")

    # Compute y_return = log(next_quarter_trade_price / this_quarter_trade_price)
    df["next_trade_price"] = df.groupby("ticker")["trade_price"].shift(-1)
    df["y_return_new"] = np.where(
        (df["trade_price"] > 0) & (df["next_trade_price"] > 0),
        np.log(df["next_trade_price"] / df["trade_price"]),
        np.nan,
    )

    cursor = conn.cursor()
    updates = 0
    for _, row in df.iterrows():
        val = None if pd.isna(row["y_return_new"]) else round(float(row["y_return_new"]), 6)
        cursor.execute(
            "UPDATE fundamental_data SET y_return = ? WHERE ticker = ? AND datadate = ?",
            (val, row["ticker"], row["datadate"]),
        )
        updates += 1
    conn.commit()
    print(f"  Updated y_return for {updates} records")


def verify_coverage(conn: sqlite3.Connection):
    """Print per-quarter coverage comparison."""
    print("\n=== Step 4: Coverage Verification ===")
    print(f"{'Quarter':<12} {'Expected':>8} {'In DB':>8} {'Coverage':>10}")
    print("-" * 42)

    for qe in QUARTER_ENDS:
        expected = get_sp500_members_for_quarter(CSV_PATH, qe)
        row = pd.read_sql(
            "SELECT COUNT(DISTINCT ticker) as cnt FROM fundamental_data WHERE datadate = ?",
            conn,
            params=[qe],
        )
        in_db = int(row["cnt"].iloc[0])
        pct = (in_db / len(expected) * 100) if expected else 0
        flag = "" if in_db >= 480 else " <---"
        print(f"{qe:<12} {len(expected):>8} {in_db:>8} {pct:>9.1f}%{flag}")


def main():
    conn = sqlite3.connect(DB_PATH)

    # Step 1: Identify missing tickers
    print("=== Step 1: Identifying missing (ticker, quarter) pairs ===")
    missing_tickers, missing_pairs, quarter_expected = identify_missing_pairs(conn)
    print(f"  Missing unique tickers: {len(missing_tickers)}")
    print(f"  Missing (ticker, quarter) pairs: {len(missing_pairs)}")

    if not missing_tickers:
        print("  Nothing to backfill!")
        verify_coverage(conn)
        conn.close()
        return

    # Show a few examples
    examples = sorted(missing_tickers)[:20]
    print(f"  Examples: {', '.join(examples)} ...")

    # Step 2: Fetch fundamentals
    fetch_and_store_fundamentals(missing_tickers)

    # Reconnect to pick up saved data
    conn.close()
    conn = sqlite3.connect(DB_PATH)

    # Re-identify what's still missing after fetch (some tickers may have no FMP data)
    _, remaining_pairs, _ = identify_missing_pairs(conn)
    newly_added = len(missing_pairs) - len(remaining_pairs)
    print(f"\n  Newly added records: {newly_added}")
    print(f"  Still missing pairs: {len(remaining_pairs)} (bankrupt/delisted tickers with no FMP data)")

    # Step 3: Fill tradedate, actual_tradedate, trade_price
    fill_tradedate_and_prices(conn)

    # Step 3b: Recompute y_return for tickers that had new records added
    all_affected = pd.read_sql(
        "SELECT DISTINCT ticker FROM fundamental_data", conn
    )
    recompute_y_return(conn, missing_tickers & set(all_affected["ticker"]))

    # Step 4: Verify
    verify_coverage(conn)

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
