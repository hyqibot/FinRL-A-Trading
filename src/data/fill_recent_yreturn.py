#!/usr/bin/env python3
"""Fill y_return for Q4 2025 and Q1 2026 using yfinance prices.

Q4 2025 (datadate=2025-12-31): y_return = log(price_2026-03-31 / adj_close_q_2025-12-31)
Q1 2026 (datadate=2026-03-31): y_return = log(today_price / adj_close_q_2026-03-31)
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(project_root, "data", "finrl_trading.db")


def get_price_on_or_before(daily_prices: pd.Series, target_date: pd.Timestamp) -> float:
    if daily_prices.empty:
        return np.nan
    mask = daily_prices.index <= target_date
    if mask.any():
        return daily_prices[mask].iloc[-1]
    return np.nan


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. Get Q4 2025 and Q1 2026 records
    q4_2025 = pd.read_sql(
        "SELECT ticker, datadate, adj_close_q, y_return FROM fundamental_data WHERE datadate = '2025-12-31'",
        conn,
    )
    q1_2026 = pd.read_sql(
        "SELECT ticker, datadate, adj_close_q, y_return FROM fundamental_data WHERE datadate = '2026-03-31'",
        conn,
    )
    q4_2025["adj_close_q"] = pd.to_numeric(q4_2025["adj_close_q"], errors="coerce")
    q1_2026["adj_close_q"] = pd.to_numeric(q1_2026["adj_close_q"], errors="coerce")

    print(f"Q4 2025 records: {len(q4_2025)}, missing y_return: {q4_2025['y_return'].isna().sum()}")
    print(f"Q1 2026 records: {len(q1_2026)}, missing y_return: {q1_2026['y_return'].isna().sum()}")

    # 2. Collect all tickers that need price updates
    all_tickers = sorted(set(q4_2025["ticker"].tolist() + q1_2026["ticker"].tolist()))
    yf_tickers = [t.replace(".", "-") for t in all_tickers]
    yf_map = {t: t.replace(".", "-") for t in all_tickers}

    # 3. Download prices: need 2026-03-31 and today's prices
    # Download from 2026-03-20 to today+1 to cover both dates
    today = date.today()
    start_str = "2026-03-20"
    end_str = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"\nDownloading prices {start_str} ~ {end_str} for {len(yf_tickers)} tickers ...")

    batch_size = 100
    all_prices = {}
    for i in range(0, len(yf_tickers), batch_size):
        batch = yf_tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(yf_tickers) - 1) // batch_size + 1}: {len(batch)} tickers")
        try:
            data = yf.download(batch, start=start_str, end=end_str, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                close_data = data["Close"] if "Close" in data.columns.get_level_values(0) else data
                for t in batch:
                    if t in close_data.columns:
                        series = close_data[t].dropna()
                        if len(series) > 0:
                            all_prices[t] = series
            else:
                if "Close" in data.columns:
                    series = data["Close"].dropna()
                    if len(series) > 0:
                        all_prices[batch[0]] = series
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"Got price data for {len(all_prices)} / {len(yf_tickers)} tickers")

    # 4. Fill Q4 2025 y_return = log(price_2026-03-31 / adj_close_q_2025-12-31)
    target_q1_end = pd.Timestamp("2026-03-31")
    updates_q4 = []
    for _, row in q4_2025.iterrows():
        tic = row["ticker"]
        adj_close = row["adj_close_q"]
        if pd.isna(adj_close) or adj_close <= 0:
            continue
        yf_tic = yf_map[tic]
        if yf_tic not in all_prices:
            continue
        next_price = get_price_on_or_before(all_prices[yf_tic], target_q1_end)
        if np.isnan(next_price) or next_price <= 0:
            continue
        y_ret = round(float(np.log(next_price / adj_close)), 6)
        updates_q4.append((y_ret, tic, "2025-12-31"))

    print(f"\nQ4 2025 y_return updates: {len(updates_q4)}")

    # 5. Fill Q1 2026 y_return = log(today_price / adj_close_q_2026-03-31)
    today_ts = pd.Timestamp(today)
    updates_q1 = []
    for _, row in q1_2026.iterrows():
        tic = row["ticker"]
        adj_close = row["adj_close_q"]
        if pd.isna(adj_close) or adj_close <= 0:
            continue
        yf_tic = yf_map[tic]
        if yf_tic not in all_prices:
            continue
        today_price = get_price_on_or_before(all_prices[yf_tic], today_ts)
        if np.isnan(today_price) or today_price <= 0:
            continue
        y_ret = round(float(np.log(today_price / adj_close)), 6)
        updates_q1.append((y_ret, tic, "2026-03-31"))

    print(f"Q1 2026 y_return updates: {len(updates_q1)}")

    # 6. Apply updates
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    for y_ret, tic, dt in updates_q4 + updates_q1:
        cursor.execute(
            "UPDATE fundamental_data SET y_return = ? WHERE ticker = ? AND datadate = ?",
            (y_ret, tic, dt),
        )
    conn.commit()
    print(f"\nTotal updates applied: {len(updates_q4) + len(updates_q1)}")

    # 7. Verify
    for label, dt in [("Q4 2025", "2025-12-31"), ("Q1 2026", "2026-03-31")]:
        check = pd.read_sql(
            f"SELECT ticker, adj_close_q, y_return FROM fundamental_data WHERE datadate = '{dt}'", conn
        )
        check["y_return"] = pd.to_numeric(check["y_return"], errors="coerce")
        print(f"\n{label}: {len(check)} records, y_return filled: {check['y_return'].notna().sum()}, "
              f"missing: {check['y_return'].isna().sum()}")
        # Show a few samples
        sample = check[check["y_return"].notna()].head(5)
        for _, r in sample.iterrows():
            print(f"  {r['ticker']}: adj_close={r['adj_close_q']}, y_return={r['y_return']:+.4f}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
