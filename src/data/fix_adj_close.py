#!/usr/bin/env python3
"""Fix adj_close_q and y_return in fundamental_data using yfinance prices."""

import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import yfinance as yf

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_PATH = os.path.join(project_root, "data", "finrl_trading.db")


def get_quarter_end_price(daily_prices: pd.Series, quarter_date: pd.Timestamp) -> float:
    """Get adjusted close on or closest before the quarter-end date."""
    if daily_prices.empty:
        return np.nan
    # Look for price on or up to 5 business days before quarter end
    mask = daily_prices.index <= quarter_date
    if mask.any():
        return daily_prices[mask].iloc[-1]
    return np.nan


def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. Get all (ticker, datadate) pairs
    df = pd.read_sql("SELECT ticker, datadate, adj_close_q FROM fundamental_data ORDER BY ticker, datadate", conn)
    df["adj_close_q"] = pd.to_numeric(df["adj_close_q"], errors="coerce")
    df["datadate"] = pd.to_datetime(df["datadate"])
    tickers = sorted(df["ticker"].unique())
    print(f"Tickers to fix: {len(tickers)}, records: {len(df)}")

    # Detect currently frozen tickers
    frozen_before = 0
    for tic, gdf in df.groupby("ticker"):
        prices = gdf["adj_close_q"].values
        for i in range(1, len(prices)):
            if prices[i] == prices[i - 1] and prices[i] > 0:
                frozen_before += 1
    print(f"Frozen adj_close_q pairs before fix: {frozen_before}")

    # 2. Map ticker symbols for yfinance (. -> -)
    yf_map = {}
    for t in tickers:
        yf_ticker = t.replace(".", "-")
        yf_map[t] = yf_ticker

    # 3. Download prices in batches
    min_date = df["datadate"].min() - pd.Timedelta(days=10)
    max_date = df["datadate"].max() + pd.Timedelta(days=10)
    start_str = min_date.strftime("%Y-%m-%d")
    end_str = max_date.strftime("%Y-%m-%d")

    print(f"Downloading prices {start_str} ~ {end_str} ...")
    yf_tickers = list(yf_map.values())

    batch_size = 100
    all_prices = {}
    for i in range(0, len(yf_tickers), batch_size):
        batch = yf_tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(yf_tickers) - 1) // batch_size + 1}: {len(batch)} tickers ...")
        try:
            data = yf.download(batch, start=start_str, end=end_str, auto_adjust=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: columns are (Price, Ticker)
                close_data = data["Close"] if "Close" in data.columns.get_level_values(0) else data
                for t in batch:
                    if t in close_data.columns:
                        series = close_data[t].dropna()
                        if len(series) > 0:
                            all_prices[t] = series
            else:
                # Single ticker
                if "Close" in data.columns:
                    series = data["Close"].dropna()
                    if len(series) > 0:
                        all_prices[batch[0]] = series
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"Got price data for {len(all_prices)} / {len(yf_tickers)} tickers")

    # 4. Update adj_close_q for each record
    updates = []
    missing_tickers = set()
    for _, row in df.iterrows():
        tic = row["ticker"]
        qdate = row["datadate"]
        yf_tic = yf_map[tic]

        if yf_tic not in all_prices:
            missing_tickers.add(tic)
            continue

        price = get_quarter_end_price(all_prices[yf_tic], qdate)
        if not np.isnan(price):
            updates.append((round(float(price), 2), tic, qdate.strftime("%Y-%m-%d")))

    print(f"Price updates ready: {len(updates)}")
    if missing_tickers:
        print(f"Tickers with no yfinance data ({len(missing_tickers)}): {sorted(missing_tickers)[:20]}...")

    # 5. Apply updates
    cursor = conn.cursor()
    cursor.execute("BEGIN")
    for adj_close, tic, dt in updates:
        cursor.execute(
            "UPDATE fundamental_data SET adj_close_q = ? WHERE ticker = ? AND datadate = ?",
            (adj_close, tic, dt),
        )
    conn.commit()
    print(f"Updated adj_close_q for {len(updates)} records")

    # 6. Recompute y_return = log(next_quarter / current_quarter)
    df2 = pd.read_sql("SELECT ticker, datadate, adj_close_q FROM fundamental_data ORDER BY ticker, datadate", conn)
    df2["adj_close_q"] = pd.to_numeric(df2["adj_close_q"], errors="coerce")
    df2["next_q_price"] = df2.groupby("ticker")["adj_close_q"].shift(-1)
    df2["y_return_new"] = np.log(df2["next_q_price"] / df2["adj_close_q"])
    df2.loc[df2["adj_close_q"] <= 0, "y_return_new"] = np.nan
    df2.loc[df2["next_q_price"] <= 0, "y_return_new"] = np.nan

    cursor.execute("BEGIN")
    y_updates = 0
    for _, row in df2.iterrows():
        val = None if pd.isna(row["y_return_new"]) else round(float(row["y_return_new"]), 6)
        cursor.execute(
            "UPDATE fundamental_data SET y_return = ? WHERE ticker = ? AND datadate = ?",
            (val, row["ticker"], row["datadate"]),
        )
        y_updates += 1
    conn.commit()
    print(f"Recomputed y_return for {y_updates} records")

    # 7. Verify fix
    df3 = pd.read_sql("SELECT ticker, datadate, adj_close_q, y_return FROM fundamental_data ORDER BY ticker, datadate", conn)
    df3["adj_close_q"] = pd.to_numeric(df3["adj_close_q"], errors="coerce")
    df3["y_return"] = pd.to_numeric(df3["y_return"], errors="coerce")

    frozen_after = 0
    for tic, gdf in df3.groupby("ticker"):
        prices = gdf["adj_close_q"].values
        for i in range(1, len(prices)):
            if prices[i] == prices[i - 1] and prices[i] > 0:
                frozen_after += 1

    zero_returns = (df3["y_return"] == 0).sum()
    null_returns = df3["y_return"].isna().sum()

    print(f"\n=== Verification ===")
    print(f"Frozen adj_close_q pairs: {frozen_before} -> {frozen_after}")
    print(f"y_return = 0: {zero_returns}")
    print(f"y_return = NaN: {null_returns}")

    # Spot-check previously frozen tickers
    for tic in ["LITE", "MSTR", "APP", "SHOP", "TTD"]:
        tdf = df3[df3["ticker"] == tic][["datadate", "adj_close_q", "y_return"]].tail(6)
        print(f"\n  {tic}:")
        for _, r in tdf.iterrows():
            yr = f"{r['y_return']:+.4f}" if pd.notna(r["y_return"]) else "NaN"
            print(f"    {r['datadate']}  adj_close={r['adj_close_q']:>10.2f}  y_return={yr}")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
