#!/usr/bin/env python3
"""Generate a standalone HTML dashboard from ml_bucket_selection Excel output.

Usage:
    python3 src/tools/dashboard.py data/nasdaq100_ml_dashboard_20260421.xlsx
    python3 src/tools/dashboard.py data/sp500_ml_dashboard_20260421.xlsx --no-open
"""

import argparse
import json
import os
import sys
import webbrowser

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ---------------------------------------------------------------------------
# Feature category classification
# ---------------------------------------------------------------------------
MOMENTUM_FEATURES = {
    "ret_1q", "ret_4q", "ret_accel", "eps_chg", "roe_chg", "gm_chg", "om_chg",
}


def classify_feature(name: str) -> str:
    if name in MOMENTUM_FEATURES:
        return "momentum"
    if name.startswith("sector_"):
        return "sector"
    return "fundamental"


CATEGORY_COLORS = {
    "momentum": "#3b82f6",
    "fundamental": "#f59e0b",
    "sector": "#6b7280",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_excel(path: str):
    """Load the 3-sheet dashboard Excel file."""
    rankings = pd.read_excel(path, sheet_name="Rankings")
    models = pd.read_excel(path, sheet_name="Models")
    try:
        features = pd.read_excel(path, sheet_name="Features")
    except ValueError:
        features = pd.DataFrame()
    return rankings, models, features


# ---------------------------------------------------------------------------
# Backtest data
# ---------------------------------------------------------------------------

def build_backtest_data(rankings: pd.DataFrame, buy_date: str, sell_date: str) -> dict | None:
    """Download daily prices via yfinance and compute cumulative returns.

    Strategy: per-bucket top-5 equal-weight, buckets equal-weight.
    This mirrors the tracking table approach used for both SP500 and NASDAQ100.
    """
    if yf is None:
        print("  [backtest] yfinance not installed — skipping backtest tab")
        return None

    # --- Per-bucket top 5 picks ---
    buckets_in_data = sorted(rankings["bucket"].unique().tolist()) if "bucket" in rankings.columns else []
    bucket_picks = {}  # {bucket: [tic, ...]}
    all_pick_tics = []
    for b in buckets_in_data:
        bdf = rankings[rankings["bucket"] == b].sort_values("rank_best")
        picks = bdf["tic"].head(5).tolist()
        bucket_picks[b] = picks
        all_pick_tics.extend(picks)
        print(f"  [backtest] {b}: {', '.join(picks)}")

    all_pick_tics = list(dict.fromkeys(all_pick_tics))  # dedupe, preserve order
    all_tics = list(dict.fromkeys(all_pick_tics + ["SPY", "QQQ"]))

    print(f"  [backtest] Downloading prices for {len(all_tics)} tickers: {buy_date} → {sell_date}")
    try:
        prices = yf.download(all_tics, start=buy_date, end=sell_date, auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  [backtest] yfinance download failed: {e}")
        return None

    # yfinance returns MultiIndex columns (Price, Ticker) when multiple tickers
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"]
    else:
        close = prices[["Close"]].copy()
        close.columns = all_tics

    # Drop tickers with no data
    close = close.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if close.empty or len(close) < 2:
        print("  [backtest] Not enough price data")
        return None

    # Daily simple returns
    daily_ret = close.pct_change().iloc[1:]
    dates = [d.strftime("%Y-%m-%d") for d in daily_ret.index]

    # --- Per-bucket equal-weight, then buckets equal-weight ---
    def portfolio_daily(tics):
        valid = [t for t in tics if t in daily_ret.columns]
        if not valid:
            return pd.Series(0.0, index=daily_ret.index)
        return daily_ret[valid].mean(axis=1)

    # Each bucket's daily return (equal-weight within bucket)
    bucket_daily = {}
    for b, picks in bucket_picks.items():
        bucket_daily[b] = portfolio_daily(picks)

    # Combined portfolio: equal-weight across buckets
    active_buckets = [b for b in bucket_daily if not (bucket_daily[b] == 0.0).all()]
    if active_buckets:
        combined_daily = pd.concat([bucket_daily[b] for b in active_buckets], axis=1).mean(axis=1)
    else:
        combined_daily = pd.Series(0.0, index=daily_ret.index)

    spy_daily = daily_ret["SPY"] if "SPY" in daily_ret.columns else pd.Series(0.0, index=daily_ret.index)
    qqq_daily = daily_ret["QQQ"] if "QQQ" in daily_ret.columns else pd.Series(0.0, index=daily_ret.index)

    # Cumulative returns: (1+r).cumprod() - 1
    def cum_ret(series):
        return ((1 + series).cumprod() - 1).tolist()

    cum = {
        "portfolio": [round(v, 6) for v in cum_ret(combined_daily)],
        "spy": [round(v, 6) for v in cum_ret(spy_daily)],
        "qqq": [round(v, 6) for v in cum_ret(qqq_daily)],
    }
    # Per-bucket cumulative
    bucket_cum = {}
    for b in active_buckets:
        bucket_cum[b] = [round(v, 6) for v in cum_ret(bucket_daily[b])]
    cum["buckets"] = bucket_cum

    # Individual stock total returns (grouped by bucket)
    stock_returns = []
    for b, picks in bucket_picks.items():
        for tic in picks:
            if tic in close.columns and len(close[tic].dropna()) >= 2:
                first = close[tic].dropna().iloc[0]
                last = close[tic].dropna().iloc[-1]
                total = (last / first) - 1
            else:
                total = 0.0
            row = rankings[rankings["tic"] == tic]
            pred = float(row["predicted_return"].iloc[0]) if len(row) > 0 and pd.notna(row["predicted_return"].iloc[0]) else 0.0
            stock_returns.append({
                "tic": tic,
                "bucket": b,
                "total_return": round(float(total), 6),
                "pred_return": round(pred, 6),
            })

    # Summary totals
    port_total = float(cum["portfolio"][-1]) if cum["portfolio"] else 0.0
    spy_total = float(cum["spy"][-1]) if cum["spy"] else 0.0
    qqq_total = float(cum["qqq"][-1]) if cum["qqq"] else 0.0

    # Per-bucket total returns
    bucket_totals = {}
    for b in active_buckets:
        bucket_totals[b] = round(float(bucket_cum[b][-1]), 6) if bucket_cum[b] else 0.0

    return {
        "dates": dates,
        "cumulative": cum,
        "stock_returns": stock_returns,
        "bucket_picks": {b: picks for b, picks in bucket_picks.items()},
        "bucket_totals": bucket_totals,
        "summary": {
            "period": f"{buy_date} → {sell_date}",
            "trading_days": len(dates),
            "portfolio_ret": round(port_total, 6),
            "spy_ret": round(spy_total, 6),
            "qqq_ret": round(qqq_total, 6),
            "alpha_spy": round(port_total - spy_total, 6),
            "alpha_qqq": round(port_total - qqq_total, 6),
        },
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(rankings: pd.DataFrame, models: pd.DataFrame, features: pd.DataFrame,
               excel_path: str, buy_date: str = None, sell_date: str = None) -> str:
    """Build a standalone HTML dashboard string."""

    # ---- Metadata ----
    basename = os.path.basename(excel_path)
    universe = basename.split("_")[0].upper() if "_" in basename else "Unknown"
    run_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    stock_count = len(rankings)
    best_model = rankings["best_model"].iloc[0] if "best_model" in rankings.columns and len(rankings) > 0 else "N/A"

    # ---- Infer dates in the data ----
    dates = sorted(rankings["datadate"].unique().tolist()) if "datadate" in rankings.columns else []

    # ---- All rankings for table (full dataset, not just top 20) ----
    display_cols = ["tic", "bucket", "gsector", "best_model", "predicted_return", "pred_ensemble_avg"]
    has_actual = "y_return" in rankings.columns and rankings["y_return"].notna().any()
    if has_actual:
        display_cols.append("y_return")

    table_rows = []
    for _, r in rankings.iterrows():
        row = {}
        for c in display_cols:
            val = r.get(c)
            if pd.isna(val):
                row[c] = None
            elif isinstance(val, float):
                row[c] = round(val, 6)
            else:
                row[c] = str(val)
        # Also include rank_best and rank_ensemble for sorting
        row["rank_best"] = int(r["rank_best"]) if "rank_best" in r and pd.notna(r.get("rank_best")) else 999
        row["rank_ensemble"] = int(r["rank_ensemble"]) if "rank_ensemble" in r and pd.notna(r.get("rank_ensemble")) else 999
        table_rows.append(row)

    # ---- Bucket top picks (for summary cards) ----
    buckets_in_data = sorted(rankings["bucket"].unique().tolist()) if "bucket" in rankings.columns else []
    bucket_cards = []
    for b in buckets_in_data:
        bdf = rankings[rankings["bucket"] == b].sort_values("rank_best")
        if len(bdf) == 0:
            continue
        top = bdf.iloc[0]
        bucket_cards.append({
            "bucket": str(b),
            "count": int(len(bdf)),
            "top_tic": str(top["tic"]),
            "top_pred": round(float(top["predicted_return"]), 4) if pd.notna(top.get("predicted_return")) else 0,
            "top_model": str(top["best_model"]) if pd.notna(top.get("best_model")) else "N/A",
        })

    # ---- Section 2: Model performance ----
    model_data = []
    if len(models) > 0:
        avg_mse = models.groupby("model")["val_mse"].mean().sort_values()
        best_model_name = avg_mse.index[0] if len(avg_mse) > 0 else ""
        for m, mse in avg_mse.items():
            model_data.append({
                "model": str(m),
                "mse": round(float(mse), 6),
                "is_best": str(m) == best_model_name,
            })

    # Per-bucket model details
    model_detail = []
    if len(models) > 0:
        for _, r in models.iterrows():
            model_detail.append({
                "bucket": str(r["bucket"]),
                "model": str(r["model"]),
                "val_mse": round(float(r["val_mse"]), 6),
                "train_size": int(r["train_size"]) if pd.notna(r.get("train_size")) else 0,
                "val_size": int(r["val_size"]) if pd.notna(r.get("val_size")) else 0,
                "infer_size": int(r["infer_size"]) if pd.notna(r.get("infer_size")) else 0,
            })

    # ---- Section 3: Feature importance (top 15 from best models) ----
    feat_data = []
    if len(features) > 0:
        best_feats = features[features["is_best"] == True].copy()  # noqa: E712
        if len(best_feats) == 0:
            best_feats = features.copy()
        avg_imp = best_feats.groupby("feature")["importance"].mean().nlargest(15)
        for feat, imp in avg_imp.items():
            cat = classify_feature(str(feat))
            feat_data.append({
                "feature": str(feat),
                "importance": round(float(imp), 6),
                "category": cat,
                "color": CATEGORY_COLORS[cat],
            })

    # ---- Backtest data ----
    backtest = None
    if buy_date:
        if not sell_date:
            sell_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        backtest = build_backtest_data(rankings, buy_date, sell_date)
    elif "tradedate" in rankings.columns:
        # Auto-detect buy date from tradedate column
        td = pd.to_datetime(rankings["tradedate"], errors="coerce").dropna()
        if len(td) > 0:
            inferred_buy = td.min().strftime("%Y-%m-%d")
            inferred_sell = sell_date or pd.Timestamp.now().strftime("%Y-%m-%d")
            print(f"  [backtest] Auto-detected buy_date={inferred_buy}, sell_date={inferred_sell}")
            backtest = build_backtest_data(rankings, inferred_buy, inferred_sell)

    # ---- Build JSON ----
    def _default(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return str(obj)

    payload = {
        "rankings": table_rows,
        "columns": display_cols,
        "models": model_data,
        "model_detail": model_detail,
        "features": feat_data,
        "bucket_cards": bucket_cards,
        "dates": [str(d) for d in dates],
        "meta": {
            "universe": universe,
            "run_date": run_date,
            "best_model": best_model,
            "stock_count": stock_count,
            "has_actual": has_actual,
            "buckets": buckets_in_data,
        },
    }
    if backtest:
        payload["backtest"] = backtest

    data_json = json.dumps(payload, default=_default)

    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_json)
    return html


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ML Stock Selection Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root {
  --sidebar-w: 220px;
  --purple: #6c47ff;
  --purple-dark: #4c2ab5;
  --purple-light: #f0ebff;
  --bg: #f7f8fa;
  --card-bg: #fff;
  --text-primary: #1a1a2e;
  --text-secondary: #64748b;
  --border: #e5e7eb;
  --green: #16a34a;
  --green-bg: rgba(22,163,74,0.08);
  --red: #dc2626;
  --red-bg: rgba(220,38,38,0.08);
  --blue: #3b82f6;
  --orange: #f59e0b;
  --gray: #6b7280;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: var(--bg); color: var(--text-primary); }

/* Sidebar */
.sidebar {
  position: fixed; left: 0; top: 0; bottom: 0; width: var(--sidebar-w);
  background: linear-gradient(180deg, #2d1b69 0%, #1a103f 100%);
  color: white; padding: 24px 0; z-index: 100; display: flex; flex-direction: column;
}
.sidebar-logo { padding: 0 20px 24px; border-bottom: 1px solid rgba(255,255,255,0.1); margin-bottom: 8px; }
.sidebar-logo h2 { font-size: 1.1rem; font-weight: 700; }
.sidebar-logo small { font-size: 0.75rem; opacity: 0.6; }
.sidebar-nav { flex: 1; }
.nav-item {
  display: flex; align-items: center; gap: 10px; padding: 10px 20px; margin: 2px 8px;
  border-radius: 8px; cursor: pointer; font-size: 0.88rem; font-weight: 500;
  color: rgba(255,255,255,0.65); transition: all 0.15s;
}
.nav-item:hover { background: rgba(255,255,255,0.08); color: white; }
.nav-item.active { background: var(--purple); color: white; }
.nav-item svg { width: 18px; height: 18px; flex-shrink: 0; }

/* Main */
.main { margin-left: var(--sidebar-w); padding: 24px 32px; min-height: 100vh; }

/* Date tabs */
.date-tabs { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.date-tab {
  padding: 6px 14px; border-radius: 6px; font-size: 0.8rem; font-weight: 500;
  background: white; border: 1px solid var(--border); cursor: pointer; color: var(--text-secondary);
}
.date-tab.active { background: var(--purple); color: white; border-color: var(--purple); }

/* Bucket cards */
.bucket-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
.bucket-card {
  background: white; border-radius: 10px; padding: 16px; border-left: 4px solid var(--purple);
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.bucket-card.growth_tech { border-left-color: #6c47ff; }
.bucket-card.cyclical { border-left-color: #f59e0b; }
.bucket-card.real_assets { border-left-color: #3b82f6; }
.bucket-card.defensive { border-left-color: #16a34a; }
.bucket-card.unified_all { border-left-color: #6c47ff; }
.bucket-card .bc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.bucket-card .bc-bucket {
  font-size: 0.75rem; font-weight: 600; text-transform: capitalize;
  padding: 2px 8px; border-radius: 4px;
}
.bucket-card .bc-bucket.growth_tech { background: #f0ebff; color: #6c47ff; }
.bucket-card .bc-bucket.cyclical { background: #fef3c7; color: #b45309; }
.bucket-card .bc-bucket.real_assets { background: #dbeafe; color: #1d4ed8; }
.bucket-card .bc-bucket.defensive { background: #dcfce7; color: #15803d; }
.bucket-card .bc-bucket.unified_all { background: #f0ebff; color: #6c47ff; }
.bucket-card .bc-count { font-size: 0.75rem; color: var(--text-secondary); }
.bucket-card .bc-tic { font-size: 1.25rem; font-weight: 700; margin-bottom: 2px; }
.bucket-card .bc-pred { font-size: 0.85rem; font-weight: 600; }
.bucket-card .bc-model { font-size: 0.72rem; color: var(--text-secondary); }

/* Section card */
.section { display: none; }
.section.active { display: block; }
.card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; }

/* Filter bar */
.filter-bar { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 16px; }
.filter-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.pill {
  padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 500;
  background: #f1f5f9; border: 1px solid transparent; cursor: pointer; color: var(--text-secondary);
  transition: all 0.15s;
}
.pill:hover { background: #e2e8f0; }
.pill.active { background: var(--text-primary); color: white; }
.rank-toggle { display: flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.rank-btn {
  padding: 5px 14px; font-size: 0.8rem; font-weight: 500; cursor: pointer;
  background: white; border: none; color: var(--text-secondary); transition: all 0.15s;
}
.rank-btn.active { background: var(--purple); color: white; }

/* Table */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
th {
  text-align: left; padding: 10px 12px; font-weight: 600; font-size: 0.75rem;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--text-secondary);
  border-bottom: 2px solid var(--border); background: #fafbfc; position: sticky; top: 0;
  cursor: pointer; user-select: none; white-space: nowrap;
}
th:hover { color: var(--text-primary); }
th .sort-icon { margin-left: 4px; font-size: 0.7rem; opacity: 0.4; }
th.sorted .sort-icon { opacity: 1; color: var(--purple); }
td { padding: 9px 12px; border-bottom: 1px solid #f3f4f6; white-space: nowrap; }
tr { transition: background 0.1s; }
tr:hover { background: #f8f9ff; }
.rank-num {
  display: inline-flex; align-items: center; justify-content: center;
  width: 26px; height: 26px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;
}
.rank-1 { background: #fef3c7; color: #b45309; }
.rank-2 { background: #e5e7eb; color: #374151; }
.rank-3 { background: #fed7aa; color: #9a3412; }
.rank-other { background: #f3f4f6; color: #6b7280; }

/* Bucket badge */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-size: 0.72rem; font-weight: 600; text-transform: capitalize;
}
.badge-growth_tech { background: #f0ebff; color: #6c47ff; }
.badge-cyclical { background: #fef3c7; color: #b45309; }
.badge-real_assets { background: #dbeafe; color: #1d4ed8; }
.badge-defensive { background: #dcfce7; color: #15803d; }
.badge-unified_all { background: #f0ebff; color: #6c47ff; }

/* Return bar */
.ret-cell { position: relative; min-width: 110px; }
.ret-bar { position: absolute; top: 3px; bottom: 3px; border-radius: 3px; }
.ret-bar.pos { background: var(--green-bg); left: 50%; }
.ret-bar.neg { background: var(--red-bg); right: 50%; }
.ret-val { position: relative; z-index: 1; font-weight: 600; }
.ret-val.pos { color: var(--green); }
.ret-val.neg { color: var(--red); }

/* Charts */
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
@media (max-width: 900px) { .chart-grid { grid-template-columns: 1fr; } }
.chart-box { position: relative; height: 360px; }
.legend-bar { display: flex; gap: 16px; margin-top: 10px; font-size: 0.78rem; color: var(--text-secondary); }
.legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle; }

/* Model detail table */
.model-detail-table { margin-top: 16px; }
.model-detail-table th { font-size: 0.72rem; }
.model-detail-table td { font-size: 0.8rem; }
.mse-best { font-weight: 700; color: var(--green); }

/* Feature section bucket tabs */
.feat-bucket-tabs { display: flex; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }

/* Backtest */
.bt-header { font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 20px; }
.bt-summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }
.bt-card {
  background: white; border-radius: 10px; padding: 18px; text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-top: 3px solid var(--gray);
}
.bt-card.portfolio { border-top-color: #6c47ff; }
.bt-card.spy { border-top-color: #6b7280; }
.bt-card.qqq { border-top-color: #f59e0b; }
.bt-card.growth_tech { border-top-color: #6c47ff; }
.bt-card.cyclical { border-top-color: #f59e0b; }
.bt-card.real_assets { border-top-color: #3b82f6; }
.bt-card.defensive { border-top-color: #16a34a; }
.bt-card .bt-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 6px; }
.bt-card .bt-value { font-size: 1.5rem; font-weight: 700; }
.bt-card .bt-value.pos { color: var(--green); }
.bt-card .bt-value.neg { color: var(--red); }
.bt-charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
@media (max-width: 900px) { .bt-charts { grid-template-columns: 1fr; } }
.bt-chart-box { position: relative; height: 380px; }
.alpha-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.alpha-table th { text-align: left; padding: 10px 14px; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; color: var(--text-secondary); border-bottom: 2px solid var(--border); background: #fafbfc; }
.alpha-table td { padding: 10px 14px; border-bottom: 1px solid #f3f4f6; }
.alpha-pos { color: var(--green); font-weight: 600; }
.alpha-neg { color: var(--red); font-weight: 600; }
</style>
</head>
<body>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-logo">
    <h2 id="sidebar-title">ML Stock</h2>
    <small>Dashboard</small>
  </div>
  <div class="sidebar-nav">
    <div class="nav-item active" data-section="rankings">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-4 0h4"/></svg>
      Stock Rankings
    </div>
    <div class="nav-item" data-section="models">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6m6 0h6m-6 0V9a2 2 0 012-2h2a2 2 0 012 2v10m6 0v-4a2 2 0 00-2-2h-2a2 2 0 00-2 2v4"/></svg>
      Model Performance
    </div>
    <div class="nav-item" data-section="features">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"/><path d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"/></svg>
      Feature Importance
    </div>
    <div class="nav-item" data-section="backtest" id="nav-backtest" style="display:none">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      Backtest
    </div>
  </div>
</div>

<!-- Main content -->
<div class="main">

  <!-- ============ SECTION: RANKINGS ============ -->
  <div class="section active" id="sec-rankings">

    <!-- Date tabs -->
    <div class="date-tabs" id="date-tabs"></div>

    <!-- Bucket summary cards -->
    <div class="bucket-cards" id="bucket-cards"></div>

    <div class="card">
      <h2>Stock Rankings <span style="font-weight:400;color:var(--text-secondary);font-size:0.85rem">Model-predicted return rankings by bucket</span></h2>

      <!-- Filters -->
      <div class="filter-bar">
        <div class="filter-pills" id="bucket-pills"></div>
        <div class="rank-toggle" id="rank-toggle">
          <button class="rank-btn active" data-mode="best">Best Model Rank</button>
          <button class="rank-btn" data-mode="ensemble">Ensemble Rank</button>
        </div>
      </div>

      <!-- Table -->
      <div class="table-wrap">
        <table id="rankings-table">
          <thead><tr></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ============ SECTION: MODELS ============ -->
  <div class="section" id="sec-models">
    <div class="chart-grid">
      <div class="card">
        <h2>Model Comparison (Avg Validation MSE)</h2>
        <div class="chart-box"><canvas id="modelChart"></canvas></div>
      </div>
      <div class="card">
        <h2>MSE by Model & Bucket</h2>
        <div class="chart-box"><canvas id="modelBucketChart"></canvas></div>
      </div>
    </div>
    <div class="card model-detail-table">
      <h2>Detailed Model Results</h2>
      <div class="table-wrap">
        <table id="model-table"><thead><tr></tr></thead><tbody></tbody></table>
      </div>
    </div>
  </div>

  <!-- ============ SECTION: FEATURES ============ -->
  <div class="section" id="sec-features">
    <div class="card">
      <h2>Feature Importance &mdash; Top 15 (Best Model, Averaged)</h2>
      <div style="max-width:700px;">
        <div class="chart-box"><canvas id="featChart"></canvas></div>
      </div>
      <div class="legend-bar">
        <span><span class="legend-dot" style="background:#3b82f6"></span>Momentum</span>
        <span><span class="legend-dot" style="background:#f59e0b"></span>Fundamental</span>
        <span><span class="legend-dot" style="background:#6b7280"></span>Sector</span>
      </div>
    </div>
  </div>

  <!-- ============ SECTION: BACKTEST ============ -->
  <div class="section" id="sec-backtest">
    <div class="bt-header" id="bt-header"></div>
    <div class="bt-summary" id="bt-summary"></div>
    <div class="bt-charts">
      <div class="card">
        <h2>Cumulative Returns</h2>
        <div class="bt-chart-box"><canvas id="btCumChart"></canvas></div>
        <div class="legend-bar" id="bt-cum-legend"></div>
      </div>
      <div class="card">
        <h2>Per-Bucket Stock Returns</h2>
        <div class="bt-chart-box"><canvas id="btStockChart"></canvas></div>
      </div>
    </div>
    <div class="card">
      <h2>Alpha Summary</h2>
      <table class="alpha-table" id="alpha-table">
        <thead><tr><th>Portfolio / Bucket</th><th>Picks</th><th>Total Return</th><th>vs SPY</th><th>vs QQQ</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const DATA = __DATA_JSON__;
const meta = DATA.meta;

// ---- Sidebar ----
document.getElementById('sidebar-title').textContent = meta.universe + ' Stock';

// ---- Navigation ----
const navItems = document.querySelectorAll('.nav-item');
navItems.forEach(item => {
  item.addEventListener('click', () => {
    navItems.forEach(n => n.classList.remove('active'));
    item.classList.add('active');
    document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
    document.getElementById('sec-' + item.dataset.section).classList.add('active');
  });
});

// ---- State ----
let currentBucket = 'all';
let rankMode = 'best'; // 'best' or 'ensemble'

// ---- Date tabs ----
(function() {
  const container = document.getElementById('date-tabs');
  DATA.dates.forEach((d, i) => {
    const btn = document.createElement('div');
    btn.className = 'date-tab' + (i === DATA.dates.length - 1 ? ' active' : '');
    btn.textContent = d;
    container.appendChild(btn);
  });
})();

// ---- Bucket cards ----
function renderBucketCards() {
  const container = document.getElementById('bucket-cards');
  container.innerHTML = '';
  DATA.bucket_cards.forEach(bc => {
    const predPct = (bc.top_pred * 100).toFixed(2);
    const predCls = bc.top_pred >= 0 ? 'pos' : 'neg';
    const predColor = bc.top_pred >= 0 ? 'var(--green)' : 'var(--red)';
    container.innerHTML += `
      <div class="bucket-card ${bc.bucket}">
        <div class="bc-header">
          <span class="bc-bucket ${bc.bucket}">${bc.bucket.replace(/_/g, ' ')}</span>
          <span class="bc-count">${bc.count} stocks</span>
        </div>
        <div class="bc-tic">#1 ${bc.top_tic}</div>
        <div class="bc-pred" style="color:${predColor}">Pred: ${bc.top_pred >= 0 ? '+' : ''}${predPct}%</div>
        <div class="bc-model">${bc.top_model}</div>
      </div>`;
  });
}
renderBucketCards();

// ---- Bucket filter pills ----
(function() {
  const container = document.getElementById('bucket-pills');
  const buckets = ['all', ...meta.buckets];
  buckets.forEach(b => {
    const pill = document.createElement('span');
    pill.className = 'pill' + (b === 'all' ? ' active' : '');
    pill.dataset.bucket = b;
    pill.textContent = b === 'all' ? 'All Buckets' : b.replace(/_/g, ' ');
    pill.style.textTransform = 'capitalize';
    pill.addEventListener('click', () => {
      container.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentBucket = b;
      renderTable();
    });
    container.appendChild(pill);
  });
})();

// ---- Rank toggle ----
(function() {
  const btns = document.querySelectorAll('#rank-toggle .rank-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      rankMode = btn.dataset.mode;
      renderTable();
    });
  });
})();

// ---- Rankings table ----
const COL_LABELS = {
  tic: 'Ticker', bucket: 'Bucket', gsector: 'Sector',
  predicted_return: 'Predicted Return', pred_ensemble_avg: 'Ensemble Avg',
  y_return: 'Actual Return', best_model: 'Best Model',
};

function renderTable() {
  const thead = document.querySelector('#rankings-table thead tr');
  const tbody = document.querySelector('#rankings-table tbody');

  // Filter
  let rows = DATA.rankings;
  if (currentBucket !== 'all') {
    rows = rows.filter(r => r.bucket === currentBucket);
  }

  // Sort by chosen rank
  const rankCol = rankMode === 'best' ? 'rank_best' : 'rank_ensemble';
  rows = rows.slice().sort((a, b) => (a[rankCol] || 999) - (b[rankCol] || 999));

  // Columns
  const cols = DATA.columns;

  // Header
  let hdr = '<th>Rank <span class="sort-icon"></span></th>';
  cols.forEach(c => {
    hdr += '<th>' + (COL_LABELS[c] || c) + ' <span class="sort-icon">&uarr;&darr;</span></th>';
  });
  thead.innerHTML = hdr;

  // Max pred for bar scaling
  const preds = rows.map(r => r.predicted_return).filter(v => v !== null && v !== undefined);
  const maxPred = Math.max(...preds.map(Math.abs), 0.001);

  // Rows
  let html = '';
  rows.forEach((row, i) => {
    const rank = i + 1;
    const rankCls = rank <= 3 ? 'rank-' + rank : 'rank-other';
    let tr = '<td><span class="rank-num ' + rankCls + '">' + rank + '</span></td>';

    cols.forEach(c => {
      const val = row[c];
      if (c === 'bucket') {
        const b = val || '';
        tr += '<td><span class="badge badge-' + b + '">' + b.replace(/_/g, ' ') + '</span></td>';
      } else if (c === 'predicted_return' || c === 'y_return' || c === 'pred_ensemble_avg') {
        if (val === null || val === undefined) {
          tr += '<td style="color:#aaa">0.00%</td>';
        } else {
          const pct = (val * 100).toFixed(2);
          const cls = val >= 0 ? 'pos' : 'neg';
          const barW = Math.min(Math.abs(val) / maxPred * 80, 80);
          tr += '<td class="ret-cell">';
          tr += '<span class="ret-bar ' + cls + '" style="width:' + barW + '%"></span>';
          tr += '<span class="ret-val ' + cls + '">' + (val >= 0 ? '+' : '') + pct + '%</span>';
          tr += '</td>';
        }
      } else if (val === null || val === undefined) {
        tr += '<td style="color:#ccc">&mdash;</td>';
      } else {
        tr += '<td>' + val + '</td>';
      }
    });
    html += '<tr>' + tr + '</tr>';
  });
  tbody.innerHTML = html;
}
renderTable();

// ---- Model Performance Chart ----
(function() {
  if (DATA.models.length === 0) return;
  const ctx = document.getElementById('modelChart').getContext('2d');
  const sorted = DATA.models.slice().sort((a, b) => a.mse - b.mse);
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(m => m.model),
      datasets: [{
        data: sorted.map(m => m.mse),
        backgroundColor: sorted.map(m => m.is_best ? '#16a34a' : '#a5b4fc'),
        borderRadius: 6, barThickness: 28,
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => 'MSE: ' + ctx.raw.toFixed(6) } }
      },
      scales: {
        x: { title: { display: true, text: 'Mean Squared Error', font: { size: 12 } }, beginAtZero: true, grid: { color: '#f0f0f0' } },
        y: { ticks: { font: { size: 13, weight: '600' } }, grid: { display: false } }
      }
    }
  });
})();

// ---- Model x Bucket Chart ----
(function() {
  if (DATA.model_detail.length === 0) return;
  const ctx = document.getElementById('modelBucketChart').getContext('2d');
  const buckets = [...new Set(DATA.model_detail.map(d => d.bucket))];
  const models = [...new Set(DATA.model_detail.map(d => d.model))];
  const bucketColors = {
    'growth_tech': '#6c47ff', 'cyclical': '#f59e0b', 'real_assets': '#3b82f6',
    'defensive': '#16a34a', 'unified_all': '#8b5cf6',
  };
  const datasets = buckets.map(b => ({
    label: b.replace(/_/g, ' '),
    data: models.map(m => {
      const d = DATA.model_detail.find(x => x.bucket === b && x.model === m);
      return d ? d.val_mse : 0;
    }),
    backgroundColor: bucketColors[b] || '#94a3b8',
    borderRadius: 4,
  }));
  new Chart(ctx, {
    type: 'bar',
    data: { labels: models, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 16, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.raw.toFixed(6) } }
      },
      scales: {
        y: { title: { display: true, text: 'MSE', font: { size: 12 } }, beginAtZero: true, grid: { color: '#f0f0f0' } },
        x: { ticks: { font: { size: 11 } }, grid: { display: false } }
      }
    }
  });
})();

// ---- Model detail table ----
(function() {
  if (DATA.model_detail.length === 0) return;
  const thead = document.querySelector('#model-table thead tr');
  const tbody = document.querySelector('#model-table tbody');
  thead.innerHTML = '<th>Bucket</th><th>Model</th><th>Val MSE</th><th>Train</th><th>Val</th><th>Infer</th>';

  // Find best MSE per bucket
  const bestMSE = {};
  DATA.model_detail.forEach(d => {
    if (!bestMSE[d.bucket] || d.val_mse < bestMSE[d.bucket]) bestMSE[d.bucket] = d.val_mse;
  });

  let html = '';
  DATA.model_detail.forEach(d => {
    const isBest = d.val_mse === bestMSE[d.bucket];
    html += '<tr>';
    html += '<td><span class="badge badge-' + d.bucket + '">' + d.bucket.replace(/_/g, ' ') + '</span></td>';
    html += '<td>' + d.model + '</td>';
    html += '<td class="' + (isBest ? 'mse-best' : '') + '">' + d.val_mse.toFixed(6) + (isBest ? ' ★' : '') + '</td>';
    html += '<td>' + d.train_size + '</td><td>' + d.val_size + '</td><td>' + d.infer_size + '</td>';
    html += '</tr>';
  });
  tbody.innerHTML = html;
})();

// ---- Feature Importance Chart ----
(function() {
  if (DATA.features.length === 0) return;
  const ctx = document.getElementById('featChart').getContext('2d');
  const feats = DATA.features.slice().reverse();
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: feats.map(f => f.feature),
      datasets: [{
        data: feats.map(f => f.importance),
        backgroundColor: feats.map(f => f.color),
        borderRadius: 5, barThickness: 20,
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => 'Importance: ' + ctx.raw.toFixed(4) } }
      },
      scales: {
        x: { title: { display: true, text: 'Importance', font: { size: 12 } }, beginAtZero: true, grid: { color: '#f0f0f0' } },
        y: { ticks: { font: { size: 12, weight: '500' } }, grid: { display: false } }
      }
    }
  });
})();

// ---- Backtest Tab ----
(function() {
  if (!DATA.backtest) return;
  const bt = DATA.backtest;
  const s = bt.summary;

  const bucketColors = {
    'growth_tech': '#6c47ff', 'cyclical': '#f59e0b', 'real_assets': '#3b82f6',
    'defensive': '#16a34a', 'unified_all': '#8b5cf6',
  };
  const bucketNames = {
    'growth_tech': 'Growth/Tech', 'cyclical': 'Cyclical', 'real_assets': 'Real Assets',
    'defensive': 'Defensive', 'unified_all': 'Unified',
  };

  // Show nav item
  document.getElementById('nav-backtest').style.display = '';

  // Header
  document.getElementById('bt-header').innerHTML =
    '<strong>Backtest:</strong> ' + s.period + ' (' + s.trading_days + ' trading days) &mdash; Per-bucket Top 5, equal-weight across buckets';

  // Summary cards: Portfolio + each bucket + SPY + QQQ
  function fmtPct(v) { return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%'; }
  function clsPct(v) { return v >= 0 ? 'pos' : 'neg'; }
  const cards = [{ label: 'Combined Portfolio', value: s.portfolio_ret, cls: 'portfolio' }];
  const activeBuckets = Object.keys(bt.bucket_totals || {});
  activeBuckets.forEach(b => {
    cards.push({ label: bucketNames[b] || b, value: bt.bucket_totals[b], cls: b });
  });
  cards.push({ label: 'SPY', value: s.spy_ret, cls: 'spy' });
  cards.push({ label: 'QQQ', value: s.qqq_ret, cls: 'qqq' });

  const sumEl = document.getElementById('bt-summary');
  sumEl.innerHTML = cards.map(c =>
    '<div class="bt-card ' + c.cls + '">' +
    '<div class="bt-label">' + c.label + '</div>' +
    '<div class="bt-value ' + clsPct(c.value) + '">' + fmtPct(c.value) + '</div>' +
    '</div>'
  ).join('');

  // Cumulative return line chart: portfolio + per-bucket + SPY + QQQ
  const cumCtx = document.getElementById('btCumChart').getContext('2d');
  const datasets = [
    { label: 'Combined', data: bt.cumulative.portfolio.map(v => v * 100), borderColor: '#6c47ff', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 3 },
  ];
  activeBuckets.forEach(b => {
    if (bt.cumulative.buckets && bt.cumulative.buckets[b]) {
      datasets.push({ label: bucketNames[b] || b, data: bt.cumulative.buckets[b].map(v => v * 100), borderColor: bucketColors[b] || '#94a3b8', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 1.5, borderDash: [4, 2] });
    }
  });
  datasets.push({ label: 'SPY', data: bt.cumulative.spy.map(v => v * 100), borderColor: '#6b7280', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [6, 3] });
  datasets.push({ label: 'QQQ', data: bt.cumulative.qqq.map(v => v * 100), borderColor: '#f59e0b', fill: false, tension: 0.3, pointRadius: 0, borderWidth: 2, borderDash: [6, 3] });

  new Chart(cumCtx, {
    type: 'line',
    data: { labels: bt.dates, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { usePointStyle: true, padding: 12, font: { size: 11 } } },
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10, font: { size: 11 } }, grid: { color: '#f0f0f0' } },
        y: { title: { display: true, text: 'Cumulative Return (%)', font: { size: 12 } }, grid: { color: '#f0f0f0' },
             ticks: { callback: function(v) { return v.toFixed(1) + '%'; } } }
      }
    }
  });

  // Individual stock bar chart (grouped by bucket)
  const stockCtx = document.getElementById('btStockChart').getContext('2d');
  const sr = bt.stock_returns;
  const spyRet = s.spy_ret * 100;
  const qqqRet = s.qqq_ret * 100;
  new Chart(stockCtx, {
    type: 'bar',
    data: {
      labels: sr.map(s => s.tic + ' [' + (bucketNames[s.bucket] || s.bucket).substring(0, 4) + ']'),
      datasets: [{
        label: 'Actual Return',
        data: sr.map(s => s.total_return * 100),
        backgroundColor: sr.map(s => bucketColors[s.bucket] || (s.total_return >= 0 ? 'rgba(22,163,74,0.7)' : 'rgba(220,38,38,0.7)')),
        borderRadius: 4, barThickness: 16,
      }]
    },
    plugins: [{
      id: 'refLines',
      afterDraw: function(chart) {
        const area = chart.chartArea;
        const ctx = chart.ctx;
        function drawRef(val, color, label) {
          const x = chart.scales.x.getPixelForValue(val);
          if (x < area.left || x > area.right) return;
          ctx.save();
          ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
          ctx.beginPath(); ctx.moveTo(x, area.top); ctx.lineTo(x, area.bottom); ctx.stroke();
          ctx.fillStyle = color; ctx.font = '11px Inter'; ctx.textAlign = 'center';
          ctx.fillText(label, x, area.top - 6);
          ctx.restore();
        }
        drawRef(spyRet, '#6b7280', 'SPY');
        drawRef(qqqRet, '#f59e0b', 'QQQ');
      }
    }],
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: function(ctx) { return ctx.raw.toFixed(2) + '%'; } } }
      },
      scales: {
        y: { ticks: { font: { size: 11, weight: '600' } }, grid: { display: false } },
        x: { title: { display: true, text: 'Return (%)', font: { size: 12 } }, grid: { color: '#f0f0f0' },
             ticks: { callback: function(v) { return v.toFixed(1) + '%'; } } }
      }
    }
  });

  // Alpha table: per-bucket rows + combined + benchmarks
  const tbody = document.querySelector('#alpha-table tbody');
  function alphaCell(v) {
    const cls = v >= 0 ? 'alpha-pos' : 'alpha-neg';
    return '<td class="' + cls + '">' + fmtPct(v) + '</td>';
  }
  let rows = '';
  activeBuckets.forEach(b => {
    const bRet = bt.bucket_totals[b] || 0;
    const picks = (bt.bucket_picks[b] || []).join(', ');
    rows += '<tr><td><span class="badge badge-' + b + '">' + (bucketNames[b] || b) + '</span></td>';
    rows += '<td style="font-size:0.8rem">' + picks + '</td>';
    rows += '<td>' + fmtPct(bRet) + '</td>';
    rows += alphaCell(bRet - s.spy_ret) + alphaCell(bRet - s.qqq_ret) + '</tr>';
  });
  rows += '<tr style="font-weight:700;border-top:2px solid var(--border)"><td>Combined Portfolio</td><td></td><td>' + fmtPct(s.portfolio_ret) + '</td>' + alphaCell(s.alpha_spy) + alphaCell(s.alpha_qqq) + '</tr>';
  rows += '<tr><td>SPY</td><td></td><td>' + fmtPct(s.spy_ret) + '</td><td>—</td><td>—</td></tr>';
  rows += '<tr><td>QQQ</td><td></td><td>' + fmtPct(s.qqq_ret) + '</td><td>—</td><td>—</td></tr>';
  tbody.innerHTML = rows;
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate HTML dashboard from ML dashboard Excel")
    parser.add_argument("excel_path", help="Path to the dashboard Excel file")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open in browser")
    parser.add_argument("-o", "--output", help="Output HTML path (default: same dir as Excel)")
    parser.add_argument("--buy-date", help="Backtest buy date (YYYY-MM-DD). Default: inferred from tradedate column")
    parser.add_argument("--sell-date", help="Backtest sell date (YYYY-MM-DD). Default: today")
    args = parser.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"Error: File not found: {args.excel_path}")
        sys.exit(1)

    print(f"Loading: {args.excel_path}")
    rankings, models, features = load_excel(args.excel_path)
    print(f"  Rankings: {len(rankings)} stocks, Models: {len(models)} rows, Features: {len(features)} rows")

    html = build_html(rankings, models, features, args.excel_path,
                      buy_date=args.buy_date, sell_date=args.sell_date)

    if args.output:
        out_path = args.output
    else:
        out_path = args.excel_path.replace(".xlsx", ".html")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    if not args.no_open:
        webbrowser.open("file://" + os.path.abspath(out_path))
        print("Opened in browser.")


if __name__ == "__main__":
    main()
