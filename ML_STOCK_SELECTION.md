# ML Stock Selection Pipeline

End-to-end guide for ML-driven stock selection with per-sector-bucket models.

**Database:** 22,909 records, 715 tickers, 64 columns (2015-Q2 ~ 2026-Q1)

**Output:** Ranked stock picks per bucket (growth_tech, cyclical, real_assets, defensive) with 7 competing ML models.

---

## CRITICAL SPEC 1: y_return Calculation

> **This is the single most important calculation in the entire ML strategy. Always verify y_return correctness before running any model.**

### Definition

```
y_return[i] = ln(next_trade_price / this_trade_price)
```

y_return is the **quarterly log return measured at tradedate prices** — the actual return an investor earns by buying at this tradedate and selling at the next tradedate.

### The Chain

```
datadate → tradedate → actual_tradedate → trade_price → y_return
```

| Step | Field | Meaning | Example (AMD Q3 2025) |
|------|-------|---------|----------------------|
| 1 | `datadate` | Quarter-end date of the financial report | 2025-09-30 |
| 2 | `tradedate` | Theoretical trade date (datadate + 2 months) | 2025-12-01 |
| 3 | `actual_tradedate` | First NYSE trading day on or after tradedate | 2025-12-01 |
| 4 | `trade_price` | adjClose on actual_tradedate | 219.76 |
| 5 | `y_return` | ln(next trade_price / this trade_price) | ln(198.62/219.76) = **-0.1011** |

### datadate → tradedate Mapping

| datadate (quarter end) | tradedate | Rationale |
|------------------------|-----------|-----------|
| 03-31 | 06-01 (same year) | Q1 report available by June |
| 06-30 | 09-01 (same year) | Q2 report available by September |
| 09-30 | 12-01 (same year) | Q3 report available by December |
| 12-31 | 03-01 (next year) | Q4 report available by March |

### Worked Example (AMD)

```
datadate    tradedate    trade_price    y_return = ln(next/this)
─────────   ─────────    ───────────    ─────────────────────────
2025-03-31  2025-06-01   114.63         ln(162.32/114.63) = +0.3479
2025-06-30  2025-09-01   162.32         ln(219.76/162.32) = +0.3030
2025-09-30  2025-12-01   219.76         ln(198.62/219.76) = -0.1011
2025-12-31  2026-03-01   198.62         NULL (next tradedate not yet reached)
```

### Common Mistakes

| Wrong approach | Why it's wrong | Correct approach |
|----------------|----------------|------------------|
| `ln(adj_close_q[t+1] / adj_close_q[t])` | adj_close_q is the quarter-end price; the investor cannot buy at quarter-end because the report isn't public yet | Use `trade_price` (price at tradedate) |
| `ln(price_2025-12-31 / price_2025-09-30)` | On 2025-09-30 the Q3 report is not yet public; you can't act on it until tradedate 2025-12-01 | Buy at tradedate, not quarter-end |
| y_return = 0 | Frozen price from delisted ticker whose adj_close_q was never updated | Set to NULL |

### Pre-Run Verification (MUST execute before every model run)

```python
import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('data/finrl_trading.db')
df = pd.read_sql('''
    SELECT ticker, datadate, trade_price, y_return
    FROM fundamental_data ORDER BY ticker, datadate
''', conn)
conn.close()
df['trade_price'] = pd.to_numeric(df['trade_price'], errors='coerce')
df['y_return'] = pd.to_numeric(df['y_return'], errors='coerce')
df['next_tp'] = df.groupby('ticker')['trade_price'].shift(-1)
df['expected'] = np.where(
    (df['trade_price'] > 0) & (df['next_tp'] > 0),
    np.log(df['next_tp'] / df['trade_price']), np.nan
)
diff = (df['y_return'] - df['expected']).abs()
bad = diff[diff > 0.0001].dropna()
assert len(bad) == 0, f"y_return mismatch: {len(bad)} rows"
assert (df['y_return'] == 0).sum() == 0, "Found y_return = 0 (frozen price)"
print("y_return verification passed")
```

### Full Recompute Procedure

When trade_price has been updated, recompute all y_return:

```python
import sqlite3, pandas as pd, numpy as np
conn = sqlite3.connect('data/finrl_trading.db')
df = pd.read_sql('SELECT ticker, datadate, trade_price FROM fundamental_data ORDER BY ticker, datadate', conn)
df['trade_price'] = pd.to_numeric(df['trade_price'], errors='coerce')
df['next_tp'] = df.groupby('ticker')['trade_price'].shift(-1)
df['y_return_new'] = np.where(
    (df['trade_price'] > 0) & (df['next_tp'] > 0),
    np.log(df['next_tp'] / df['trade_price']), np.nan
)
cursor = conn.cursor()
for _, row in df.iterrows():
    val = None if pd.isna(row['y_return_new']) else round(float(row['y_return_new']), 6)
    cursor.execute('UPDATE fundamental_data SET y_return = ? WHERE ticker = ? AND datadate = ?',
                   (val, row['ticker'], row['datadate']))
conn.commit()
conn.close()
```

---

## CRITICAL SPEC 2: Train / Inference Universe Construction (Point-in-Time)

> **Every training sample and every inference sample must pass the same point-in-time SP500 membership check. This is the core anti-leakage mechanism.**

### Three Principles

| Principle | Rule | Rationale |
|-----------|------|-----------|
| **A. Universe is point-in-time** | For date d, only include tickers that were known SP500 members on date d | Prevents using future index composition to construct historical cross-sections |
| **B. Feature history can look back** | Compute features (momentum, QoQ changes) on full historical data before filtering | A ticker's own price/fundamental history is public and can be used for feature engineering |
| **C. Train and inference rules are consistent** | Same `membership(tradedate, ticker)` check for both train and inference | Prevents train/test distribution mismatch |

### How It Works

For every row `(ticker, datadate)` in the database:

```
1. Map datadate → tradedate
2. Look up SP500 membership at tradedate (latest snapshot on or before tradedate)
3. Keep the row only if ticker ∈ SP500(tradedate)
```

This filter applies **uniformly** to all rows — training, validation, and inference.

### CIEN Example (added to SP500 on 2026-02-09)

**Scenario: Stock selection on 2026-03-01** (datadate=2025-12-31, tradedate=2026-03-01)

| datadate | tradedate | CIEN in SP500? | Result |
|----------|-----------|----------------|--------|
| 2025-03-31 | 2025-06-01 | No | **Excluded** from training |
| 2025-06-30 | 2025-09-01 | No | **Excluded** from training |
| 2025-09-30 | 2025-12-01 | No | **Excluded** from training |
| 2025-12-31 | 2026-03-01 | **Yes** (added 02-09) | **Included** in inference |

**Scenario: Stock selection on 2025-12-01** (datadate=2025-09-30, tradedate=2025-12-01)

CIEN is excluded from **both** training and inference — it was not an SP500 member on 2025-12-01.

### LITE Example (added to SP500 on 2026-03-10)

**Scenario: Stock selection on 2026-03-01** (tradedate=2026-03-01)

LITE is **excluded** — it was added on 03-10, after the tradedate of 03-01.

### Pipeline Execution Order

```
1. Load ALL records from DB (22,909 rows, 715 tickers)
2. Compute momentum features on full data    ← Principle B: feature history looks back
3. Prep features (fillna, winsorize)         ← Still on full data
4. Apply --infer-date filter (if set)        ← Keep only specified inference quarter
5. POINT-IN-TIME SP500 FILTER               ← Principle A & C: uniform membership check
6. Assign sector buckets
7. Train / validate / infer per bucket
```

Features are computed **before** filtering (Step 2-3), so a ticker's own historical data contributes to its momentum and QoQ features. The membership filter (Step 5) then determines which rows enter training and inference.

### Membership Data Source

File: `data/sp500_historical_constituents.csv`

- 2,709 daily snapshots from 1996-01-02 to 2026-04-17
- Each row: `date, tickers` (comma-separated list of ~503 members)
- Key 2026 transitions:

| Date | Change |
|------|--------|
| 2026-02-09 | CIEN replaces DAY |
| 2026-03-10 | VRT, LITE, COHR, SATS replace MTCH, MOH, LW, PAYC |
| 2026-04-09 | CASY replaces HOLX |

### Implementation

Code: `src/strategies/ml_bucket_selection.py` lines 542-586

```python
DATADATE_TO_TRADEDATE_MAP = {
    "03-31": ("06-01", 0),
    "06-30": ("09-01", 0),
    "09-30": ("12-01", 0),
    "12-31": ("03-01", 1),
}

def datadate_to_tradedate(datadate_str):
    mm_dd = datadate_str[5:]
    year = int(datadate_str[:4])
    mapping = DATADATE_TO_TRADEDATE_MAP.get(mm_dd)
    if mapping is None:
        return None
    target_mmdd, year_add = mapping
    return f"{year + year_add}-{target_mmdd}"

# For each tradedate, look up SP500 members and keep only matching rows
for td in unique_tradedates:
    sp500_members = get_sp500_at(td)
    td_mask = df["_tradedate_pit"] == td
    pit_keep = pit_keep | (td_mask & df["tic"].isin(sp500_members))
```

### Expected Counts (as of 2026-04-17)

| Selection Date | --val-cutoff | --infer-date | Train | Val | Inference |
|----------------|-------------|--------------|-------|-----|-----------|
| 2025-12-01 | 2025-06-30 | 2025-09-30 | ~17,000 | ~980 | ~496 |
| 2026-03-01 | 2025-09-30 | 2025-12-31 | ~19,000 | ~500 | ~503 |

---

## Overview

```
Step 1: Setup              →  .env, venv, dependencies
Step 2: Fetch Data         →  FMP API → SQLite + CSV
Step 2b: Backfill History  →  Fill ex-SP500 members
Step 3: Data Cleaning      →  Fix trade_price, recompute y_return
Step 4: Run ML             →  4 bucket models → predictions CSV
Step 5: Mixed-Vintage Run  →  Use latest available data per ticker for today's picks
```

---

## Prerequisites

### 1. Environment Variables

Create `.env` in project root:

```bash
FMP_API_KEY=your_fmp_api_key_here
```

### 2. Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn requests pyyaml pandas-market-calendars tzlocal tqdm
pip install lightgbm xgboost  # optional but recommended
brew install libomp            # macOS only, for LightGBM
```

---

## Step 1: Fetch & Store Fundamental Data

**Script:** `src/data/fetch_and_store_fundamentals.py`

```bash
python3 src/data/fetch_and_store_fundamentals.py \
    --start-date 2015-01-01 --end-date 2026-04-01
```

Fetches SP500 universe via FMP API, calls 4 endpoints per ticker (income-statement, balance-sheet, cash-flow, ratios), computes 52 fundamental factors per quarter, and stores to SQLite.

**52 Factors:**

| Category | Factors |
|----------|---------|
| Valuation (7) | PE, PS, PB, PEG, price_to_fcf, price_to_ocf, ev_multiple |
| Profitability (9) | EPS, ROE, net_income_ratio, gross_margin, operating_margin, ebitda_margin, pretax_margin, effective_tax_rate, ebt_per_ebit |
| Liquidity (3) | cur_ratio, quick_ratio, cash_ratio |
| Leverage (8) | debt_ratio, debt_to_equity, debt_to_assets, debt_to_capital, lt_debt_to_capital, interest_coverage, debt_service_coverage, debt_to_mktcap |
| Efficiency (6) | acc_rec_turnover, asset_turnover, fixed_asset_turnover, inventory_turnover, payables_turnover, wc_turnover |
| Cash Flow (10) | fcf_per_share, ocf_per_share, cash_per_share, capex_per_share, fcf_to_ocf, ocf_ratio, ocf_to_sales, ocf_coverage, st_ocf_coverage, capex_coverage |
| Per-Share (5) | BPS, DPS, revenue_per_share, tangible_bvps, interest_debt_per_share |
| Dividend (3) | dividend_payout, dividend_yield, div_capex_coverage |
| Solvency (1) | solvency_ratio |

**Removed (collinear):** `price_to_fair_value` (= PB), `financial_leverage` (= D/E + 1), `net_income_per_ebt` (= 1 - ETR)

---

## Step 1b: Backfill Historical SP500 Members

**Script:** `src/data/backfill_historical_sp500.py`

Fills gaps for ex-SP500 members using `sp500_historical_constituents.csv`.

```bash
python3 src/data/backfill_historical_sp500.py
```

1. Identifies missing (ticker, quarter) pairs by comparing SP500 membership vs DB
2. Fetches fundamentals from FMP for missing tickers
3. Fills tradedate, actual_tradedate, trade_price
4. Recomputes y_return using trade_price (see CRITICAL SPEC 1)

---

## Step 2: Run ML Bucket Selection

**Script:** `src/strategies/ml_bucket_selection.py`

```bash
# Backtest 2025/12/01 selection
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-06-30 --infer-date 2025-09-30

# Backtest 2026/03/01 selection
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --infer-date 2025-12-31

# Mixed-vintage (today's live picks)
python3 src/strategies/ml_bucket_selection.py \
    --val-cutoff 2025-09-30 --mixed-vintage
```

**Arguments:**

| Arg | Default | Description |
|-----|---------|-------------|
| `--val-cutoff` | 2025-12-31 | Last quarter whose y_return is fully realized |
| `--val-quarters` | 3 | Number of validation quarters |
| `--infer-date` | None | Only infer on this specific datadate (e.g., 2025-12-31) |
| `--mixed-vintage` | off | Inference: per current-SP500 ticker, use latest available datadate |
| `--db` | data/finrl_trading.db | SQLite database path |
| `--output-dir` | data/ | Output directory for CSVs |

### Sector-to-Bucket Mapping

```
growth_tech:  Information Technology, Communication Services
cyclical:     Consumer Discretionary, Financials, Industrials
real_assets:  Energy, Materials, Real Estate
defensive:    Healthcare, Consumer Staples, Utilities
```

### Feature Pipeline

1. **Momentum features** computed on full data (before PIT filter):
   - Price: ret_1q, ret_2q, ret_4q (from adj_close_q pct_change)
   - Fundamental: eps_chg, roe_chg, gm_chg, om_chg (QoQ diffs)
2. Fill missing with global median, replace inf with 0
3. **Winsorize** at 1st/99th percentile
4. **Point-in-time SP500 filter** (see CRITICAL SPEC 2)
5. One-hot encode gsector as sub-sector indicators within each bucket
6. StandardScaler before model training

### val-cutoff Logic

`--val-cutoff` must be the last datadate whose `y_return` is fully realized (both tradedates in the past).

```
datadate    tradedate    y_return                            status (2026-04-17)
─────────   ─────────    ──────────────────────              ────────────────────
2025-06-30  2025-09-01   ln(P_2025-12-01 / P_2025-09-01)    realized → train/val
2025-09-30  2025-12-01   ln(P_2026-03-01 / P_2025-12-01)    realized → val (last)
2025-12-31  2026-03-01   ln(P_2026-06-01 / P_2026-03-01)    NOT YET  → inference
```

**Therefore `--val-cutoff 2025-09-30`** as of April 2026.

### Train / Validation / Inference Split

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAIN: ~38 quarters (2015-Q2 ~ 2024-Q4)                       │
│  Point-in-time SP500 filtered, y_return fully realized          │
│  ~17,000 records, ~660 unique tickers                           │
├─────────────────────────────────────────────────────────────────┤
│  VALIDATION: 2-3 quarters up to val_cutoff                      │
│  Used for model selection (best MSE wins)                       │
│  After selection, retrain on train+val combined                 │
├─────────────────────────────────────────────────────────────────┤
│  INFERENCE: datadate = --infer-date (or all > val_cutoff)       │
│  Point-in-time SP500 filtered, ~496-503 stocks                  │
│  Predictions generated by best model per bucket                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mixed-Vintage Mode (`--mixed-vintage`)

Not all companies report at the same time. Mixed-vintage uses the **latest available** data per current-SP500 ticker:

```
  503 current SP500 tickers
    ├── ~76 early reporters → use Q1 2026 (datadate=2026-03-31)
    └── ~427 not yet reported → use Q4 2025 (datadate=2025-12-31)
```

Training data still uses point-in-time filter. All inference tickers are ranked together within each bucket.

### Model Competition (7 models per bucket)

| # | Model | Key Params |
|---|-------|------------|
| 1 | RandomForest | n_estimators=200, max_depth=8 |
| 2 | XGBoost | n_estimators=200, max_depth=6, lr=0.05 |
| 3 | LightGBM | n_estimators=200, max_depth=6, lr=0.05 |
| 4 | HistGradientBoosting | max_iter=200, max_depth=6, lr=0.05 |
| 5 | ExtraTrees | n_estimators=200, max_depth=8 |
| 6 | Ridge | alpha=1.0 (linear baseline) |
| 7 | Stacking | Top 3 by val MSE + Ridge meta-learner, cv=3 |

Best model selected by **validation MSE** (lowest wins). Each bucket may choose a different model.

### Ensemble

All 7 models predict. Final ensemble uses **inverse-MSE weighted average**:

```
weight_i = (1 / MSE_i) / Σ(1 / MSE_j)
pred_ensemble = Σ(weight_i × pred_i)
```

---

## Output Files

### `data/sp500_ml_bucket_predictions_{timestamp}.csv`

| Column | Description |
|--------|-------------|
| tic | Ticker symbol |
| datadate | Quarter date |
| bucket | growth_tech / cyclical / real_assets / defensive |
| y_return | Actual forward return (NaN if future) |
| predicted_return | Best model's predicted return |
| pred_RF...pred_Stacking | Individual model predictions |
| pred_ensemble_avg | Inverse-MSE weighted ensemble |
| rank_best | Within-bucket rank by best model |
| rank_ensemble | Within-bucket rank by ensemble |

### `data/sp500_ml_bucket_model_results_{timestamp}.csv`

Per-model validation metrics (bucket, model, val_mse, train/val/infer sizes).

### `data/sp500_ml_feature_importance_{timestamp}.csv`

Per-feature importance for each model in each bucket.

---

## Backtesting Guide

| Selection Date | What You're Testing | Command |
|----------------|--------------------|---------|
| 2025/12/01 | Buy 12/1, sell 3/1 | `--val-cutoff 2025-06-30 --infer-date 2025-09-30` |
| 2026/03/01 | Buy 3/1, sell 6/1 | `--val-cutoff 2025-09-30 --infer-date 2025-12-31` |

The mapping: to select stocks on tradedate T, use `--infer-date` = the datadate that maps to T.

| tradedate (selection) | datadate (--infer-date) | val-cutoff |
|----------------------|------------------------|------------|
| 2025-06-01 | 2025-03-31 | 2024-12-31 |
| 2025-09-01 | 2025-06-30 | 2025-03-31 |
| 2025-12-01 | 2025-09-30 | 2025-06-30 |
| 2026-03-01 | 2025-12-31 | 2025-09-30 |

---

## Codebase Reference

```
src/data/
  data_fetcher.py                   # FMP API client, factor computation
  data_store.py                     # SQLite persistence layer
  fetch_and_store_fundamentals.py   # Fetch data from FMP → SQLite
  backfill_historical_sp500.py      # Backfill ex-SP500 members
  fix_adj_close.py                  # Fix adj_close_q via yfinance
  fill_recent_yreturn.py            # Fill y_return for latest quarters

src/strategies/
  ml_bucket_selection.py            # Main ML pipeline
                                    #   Point-in-time SP500 filter (lines 542-586)
                                    #   datadate_to_tradedate() mapping
                                    #   get_sp500_at() membership lookup
                                    #   --infer-date, --mixed-vintage

data/
  finrl_trading.db                  # SQLite (22,909 records, 715 tickers, 64 columns)
  sp500_historical_constituents.csv # SP500 membership snapshots (1996-2026, 2,709 rows)
  sp500_ml_bucket_predictions_*.csv # ML predictions (timestamped)
```

---

## Data Quality Notes

- **y_return = 0 rows:** 0 (frozen prices from delisted tickers set to NULL)
- **gsector NULL:** 0 (all 715 tickers have sector assigned)
- **Duplicates:** 0
- **9 tickers with ALL y_return NULL:** AABA, BXLT, NBL, PCL, POM, RHT, SNI, TWC, XL (delisted, no price data available; fundamentals retained for feature computation but never enter training)
- **y_return NULL causes:** last quarter per ticker (no next trade_price), frozen adj_close_q (delisted), missing trade_price

### Known Quirks

- **STI (SunTrust → Solidion):** Ticker reused. Old SunTrust data ($539) and new Solidion data ($23) coexist.
- **FRC y_return = -5.88:** First Republic Bank collapse ($122 → $0.34). Real event, kept in data.

---

## Notes

- **Rebalance cadence:** Run `--mixed-vintage` weekly during earnings season for fresher picks.
- **val-cutoff updates:** Advance by one quarter when the next tradedate becomes past.
- **FMP API limits:** `limit=40` returns ~10 years. Starter plan may return fewer.
- **Ticker format:** FMP uses `-` (BRK-B), DB uses `.` (BRK.B). Conversion is automatic.
- **random_state=42:** All models are deterministic. Same data + same params = same results.
