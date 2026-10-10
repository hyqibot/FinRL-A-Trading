#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────
STRATEGY=""
CONFIG=""
DATA_DIR="data/fmp_daily"
START_DATE="2023-01-01"
END_DATE="2024-12-31"
MODE=""                  # backtest | single | paper
SINGLE_DATE=""
SKIP_DOWNLOAD=false
FREQ="W-FRI"
NO_FAST_TRACK=false
DRY_RUN=false
ACCOUNT=""               # Alpaca account name (paper mode)

# ── Strategy registry (add new strategies here) ──────────
# Format: strategy_name|config_path|runner_path
STRATEGIES="
adaptive_rotation|src/strategies/AdaptiveRotationConf_v1.2.1.yaml|src/strategies/run_adaptive_rotation_strategy.py
"

resolve_strategy() {
    local input="$1"
    local name="" cfg="" runner=""

    # Check if input is a file path
    if [[ -f "$input" ]]; then
        cfg="$input"
        while IFS='|' read -r n c r; do
            [[ -z "$n" ]] && continue
            if [[ "$c" == "$cfg" ]]; then
                name="$n"; runner="$r"; break
            fi
        done <<< "$STRATEGIES"
        if [[ -z "$runner" ]]; then
            echo "Error: no runner registered for config '$cfg'" >&2
            echo "Register it in STRATEGIES in deploy.sh" >&2
            return 1
        fi
    else
        while IFS='|' read -r n c r; do
            [[ -z "$n" ]] && continue
            if [[ "$n" == "$input" ]]; then
                name="$n"; cfg="$c"; runner="$r"; break
            fi
        done <<< "$STRATEGIES"
        if [[ -z "$name" ]]; then
            echo "Error: unknown strategy '$input'" >&2
            echo "" >&2
            echo "Available strategies:" >&2
            while IFS='|' read -r n c r; do
                [[ -z "$n" ]] && continue
                echo "  $n  ->  $c" >&2
            done <<< "$STRATEGIES"
            echo "" >&2
            echo "Or pass a yaml config path: --strategy path/to/config.yaml" >&2
            return 1
        fi
    fi

    STRATEGY="$name"
    CONFIG="$cfg"
    RUNNER="$runner"
}

# ── Usage ─────────────────────────────────────────────────
usage() {
    cat <<EOF
FinRL Trading - Strategy Deploy & Run
======================================

Usage: $0 --strategy <NAME|CONFIG_PATH> --mode <MODE> [OPTIONS]

Strategy Selection (required):
  --strategy NAME        Strategy name or path to config yaml
                         Available strategies:
EOF
    while IFS='|' read -r n c r; do
        [[ -z "$n" ]] && continue
        echo "                           $n  -> $c"
    done <<< "$STRATEGIES"
    cat <<EOF

Run Modes (required):
  --mode backtest        Historical backtest over a date range
  --mode single          Run strategy for a single date (signal only)
  --mode paper           Generate today's signal + execute on Alpaca paper trading

Backtest Options:
  --start DATE           Backtest start date  (default: $START_DATE)
  --end DATE             Backtest end date    (default: $END_DATE)
  --freq FREQ            Rebalance frequency  (default: $FREQ)
  --no-fast-track        Disable daily Fast Risk-Off monitoring

Single / Paper Options:
  --date DATE            Decision date (default: today)

Paper Trading Options:
  --dry-run              Preview orders without executing
  --account NAME         Alpaca account name (default: from .env)

General Options:
  --data-dir PATH        Data directory       (default: $DATA_DIR)
  --skip-download        Skip data download (use existing data)
  --list                 List available strategies and exit
  --help                 Show this help

Examples:
  # Backtest
  ./deploy.sh --strategy adaptive_rotation --mode backtest
  ./deploy.sh --strategy adaptive_rotation --mode backtest --start 2020-01-01 --end 2025-12-31

  # Single date signal (no trading)
  ./deploy.sh --strategy adaptive_rotation --mode single --date 2024-12-31

  # Paper trading (generate signal + execute on Alpaca)
  ./deploy.sh --strategy adaptive_rotation --mode paper
  ./deploy.sh --strategy adaptive_rotation --mode paper --dry-run
  ./deploy.sh --strategy adaptive_rotation --mode paper --date 2024-12-31

  # Skip data download
  ./deploy.sh --strategy adaptive_rotation --mode backtest --skip-download
EOF
    exit 0
}

list_strategies() {
    echo "Available strategies:"
    echo ""
    while IFS='|' read -r n c r; do
        [[ -z "$n" ]] && continue
        echo "  $n"
        echo "    Config: $c"
        echo "    Runner: $r"
        echo ""
    done <<< "$STRATEGIES"
    echo "Config files found:"
    find src/strategies -name "*.yaml" 2>/dev/null | sort | while read -r f; do
        echo "  $f"
    done
    exit 0
}

# ── Parse args ────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy)      STRATEGY="$2";      shift 2 ;;
        --mode)          MODE="$2";          shift 2 ;;
        --start)         START_DATE="$2";    shift 2 ;;
        --end)           END_DATE="$2";      shift 2 ;;
        --date)          SINGLE_DATE="$2";   shift 2 ;;
        --config)        CONFIG="$2";        shift 2 ;;
        --data-dir)      DATA_DIR="$2";      shift 2 ;;
        --freq)          FREQ="$2";          shift 2 ;;
        --no-fast-track) NO_FAST_TRACK=true; shift   ;;
        --skip-download) SKIP_DOWNLOAD=true; shift   ;;
        --dry-run)       DRY_RUN=true;       shift   ;;
        --account)       ACCOUNT="$2";       shift 2 ;;
        --list)          list_strategies ;;
        --help|-h)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── Validate required args ────────────────────────────────
if [[ -z "$STRATEGY" ]]; then
    echo "Error: --strategy is required"
    echo ""
    echo "Available strategies:"
    while IFS='|' read -r n c r; do
        [[ -z "$n" ]] && continue
        echo "  $n  ->  $c"
    done <<< "$STRATEGIES"
    echo ""
    echo "Run ./deploy.sh --help for usage"
    exit 1
fi

if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required (backtest | single | paper)"
    echo "Run ./deploy.sh --help for usage"
    exit 1
fi

RUNNER=""
resolve_strategy "$STRATEGY"

if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

# ── Paper mode: validate .env ─────────────────────────────
if [[ "$MODE" == "paper" ]]; then
    if [[ ! -f ".env" ]]; then
        echo "Error: .env file not found (required for paper trading)"
        echo "Copy .env.example to .env and fill in your Alpaca credentials"
        exit 1
    fi
    # Quick check for placeholder values (only check main account keys)
    if grep -q "^APCA_API_KEY=your_alpaca_api_key_here" .env 2>/dev/null; then
        echo "Error: Alpaca API credentials not configured in .env"
        echo "Set APCA_API_KEY and APCA_API_SECRET to your actual keys"
        exit 1
    fi
fi

# Default date for single/paper mode
if [[ "$MODE" == "single" || "$MODE" == "paper" ]]; then
    if [[ -z "$SINGLE_DATE" ]]; then
        SINGLE_DATE=$(date +%Y-%m-%d)
    fi
fi

# ── Banner ────────────────────────────────────────────────
echo "=============================================="
echo " FinRL Trading - Strategy Deploy"
echo "=============================================="
echo " Strategy:   $STRATEGY"
echo " Config:     $CONFIG"
echo " Mode:       $MODE"
if [[ "$MODE" == "backtest" ]]; then
    echo " Period:     $START_DATE ~ $END_DATE"
    echo " Frequency:  $FREQ"
    echo " Fast Track: $([ "$NO_FAST_TRACK" = true ] && echo Disabled || echo Enabled)"
elif [[ "$MODE" == "single" ]]; then
    echo " Date:       $SINGLE_DATE"
elif [[ "$MODE" == "paper" ]]; then
    echo " Date:       $SINGLE_DATE"
    echo " Dry Run:    $([ "$DRY_RUN" = true ] && echo Yes || echo No)"
    echo " Account:    ${ACCOUNT:-default}"
fi
echo " Data Dir:   $DATA_DIR"
echo "=============================================="

# ── Step 1: Install dependencies ──────────────────────────
echo ""
echo "[1/3] Checking dependencies..."

MISSING=()
python3 -c "import yaml"                    2>/dev/null || MISSING+=("pyyaml")
python3 -c "import pandas_market_calendars" 2>/dev/null || MISSING+=("pandas-market-calendars")
python3 -c "import yfinance"                2>/dev/null || MISSING+=("yfinance")
python3 -c "import pandas"                  2>/dev/null || MISSING+=("pandas")
python3 -c "import numpy"                   2>/dev/null || MISSING+=("numpy")
python3 -c "import scipy"                   2>/dev/null || MISSING+=("scipy")

if [[ "$MODE" == "paper" ]]; then
    python3 -c "import dotenv"   2>/dev/null || MISSING+=("python-dotenv")
    python3 -c "import requests" 2>/dev/null || MISSING+=("requests")
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "  Installing: ${MISSING[*]}"
    pip3 install -q "${MISSING[@]}"
else
    echo "  All dependencies OK"
fi

# ── Step 2: Download data ─────────────────────────────────
echo ""
echo "[2/3] Preparing data..."

if [[ "$SKIP_DOWNLOAD" == true ]] && [[ -d "$DATA_DIR" ]]; then
    FILE_COUNT=$(ls "$DATA_DIR"/*.csv 2>/dev/null | wc -l | tr -d ' ')
    echo "  Skipping download ($FILE_COUNT files in $DATA_DIR)"
else
    mkdir -p "$DATA_DIR"
    python3 - "$CONFIG" "$DATA_DIR" <<'PYEOF'
import sys, yaml, yfinance as yf, pandas as pd
from pathlib import Path

config_path, data_dir = sys.argv[1], Path(sys.argv[2])

with open(config_path) as f:
    config = yaml.safe_load(f)

# Extract all symbols from config
symbols = set()

for group_name, group in config.get("asset_groups", {}).items():
    for sym in group.get("symbols", []):
        symbols.add(sym)

fallback = config.get("portfolio", {}).get("fallback", {})
for sym in fallback.get("symbols", []):
    symbols.add(sym)

bench = config.get("benchmark", {})
if "excess_return_benchmark" in bench:
    symbols.add(bench["excess_return_benchmark"])

symbols.update(["^GSPC", "^VIX", "SPY", "QQQ"])

symbols = sorted(symbols)
start = config.get("dates", {}).get("start_date", "2017-01-01")
print(f"  Extracted {len(symbols)} symbols from {config_path}")
print(f"  Download from {start}")

failed = []
for sym in symbols:
    try:
        df = yf.download(sym, start=start, progress=False, auto_adjust=False)
        if df.empty:
            failed.append(sym); continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df[["date","open","high","low","close","volume"]]
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df.sort_values("date").to_csv(data_dir / f"{sym}_daily.csv", index=False)
        print(f"    OK: {sym} ({len(df)} rows)")
    except Exception as e:
        print(f"    FAIL: {sym} - {e}")
        failed.append(sym)

print(f"  Done: {len(symbols)-len(failed)}/{len(symbols)} succeeded")
if failed:
    print(f"  Failed: {failed}")
    sys.exit(1)
PYEOF
fi

# ── Step 3: Run strategy ─────────────────────────────────
echo ""
echo "[3/3] Running strategy..."

if [[ "$MODE" == "backtest" ]]; then
    # ── Backtest mode ──────────────────────────────────
    CMD=(python3 "$RUNNER" --config "$CONFIG" --data-dir "$DATA_DIR")
    CMD+=(--backtest --start "$START_DATE" --end "$END_DATE" --freq "$FREQ")
    [[ "$NO_FAST_TRACK" == true ]] && CMD+=(--no-daily-fast-track)
    "${CMD[@]}"

elif [[ "$MODE" == "single" ]]; then
    # ── Single date mode (signal only) ─────────────────
    CMD=(python3 "$RUNNER" --config "$CONFIG" --data-dir "$DATA_DIR")
    CMD+=(--date "$SINGLE_DATE")
    "${CMD[@]}"

elif [[ "$MODE" == "paper" ]]; then
    # ── Paper trading mode ─────────────────────────────
    # Step 3a: Generate signal, then 3b: Execute on Alpaca
    python3 - "$CONFIG" "$DATA_DIR" "$SINGLE_DATE" "$DRY_RUN" "$ACCOUNT" <<'PYEOF'
import sys, json
from pathlib import Path
from datetime import datetime

config_path = sys.argv[1]
data_dir = sys.argv[2]
as_of_date = sys.argv[3]
dry_run = sys.argv[4].lower() == "true"
account_name = sys.argv[5] if sys.argv[5] else None

# Add project root to path
project_root = Path(__file__).resolve().parent if Path(__file__).exists() else Path.cwd()
sys.path.insert(0, str(project_root))

# ── 3a: Generate target weights ──────────────────────
print(f"\n--- Step 3a: Generating signal for {as_of_date} ---\n")

from src.strategies.adaptive_rotation import AdaptiveRotationEngine
from src.strategies.adaptive_rotation.data_preprocessor import DataPreprocessor
from src.strategies.adaptive_rotation.config_loader import load_config

config = load_config(config_path)
preprocessor = DataPreprocessor(config)
preprocessor.load_and_prepare(data_dir=data_dir)

engine = AdaptiveRotationEngine(config=config_path, data_preprocessor=preprocessor)
config = engine.get_config()

raw_data = preprocessor.get_data_as_of(as_of_date)
price_data = {symbol: df['close'] for symbol, df in raw_data.items()}

weights, audit_log = engine.run(
    price_data=price_data,
    as_of_date=as_of_date
)

print(f"  Market Regime: {weights.regime_state}")
print(f"  Total Invested: {weights.get_invested_weight():.2%}")
print(f"  Cash Position:  {weights.cash_weight:.2%}")
print(f"\n  Target Portfolio ({len(weights.weights)} assets):")
print("  " + "-" * 36)
for symbol, weight in sorted(weights.weights.items(), key=lambda x: x[1], reverse=True):
    print(f"    {symbol:8s}: {weight:7.2%}")

target_weights = weights.weights

if not target_weights:
    print("\n  No positions to trade. Exiting.")
    sys.exit(0)

# Save signal to file
output_dir = Path(config.paths.weights_dir)
output_dir.mkdir(parents=True, exist_ok=True)
signal_file = output_dir / f"signal_{as_of_date}.json"
signal_data = {
    "date": as_of_date,
    "regime": weights.regime_state,
    "invested": weights.get_invested_weight(),
    "cash": weights.cash_weight,
    "weights": target_weights,
}
with open(signal_file, "w") as f:
    json.dump(signal_data, f, indent=2, default=str)
print(f"\n  Signal saved to: {signal_file}")

# ── 3b: Execute on Alpaca paper trading ──────────────
print(f"\n--- Step 3b: {'[DRY RUN] ' if dry_run else ''}Executing on Alpaca Paper Trading ---\n")

from src.trading.alpaca_manager import AlpacaManager, create_alpaca_account_from_env

if account_name:
    account = create_alpaca_account_from_env(account_name)
else:
    account = create_alpaca_account_from_env()

if not account.is_paper:
    print("  ERROR: Account is NOT paper trading!")
    print(f"  Base URL: {account.base_url}")
    print("  Refusing to execute on live account via deploy.sh")
    print("  Use the Alpaca dashboard or a dedicated live trading script instead")
    sys.exit(1)

manager = AlpacaManager([account])

# Show account status
account_info = manager.get_account_info()
print(f"  Account:        {account.name} (paper)")
print(f"  Equity:         ${float(account_info.get('equity', 0)):,.2f}")
print(f"  Cash:           ${float(account_info.get('cash', 0)):,.2f}")
print(f"  Portfolio Value: ${float(account_info.get('portfolio_value', 0)):,.2f}")

positions = manager.get_positions()
print(f"  Current Positions: {len(positions)}")
if positions:
    for pos in positions:
        sym = pos.get("symbol", "?")
        qty = pos.get("qty", 0)
        mv  = float(pos.get("market_value", 0))
        print(f"    {sym:8s}: {qty} shares (${mv:,.2f})")

print(f"\n  Target weights: {json.dumps(target_weights, indent=4)}")

# Execute rebalance
result = manager.execute_portfolio_rebalance(
    target_weights=target_weights,
    account_name=account.name,
    dry_run=dry_run,
    market_closed_action='skip'
)

print(f"\n  {'[DRY RUN] ' if dry_run else ''}Rebalance result:")
if dry_run or result.get("orders_plan"):
    orders_plan = result.get("orders_plan", result)
    print(f"    Plan: {json.dumps(orders_plan, indent=4, default=str)}")
else:
    n_placed = result.get("orders_placed", 0)
    orders = result.get("orders", [])
    print(f"    Orders placed: {n_placed}")
    for o in orders:
        if isinstance(o, dict):
            side = o.get("side", "?")
            sym = o.get("symbol", "?")
            qty = o.get("qty", o.get("quantity", "?"))
        else:
            side = getattr(o, "side", "?")
            sym = getattr(o, "symbol", "?")
            qty = getattr(o, "qty", getattr(o, "quantity", "?"))
        print(f"      {str(side).upper():5s} {sym:8s} x {qty}")

# Save execution log
exec_file = output_dir / f"execution_{as_of_date}.json"
exec_data = {
    "date": as_of_date,
    "dry_run": dry_run,
    "account": account.name,
    "signal": signal_data,
    "result": result,
}
with open(exec_file, "w") as f:
    json.dump(exec_data, f, indent=2, default=str)
print(f"\n  Execution log saved to: {exec_file}")
PYEOF

else
    echo "Error: unknown mode '$MODE' (use backtest | single | paper)"
    exit 1
fi

echo ""
echo "=============================================="
echo " Done!"
echo "=============================================="
