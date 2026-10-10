"""
YAML Auto-Write with GICS Preclassification

Load SP500 and preclassify each symbol into one of:
   - growth_tech
   - real_assets
   - defensive
Select symbols per category (target N, default 7):
   - top_quantile + predicted_return > 0
   - If not enough, relax to top-N by predicted_return within category
Write selected symbols into AdaptiveRotationConf asset_groups section only.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re
import pandas as pd
import yaml
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.data.data_fetcher import fetch_fundamental_data, fetch_sp500_tickers
#from src.strategies.base_strategy import StrategyConfig
from src.strategies.ml_strategy import MLStockSelectionStrategy


BUCKET_ORDER = ["growth_tech", "real_assets", "defensive"]
BUCKET_TO_GROUP = {
    "growth_tech": "group_a_growth_tech",
    "cyclical": "group_b_cyclical",
    "real_assets": "group_c_real_assets",
    "defensive": "group_d_defensive",
}

SECTOR_TO_BUCKET = {
    # Growth tech — 高增长、高估值
    "information technology": "growth_tech",
    "technology": "growth_tech",
    "communication services": "growth_tech",

    # Cyclical — 跟经济周期走
    "consumer discretionary": "cyclical",
    "consumer cyclical": "cyclical",
    "financials": "cyclical",
    "financial services": "cyclical",
    "industrials": "cyclical",

    # Real assets — 跟通胀/大宗商品走
    "energy": "real_assets",
    "materials": "real_assets",
    "basic materials": "real_assets",
    "real estate": "real_assets",

    # Defensive — 抗衰退、低波动
    "health care": "defensive",
    "healthcare": "defensive",
    "consumer staples": "defensive",
    "consumer defensive": "defensive",
    "utilities": "defensive",
}


@dataclass
class GroupSelectionSummary:
    bucket: str
    candidates: int
    selected: int
    selection_mode: str
    strict_threshold: float

@dataclass
class LocalStrategyConfig:
    """Minimal config shim to avoid hard dependency on base_strategy module."""

    name: str = "BaseStrategy"

def _norm_sector(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def map_sector_to_bucket(sector: object) -> str:
    key = _norm_sector(sector)
    if key in SECTOR_TO_BUCKET:
        return SECTOR_TO_BUCKET[key]

    # Conservative fallback: classify unknown sectors as defensive
    return "defensive"


def preclassify_universe(tickers_df: pd.DataFrame) -> pd.DataFrame:
    df = tickers_df.copy()

    if "tickers" not in df.columns:
        if "symbol" in df.columns:
            df = df.rename(columns={"symbol": "tickers"})
        else:
            raise ValueError("SP500 universe is missing 'tickers' column")

    if "sectors" not in df.columns:
        df["sectors"] = None

    if "dateFirstAdded" not in df.columns:
        df["dateFirstAdded"] = None

    df["gics_bucket"] = df["sectors"].apply(map_sector_to_bucket)
    return df[["tickers", "sectors", "dateFirstAdded", "gics_bucket"]]


def ensure_fundamental_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "datadate" not in out.columns and "date" in out.columns:
        out["datadate"] = out["date"]

    if "gvkey" not in out.columns and "tic" in out.columns:
        out["gvkey"] = out["tic"]

    # y_return is required for training labels, but rows with NaN y_return may
    # still be valid trade-time candidates and should not be dropped here.
    if "y_return" not in out.columns:
        raise ValueError("fundamentals are missing required label column: y_return")

    if "gvkey" not in out.columns or "datadate" not in out.columns:
        raise ValueError("fundamentals are missing required columns: gvkey/datadate")

    return out


def select_effective_trade_cutoff(
    fundamentals: pd.DataFrame,
    as_of_date: str | None = None,
    report_lag_quarters: int = 2,
) -> pd.Timestamp:
    """Pick trade cutoff by calendar quarter lag (default: use quarter T-2)."""
    if len(fundamentals) == 0:
        raise ValueError("empty fundamentals")

    d = fundamentals.copy()
    d["datadate"] = pd.to_datetime(d["datadate"], errors="coerce")
    available_dates = pd.DatetimeIndex(sorted(d["datadate"].dropna().unique()))
    if len(available_dates) == 0:
        raise ValueError("no valid datadate in fundamentals")

    lag = max(int(report_lag_quarters), 0)
    if as_of_date:
        as_of = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(as_of):
            raise ValueError(f"invalid as_of_date: {as_of_date}")
    else:
        as_of = pd.Timestamp.today().normalize()

    target_period = as_of.to_period("Q") - lag
    target_date = pd.Timestamp(target_period.end_time).normalize()

    candidates = available_dates[available_dates <= target_date]
    if len(candidates) == 0:
        return available_dates.min()
    return candidates.max()


def load_sp500_universe_online_first(preferred_source: str) -> pd.DataFrame:
    """Fetch SP500 online first, then fallback to local csv when online is unavailable."""
    local_path = Path(project_root) / "data" / "sp500_tickers.csv"
    online_cache_path = Path(project_root) / "data" / "sp500_tickers_online_latest.csv"
    secondary_local_path = Path(project_root) / "data" / "gics_ticker_mapping_from_fetch.csv"

    def _normalize_local(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "tickers" not in out.columns and "symbol" in out.columns:
            out = out.rename(columns={"symbol": "tickers"})
        if "sectors" not in out.columns:
            out["sectors"] = None
        if "dateFirstAdded" not in out.columns:
            out["dateFirstAdded"] = None
        out = out.dropna(subset=["tickers"]).copy()
        return out[["tickers", "sectors", "dateFirstAdded"]]

    try:
        online_df = fetch_sp500_tickers(
            output_path=str(online_cache_path),
            preferred_source=preferred_source,
        )
        if online_df is not None and len(online_df) > 0:
            _normalize_local(online_df).to_csv(local_path, index=False)
            return online_df
        print("warning: online SP500 universe is empty, fallback to local csv")
    except Exception as exc:
        print(f"warning: online SP500 fetch failed, fallback to local csv: {exc}")

    if local_path.exists():
        local_df = _normalize_local(pd.read_csv(local_path))
        if len(local_df) > 0:
            return local_df

    if secondary_local_path.exists():
        secondary_df = _normalize_local(pd.read_csv(secondary_local_path))
        if len(secondary_df) > 0:
            secondary_df.to_csv(local_path, index=False)
            print("warning: primary local SP500 csv is empty, using secondary local mapping fallback")
            return secondary_df

    raise ValueError("SP500 universe unavailable from both online source and local csv")


def select_min_per_bucket(
    pred_df: pd.DataFrame,
    min_per_group: int,
    top_quantile: float,
) -> Tuple[pd.DataFrame, List[GroupSelectionSummary]]:
    selections = []
    summaries: List[GroupSelectionSummary] = []

    for bucket in BUCKET_ORDER:
        g = pred_df[pred_df["gics_bucket"] == bucket].copy()
        g = g.sort_values("predicted_return", ascending=False)

        if g.empty:
            summaries.append(
                GroupSelectionSummary(
                    bucket=bucket,
                    candidates=0,
                    selected=0,
                    selection_mode="no_candidates",
                    strict_threshold=float("nan"),
                )
            )
            continue

        threshold = float(g["predicted_return"].quantile(top_quantile))
        strict = g[(g["predicted_return"] >= threshold) & (g["predicted_return"] > 0)].copy()

        if len(strict) >= min_per_group:
            selected = strict.sort_values("predicted_return", ascending=False).head(min_per_group).copy()
            mode = "strict_quantile"
        else:
            selected = g.head(min_per_group).copy()
            mode = "relaxed_topk"

        selected["gics_bucket"] = bucket
        selected["selection_mode"] = mode
        selected["strict_threshold"] = threshold
        selected["bucket_rank"] = range(1, len(selected) + 1)

        summaries.append(
            GroupSelectionSummary(
                bucket=bucket,
                candidates=len(g),
                selected=len(selected),
                selection_mode=mode,
                strict_threshold=threshold,
            )
        )

        selections.append(selected)

    if not selections:
        return pd.DataFrame(columns=["gvkey", "predicted_return", "gics_bucket"]), summaries

    return pd.concat(selections, ignore_index=True), summaries


def replace_asset_groups_section(
    yaml_path: Path,
    selected_symbols: Dict[str, List[str]],
) -> None:
    original_text = yaml_path.read_text(encoding="utf-8")

    # Parse max_assets from current config to avoid changing other behavior
    parsed = yaml.safe_load(original_text)
    current_groups = (parsed or {}).get("asset_groups", {})

    def _max_assets(group_name: str) -> int:
        group_cfg = current_groups.get(group_name, {}) if isinstance(current_groups, dict) else {}
        value = group_cfg.get("max_assets", 2) if isinstance(group_cfg, dict) else 2
        try:
            return int(value)
        except Exception:
            return 2

    new_block_lines = [
        "asset_groups:",
        f"  group_a_growth_tech:",
        f"    max_assets: {_max_assets('group_a_growth_tech')}",
        "    symbols:",
    ]
    for s in selected_symbols.get("growth_tech", []):
        new_block_lines.append(f"      - {s}")

    new_block_lines.extend([
        "",
        "  group_b_real_assets:",
        f"    max_assets: {_max_assets('group_b_real_assets')}",
        "    symbols:",
    ])
    for s in selected_symbols.get("real_assets", []):
        new_block_lines.append(f"      - {s}")

    new_block_lines.extend([
        "",
        "  group_c_defensive:",
        f"    max_assets: {_max_assets('group_c_defensive')}",
        "    symbols:",
    ])
    for s in selected_symbols.get("defensive", []):
        new_block_lines.append(f"      - {s}")

    new_block = "\n".join(new_block_lines) + "\n"

   # Replace only the asset_groups block and preserve everything else verbatim.
    pattern_comment_boundary = re.compile(
        r"asset_groups:\r?\n.*?(?=\r?\n# -----------------------\r?\n# Market Regime \(Slow\)\r?\n# -----------------------)",
        re.DOTALL,
    )
    pattern_key_boundary = re.compile(
        r"asset_groups:\r?\n.*?(?=\r?\nmarket_regime:)",
        re.DOTALL,
    )

    if pattern_comment_boundary.search(original_text):
        updated_text = pattern_comment_boundary.sub(new_block.rstrip("\n"), original_text, count=1)
    elif pattern_key_boundary.search(original_text):
        updated_text = pattern_key_boundary.sub(new_block.rstrip("\n"), original_text, count=1)
    else:
        raise ValueError("Could not locate asset_groups section boundaries in YAML")

    yaml_path.write_text(updated_text, encoding="utf-8")


def main() -> None:
    default_data_dir = Path(project_root) / "data"
    default_data_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="ML + GICS preclassification with YAML asset_groups auto-write")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-04-01")
    parser.add_argument("--top-quantile", type=float, default=0.75)
    parser.add_argument("--test-quarters", type=int, default=4)
    parser.add_argument("--min-per-group", type=int, default=7)
    parser.add_argument("--limit", type=int, default=10000, help="Universe cap; use large value for full SP500")
    parser.add_argument(
        "--yaml-path",
        default=str(Path(project_root) / "src" / "strategies" / "AdaptiveRotationConf_v1.2.1.yaml"),
        help="Target YAML file to update asset_groups",
    )
    parser.add_argument(
        "--output-csv",
        default=str(default_data_dir / "ml_weights_fmp_sp500_gics.csv"),
        help="Output selected symbols csv",
    )
    parser.add_argument("--preferred-source", default="FMP")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Evaluation date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--report-lag-quarters",
        type=int,
        default=2,
        help="Use quarter T-lag as trade cutoff (default: 2 => use previous previous quarter)",
    )
    args = parser.parse_args()

    tickers = load_sp500_universe_online_first(preferred_source=args.preferred_source)
    tickers = preclassify_universe(tickers)

    if args.limit > 0:
        tickers = tickers.head(args.limit).copy()

    if len(tickers) == 0:
        raise ValueError("No tickers available after universe loading and limit filtering")

    fundamentals = fetch_fundamental_data(
        tickers,
        args.start_date,
        args.end_date,
        preferred_source=args.preferred_source,
    )
    fundamentals = ensure_fundamental_schema(fundamentals)

    cutoff_date = select_effective_trade_cutoff(
        fundamentals,
        as_of_date=args.as_of_date,
        report_lag_quarters=args.report_lag_quarters,
    )
    latest_date = pd.to_datetime(fundamentals["datadate"]).max()
    if pd.notna(latest_date) and cutoff_date < latest_date:
        as_of_text = args.as_of_date if args.as_of_date else pd.Timestamp.today().strftime("%Y-%m-%d")
        print(
            f"info: as_of={as_of_text}, report_lag_quarters={args.report_lag_quarters}, "
            f"effective trade cutoff={cutoff_date.date()} (latest available={latest_date.date()})"
        )
    fundamentals = fundamentals[pd.to_datetime(fundamentals["datadate"]) <= cutoff_date].copy()

    strategy = MLStockSelectionStrategy(LocalStrategyConfig(name="ML GICS Preclassified Selection"))

    # rolling predictions, keep the latest prediction for each ticker.
    unique_quarters = sorted(pd.to_datetime(fundamentals["datadate"]).dropna().unique())
    max_train_quarters = len(unique_quarters) - args.test_quarters - 1
    train_quarters = min(16, max_train_quarters)

    if train_quarters >= 1:
        try:
            pred_all = strategy._rolling_train_all_date(
                fundamentals=fundamentals,
                train_quarters=train_quarters,
                test_quarters=args.test_quarters,
            )
        except Exception:
            pred_all = pd.DataFrame()
    else:
        pred_all = pd.DataFrame()

    if pred_all is None or len(pred_all) == 0:
        try:
            single_pred, single_meta = strategy._rolling_train_single_date(
                fundamentals=fundamentals,
                test_quarters=args.test_quarters,
            )
            single_pred = single_pred.copy()
            single_pred["date"] = pd.to_datetime(single_meta.get("trade_date"))
            pred_all = single_pred
        except Exception as exc:
            print(f"single-date prediction fallback failed: {exc}")
            pred_all = pd.DataFrame()

    if pred_all is None or len(pred_all) == 0:
        raise ValueError("No predictions generated after rolling and single fallback")

    pred_all = pred_all.copy()
    pred_all["gvkey"] = pred_all["gvkey"].astype(str)
    pred_all["date"] = pd.to_datetime(pred_all["date"]) if "date" in pred_all.columns else pd.NaT
    pred_all = pred_all.sort_values(["gvkey", "date"])
    pred_df = pred_all.groupby("gvkey", as_index=False).tail(1)[["gvkey", "predicted_return", "date"]].copy()
    trade_date = pred_df["date"].max() if "date" in pred_df.columns else None

    universe_map = tickers[["tickers", "sectors", "gics_bucket"]].copy()
    universe_map["tickers"] = universe_map["tickers"].astype(str)

    merged = pred_df.merge(universe_map, left_on="gvkey", right_on="tickers", how="left")

    # Fallback classification from fundamentals.gsector if ticker-sector merge misses
    if merged["gics_bucket"].isna().any() and "gsector" in fundamentals.columns:
        sec = fundamentals[["gvkey", "gsector"]].copy()
        sec["gvkey"] = sec["gvkey"].astype(str)
        sec = sec.dropna(subset=["gsector"]).drop_duplicates(subset=["gvkey"], keep="last")
        merged = merged.merge(sec, on="gvkey", how="left", suffixes=("", "_fund"))
        merged["gics_bucket"] = merged["gics_bucket"].fillna(merged["gsector"].apply(map_sector_to_bucket))

    merged = merged.dropna(subset=["gics_bucket"]).copy()

    selected, summaries = select_min_per_bucket(
        merged[["gvkey", "predicted_return", "gics_bucket"]].copy(),
        min_per_group=args.min_per_group,
        top_quantile=args.top_quantile,
    )

    selected = selected.sort_values(["gics_bucket", "predicted_return"], ascending=[True, False])

    out_path = Path(args.output_csv)
    if not out_path.is_absolute():
        out_path = Path(project_root) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    symbols_by_bucket = {
        bucket: selected[selected["gics_bucket"] == bucket]["gvkey"].astype(str).tolist()
        for bucket in BUCKET_ORDER
    }

    yaml_path = Path(args.yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = Path(project_root) / yaml_path

    replace_asset_groups_section(yaml_path, symbols_by_bucket)

    print("trade_date:", trade_date)
    print("pred_rows:", len(pred_df))
    for s in summaries:
        print(
            f"bucket={s.bucket}, candidates={s.candidates}, selected={s.selected}, "
            f"mode={s.selection_mode}, strict_threshold={s.strict_threshold:.6f}"
        )
        bucket_symbols = symbols_by_bucket.get(s.bucket, [])
        print(f"  symbols={', '.join(bucket_symbols)}")
    print("output_csv:", out_path)
    print("yaml_updated:", yaml_path)


if __name__ == "__main__":
    main()
