# -*- coding: utf-8 -*-
"""CLI long-run entry for A-share FinRL rolling DRL (same path as Streamlit UI).

Example (from project root, venv activated):
  python -m src.ashare_finrl_cli --universe hs300.txt --cache data/cache_hs300 --start-year 2018 --max-windows 2 --max-stocks 13 --algos a2c,ppo --a2c-timesteps 5000 --ppo-timesteps 8000 --resume

Full resume of a previous run (default output artifacts/ashare_drl):
  python -m src.ashare_finrl_cli --resume --algos a2c,ppo,ddpg

Do NOT start a 99-window train from the UI; use this CLI and watch live_progress.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_codes(universe_path: str, manual: str = "") -> List[str]:
    if manual and manual.strip():
        return [
            c.strip().zfill(6)
            for c in manual.replace("，", ",").replace(" ", ",").split(",")
            if c.strip()
        ]
    codes: List[str] = []
    with open(universe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")[0].strip()
            if line and line[0].isdigit():
                codes.append(line.zfill(6))
    return codes


def _cache_path_for_year(cache: str, start_year: int, need_ind: bool = False) -> str:
    from src.config.ashare_tdx import DEFAULT_START_YEAR

    cache_path = str(cache) if cache else cache
    if cache_path and int(start_year) != int(DEFAULT_START_YEAR):
        suffix = "_y%d" % int(start_year)
        if not str(cache_path).endswith(suffix):
            cache_path = str(cache_path) + suffix
    if need_ind and cache_path and not str(cache_path).endswith("_ind"):
        cache_path = str(cache_path) + "_ind"
    return cache_path


def _parse_algos(s: str) -> List[str]:
    parts = [p.strip().lower() for p in (s or "").replace(";", ",").split(",") if p.strip()]
    return parts or ["a2c", "ppo", "ddpg"]


def _build_timesteps(args: argparse.Namespace, algos: List[str]) -> Optional[Dict[str, int]]:
    mapping: Dict[str, int] = {}
    for algo in algos:
        attr = "%s_timesteps" % algo.replace("-", "_")
        val = getattr(args, attr, None)
        if val is not None:
            mapping[algo] = int(val)
    if args.total_timesteps is not None:
        return {a: int(args.total_timesteps) for a in algos}
    return mapping or None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="A-share FinRL rolling DRL long-run (generate_finrl_rolling_weights)"
    )
    parser.add_argument(
        "--universe",
        default=str(REPO_ROOT / "hs300.txt"),
        help="Universe file (one code per line). Default: hs300.txt",
    )
    parser.add_argument(
        "--manual-codes",
        default="",
        help="Comma-separated codes; if set, overrides --universe",
    )
    parser.add_argument(
        "--cache",
        default=str(REPO_ROOT / "data" / "cache_hs300"),
        help="Dataset cache prefix (same as UI)",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument(
        "--tdx-dir",
        default=None,
        help="Tongdaxin root (optional; uses config default if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "ashare_drl"),
        help="Artifacts root (models/progress/live_progress)",
    )
    parser.add_argument("--max-windows", type=int, default=None, help="0/None = all windows")
    parser.add_argument("--max-stocks", type=int, default=None, help="0/None = no cap")
    parser.add_argument(
        "--algos",
        default="a2c,ppo,ddpg",
        help="Comma-separated: a2c,ppo,ddpg,td3,sac",
    )
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override all algos")
    parser.add_argument("--a2c-timesteps", type=int, default=None)
    parser.add_argument("--ppo-timesteps", type=int, default=None)
    parser.add_argument("--ddpg-timesteps", type=int, default=None)
    parser.add_argument("--td3-timesteps", type=int, default=None)
    parser.add_argument("--sac-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from progress.json (default on)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore progress.json and start fresh")
    parser.add_argument("--rebalance-freq", default="Q")
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--testing-window-days", type=int, default=365)
    parser.add_argument("--max-rolling-window-days", type=int, default=1095)
    args = parser.parse_args(argv)

    resume = False if args.no_resume else True
    algos = _parse_algos(args.algos)
    max_windows = args.max_windows if args.max_windows and args.max_windows > 0 else None
    max_stocks = args.max_stocks if args.max_stocks and args.max_stocks > 0 else None
    timesteps = _build_timesteps(args, algos)

    from src.config.ashare_tdx import get_tdx_dir
    from src.data.ashare_data_adapter import build_dataset
    from src.strategies.ashare_finrl_drl import generate_finrl_rolling_weights

    tdx_dir = args.tdx_dir or get_tdx_dir()
    codes = _load_codes(args.universe, args.manual_codes)
    if not codes:
        print("ERROR: empty universe", file=sys.stderr)
        return 2

    cache_path = _cache_path_for_year(args.cache, int(args.start_year), need_ind=False)
    print(
        "CLI FinRL rolling: codes=%d start_year=%d cache=%s algos=%s max_windows=%s max_stocks=%s resume=%s out=%s"
        % (
            len(codes),
            args.start_year,
            cache_path,
            algos,
            max_windows,
            max_stocks,
            resume,
            args.output_dir,
        ),
        flush=True,
    )

    def progress(msg: str) -> None:
        print(msg, flush=True)

    ds = build_dataset(
        codes,
        start_year=int(args.start_year),
        with_industry=False,
        cache_path=cache_path,
        tdx_dir=tdx_dir,
    )
    prices = ds.get("prices")
    if prices is None or getattr(prices, "empty", True):
        print("ERROR: empty prices panel", file=sys.stderr)
        return 3

    kwargs = {
        "algos": algos,
        "testing_window_days": int(args.testing_window_days),
        "max_rolling_window_days": int(args.max_rolling_window_days),
        "lookback": int(args.lookback),
        "rebalance_freq": args.rebalance_freq,
        "seed": int(args.seed),
        "resume": resume,
        "output_dir": args.output_dir,
        "max_stocks": max_stocks,
        "max_windows": max_windows,
        "progress": progress,
    }
    if timesteps:
        kwargs["timesteps_by_algo"] = timesteps

    weights = generate_finrl_rolling_weights(prices, **kwargs)
    print(
        "DONE latest weights n=%d sum=%.6f -> %s"
        % (len(weights), float(weights["weight"].sum()), args.output_dir),
        flush=True,
    )
    print("live_progress: %s" % os.path.join(args.output_dir, "live_progress.txt"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())