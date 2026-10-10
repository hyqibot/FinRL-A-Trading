# -*- coding: utf-8 -*-
"""
跑 FinRL-Trading 策略 -> 导出 a_share_weights.csv
================================================
策略代码一行不改：构建 A股数据集后调用策略的 generate_weights()，
把 result.weights 落盘。支持 ML 基本面选股（主）与价格类（等权/最小方差）。

用法:
  python run_ashare_weights.py --strategy ml --universe hs300.txt --out a_share_weights.csv
  python run_ashare_weights.py --strategy ml --codes 600519 000001 600036
  python run_ashare_weights.py --strategy equal --codes 600519 000001
环境:
  通达信根目录：--tdx-dir / 环境变量 TDX_DIR / user_config.json 的 TDX_CONFIG.tdx_dir
  （键名 TDX_CONFIG.tdx_dir；文档默认 D:\\dgzq）
"""
import os
import sys
import argparse
import logging

import pandas as pd

# 让 `src` 包可导入（从仓库根运行）
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.ashare_data_adapter import build_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ashare_weights")


def load_universe(args):
    if args.codes:
        return [str(c).zfill(6) for c in args.codes]
    if args.universe:
        codes = []
        with open(args.universe, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split(",")[0].strip()
                if line and line[0].isdigit():
                    codes.append(line.zfill(6))
        return codes
    # 默认一个小样例
    return ["600519", "000001", "600036", "000002", "601318", "600000", "000858", "002415"]


def build_strategy(name):
    name = name.lower()
    if name in ("ml", "ml_select", "mlstock"):
        from src.strategies.ml_strategy import MLStockSelectionStrategy
        from src.strategies.base_strategy import StrategyConfig
        cfg = StrategyConfig(name="MLStockSelector")
        return MLStockSelectionStrategy(cfg), "ml"
    if name in ("equal", "equal_weight", "ew"):
        from src.strategies.base_strategy import EqualWeightStrategy, StrategyConfig
        return EqualWeightStrategy(StrategyConfig(name="EqualWeight")), "price"
    if name in ("minvar", "min_variance", "mv"):
        # 用 ML 选股框架的最小方差权重，或等权策略+min_variance；这里用 ML 以复用权重器
        from src.strategies.ml_strategy import MLStockSelectionStrategy
        from src.strategies.base_strategy import StrategyConfig
        return MLStockSelectionStrategy(StrategyConfig(name="MLMinVar")), "ml"
    raise SystemExit(f"未知策略: {name}（可选 ml / equal / minvar）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="ml")
    ap.add_argument("--codes", nargs="*", default=None)
    ap.add_argument("--universe", default=None, help="每行一个代码（或 代码,名称）的txt/csv")
    ap.add_argument("--out", default="a_share_weights.csv")
    ap.add_argument("--start-year", type=int, default=2018)
    ap.add_argument("--prediction-mode", default="single", choices=["single", "rolling"])
    ap.add_argument("--weight-method", default="min_variance", choices=["equal", "min_variance"])
    ap.add_argument("--no-industry", action="store_true", help="跳过行业映射（更快；base ML 不需要）")
    ap.add_argument("--cache", default=None, help="数据集缓存前缀（命中则不再拉网络）")
    ap.add_argument("--min-weight", type=float, default=1e-4, help="滤掉低于该权重的噪声仓（默认0.01%），再重新归一")
    ap.add_argument("--tdx-dir", default=None, help="通达信安装根目录（含 vipdoc）；默认读 TDX_DIR / user_config.json TDX_CONFIG.tdx_dir")
    args = ap.parse_args()

    codes = load_universe(args)
    logger.info(f"股票池 {len(codes)} 只: {codes[:10]}{' ...' if len(codes) > 10 else ''}")

    strategy, kind = build_strategy(args.strategy)
    # 仓库精简版 BaseStrategy 未建 logger，而 ml_strategy 用到 self.logger；外部注入，不改策略代码
    if not hasattr(strategy, "logger") or strategy.logger is None:
        strategy.logger = logging.getLogger(strategy.__class__.__name__)
    ds = build_dataset(codes, start_year=args.start_year, with_industry=not args.no_industry, cache_path=args.cache, tdx_dir=args.tdx_dir)

    if kind == "ml":
        if ds["fundamentals"].empty:
            raise SystemExit("基本面为空，无法跑 ML 策略")
        data = {"fundamentals": ds["fundamentals"], "prices": ds["prices"]}
        result = strategy.generate_weights(
            data,
            prediction_mode=args.prediction_mode,
            weight_method=args.weight_method,
        )
    else:
        # 价格类：策略吃 {tic: close_series} 或长表，视具体策略而定；等权只需候选池
        data = {"prices": ds["prices"], "fundamentals": ds["fundamentals"]}
        result = strategy.generate_weights(data)

    weights = result.weights.copy()
    # 统一列：tic 为6位码（gvkey==tic），weight
    if "tic" not in weights.columns and "gvkey" in weights.columns:
        weights["tic"] = weights["gvkey"].astype(str).str.zfill(6)
    if "gvkey" in weights.columns:
        weights["gvkey"] = weights["gvkey"].astype(str).str.zfill(6)
    if "tic" in weights.columns:
        weights["tic"] = weights["tic"].astype(str).str.zfill(6)
    if "weight" in weights.columns:
        weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce").fillna(0.0)
        # 去重：同一标的只保留权重最大的一行（min_variance 偶发重复行）
        if "tic" in weights.columns:
            weights = weights.sort_values("weight", ascending=False).drop_duplicates(subset="tic")
        # 滤掉接近0的噪声仓（min_variance 会给出 ~1e-19 的数值零），再重新归一
        n_before = len(weights)
        weights = weights[weights["weight"] > args.min_weight].copy()
        total = weights["weight"].sum()
        if total > 0:
            weights["weight"] = weights["weight"] / total
        weights = weights.sort_values("weight", ascending=False).reset_index(drop=True)
        logger.info(f"去重+滤微仓: {n_before} -> {len(weights)} 只（阈值 {args.min_weight}），权重和 {weights['weight'].sum():.4f}")
    weights.to_csv(args.out, index=False, encoding="utf-8-sig")
    logger.info(f"已导出 {len(weights)} 只 -> {args.out}")
    print(weights.head(20).to_string())


if __name__ == "__main__":
    main()
