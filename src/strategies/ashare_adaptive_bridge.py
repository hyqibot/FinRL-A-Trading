# -*- coding: utf-8 -*-
"""把 A 股价格喂给原项目 AdaptiveRotationEngine。

引擎硬编码需要 ^GSPC / ^VIX / QQQ 日线 CSV（文件名不变，内容换成 A 股代理序列）：
  ^GSPC  <- 股票池等权指数
  QQQ    <- 成长组等权指数
  ^VIX   <- 由 ^GSPC 20 日实现波动率缩放成类 VIX

资产组：按行业关键词分成 4 组。不改引擎源码。
写出 YAML 用内置 dumper；加载仍走原 config_loader（需 PyYAML，已列入 requirements）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ashare_adaptive_bridge")

BUCKET_KEYWORDS = {
    "group_a_growth_tech": ("电子", "计算机", "通信", "传媒", "半导体", "软件", "互联网", "军工", "国防"),
    "group_b_cyclical": ("银行", "非银", "证券", "保险", "房地产", "建筑", "机械", "汽车", "化工", "有色", "钢铁"),
    "group_c_real_assets": ("石油", "煤炭", "燃气", "采掘", "黄金", "大宗", "基础化工", "农林"),
    "group_d_defensive": ("食品", "饮料", "医药", "生物", "公用", "电力", "交通", "家电", "商贸", "社会服务"),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dump_yaml(obj, indent=0) -> str:
    sp = "  " * indent
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append("%s%s:" % (sp, k))
                chunk = _dump_yaml(v, indent + 1)
                if chunk:
                    lines.append(chunk)
            elif v is None:
                lines.append("%s%s: null" % (sp, k))
            elif isinstance(v, bool):
                lines.append("%s%s: %s" % (sp, k, "true" if v else "false"))
            elif isinstance(v, (int, float)):
                lines.append("%s%s: %s" % (sp, k, v))
            else:
                s = str(v).replace("\\", "/")
                if any(ch in s for ch in ":#{}[]&*?|>!%@`,"):
                    s = '"' + s.replace('"', "'") + '"'
                lines.append("%s%s: %s" % (sp, k, s))
        return "\n".join(lines)
    if isinstance(obj, list):
        lines = []
        for item in obj:
            if isinstance(item, (dict, list)):
                lines.append("%s-" % sp)
                chunk = _dump_yaml(item, indent + 1)
                if chunk:
                    lines.append(chunk)
            else:
                lines.append("%s- %s" % (sp, item))
        return "\n".join(lines)
    return "%s%s" % (sp, obj)


def _ohlcv_from_series(close: pd.Series) -> pd.DataFrame:
    close = close.dropna().astype(float)
    return pd.DataFrame({
        "date": close.index,
        "open": close.values,
        "high": close.values,
        "low": close.values,
        "close": close.values,
        "volume": np.full(len(close), 1e6),
    })


def _write_daily_csv(symbol: str, ohlcv: pd.DataFrame, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    out = ohlcv.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(data_dir / ("%s_daily.csv" % symbol), index=False)


def _panel_close(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.zfill(6)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["tic", "date"]).drop_duplicates(["tic", "date"], keep="last")
    return df.pivot(index="date", columns="tic", values="close").sort_index().ffill()


def _assign_buckets(codes: List[str], industry_map: Dict[str, str]) -> Dict[str, List[str]]:
    buckets = {k: [] for k in BUCKET_KEYWORDS}
    unmatched = []
    for code in codes:
        ind = str(industry_map.get(code) or industry_map.get(code.zfill(6)) or "")
        placed = False
        for bucket, keys in BUCKET_KEYWORDS.items():
            if any(k in ind for k in keys):
                buckets[bucket].append(code.zfill(6))
                placed = True
                break
        if not placed:
            unmatched.append(code.zfill(6))
    for i, code in enumerate(unmatched):
        buckets[list(BUCKET_KEYWORDS.keys())[i % 4]].append(code)
    return buckets


def _pick_liquid(wide: pd.DataFrame, codes: List[str], n: int) -> List[str]:
    avail = [c for c in codes if c in wide.columns]
    if not avail:
        return []
    sub = wide[avail].iloc[-60:] if len(wide) >= 60 else wide[avail]
    score = sub.pct_change().abs().mean() * sub.mean()
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return list(score.sort_values(ascending=False).head(n).index)


def build_ashare_adaptive_config(
    groups: Dict[str, List[str]],
    data_root: Path,
    output_root: Path,
    max_assets: int = 3,
) -> Dict[str, Any]:
    asset_groups = {
        name: {"max_assets": int(max_assets), "symbols": list(syms)}
        for name, syms in groups.items()
    }
    return {
        "strategy": {
            "name": "adaptive_multi_asset_rotation_ashare",
            "version": "ashare-1.0",
            "base_frequency": "daily",
            "rebalance_frequency": "weekly",
        },
        "paths": {
            "data_root": str(data_root).replace("\\", "/"),
            "output_root": str(output_root).replace("\\", "/"),
            "state_dir": str(output_root / "state").replace("\\", "/"),
            "audit_dir": str(output_root / "audit").replace("\\", "/"),
            "weights_dir": str(output_root / "weights").replace("\\", "/"),
        },
        "dates": {"start_date": "2018-01-01", "end_date": None},
        "history": {"minimum_history_weeks": 26},
        "benchmark": {"excess_return_benchmark": "QQQ"},
        "asset_groups": asset_groups,
        "market_regime": {
            "slow_regime": {
                "trend_ma_weeks": 26,
                "drawdown_weeks": 13,
                "drawdown_threshold": 0.10,
                "volatility": {"vix_lookback_years": 3, "vix_z_threshold": 3.0},
                "persistence_weeks": 2,
                "mapping": {
                    "risk_on": {"risk_score": 0, "group_cap": 1.0, "cash_floor": 0.0},
                    "neutral": {"risk_score": 1, "group_cap": 0.7, "cash_floor": 0.1},
                    "risk_off": {"risk_score": 2, "group_cap": 0.5, "cash_floor": 0.3},
                },
            }
        },
        "fast_risk_off": {
            "price_shock": {"lookback_days": 3, "drawdown_threshold": -0.03},
            "volatility_shock": {"vix_z_threshold": 3.0, "delta_vix_z_threshold": 3.5},
            "behavior": {"group_cap": 0.3, "cash_floor": 0.5, "duration_days": 10},
            "stop_loss_multiplier": 0.7,
        },
        "group_strength": {"metric": "risk_adjusted_return", "lookback_weeks": 12, "trend_filter": True},
        "ranking": {
            "method": "zscore",
            "robust": True,
            "top_n_per_group": 3,
            "zscore_window": 12,
            "max_zscore": 20.0,
            "min_mad_threshold": 1.0e-6,
        },
        "exception": {
            "z_threshold": 2.5,
            "lookback_weeks": 4,
            "min_trigger_count": 2,
            "reentry": {"cooldown_weeks": 6, "stricter_threshold_multiplier": 1.5},
            "strong_signal": {
                "enabled": True,
                "z_threshold": 3.5,
                "return_multiplier": 1.5,
                "return_lookback_weeks": 12,
                "require_positive_return": True,
            },
        },
        "portfolio": {
            "max_active_groups": 2,
            "allow_exception": True,
            "exception_weight_multiplier": 1.5,
            "weighting": {"scheme": "equal", "residual_to_cash": True},
            "fallback": {"enabled": True, "symbols": ["^GSPC", "QQQ"], "allocation": "equal"},
        },
        "stop_loss": {
            "frequency": "daily",
            "absolute": {"enabled": True, "threshold": -0.05},
            "trailing": {"enabled": True, "threshold": -0.10},
        },
        "cooldown": {"after_stop_days": 20, "block_reentry": True},
        "audit": {"enabled": True, "save_weights": True},
        "state": {"persist": True},
    }


def prepare_adaptive_data(
    prices: pd.DataFrame,
    industry_map: Optional[Dict[str, str]] = None,
    data_dir: Optional[str] = None,
    max_per_group: int = 5,
    max_assets: int = 3,
) -> Tuple[Path, Path, Dict[str, List[str]]]:
    root = _project_root()
    data_root = Path(data_dir) if data_dir else (root / "data" / "ashare_adaptive_daily")
    output_root = root / "src" / "strategies" / "output" / "adaptive_rotation_ashare"
    data_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    wide = _panel_close(prices)
    codes = [str(c).zfill(6) for c in wide.columns]
    industry_map = industry_map or {}
    raw_groups = _assign_buckets(codes, industry_map)
    groups = {}
    for name, syms in raw_groups.items():
        picked = _pick_liquid(wide, syms, max_per_group)
        groups[name] = picked
        for sym in picked:
            _write_daily_csv(sym, _ohlcv_from_series(wide[sym].dropna()), data_root)

    ew = wide.mean(axis=1).dropna()
    _write_daily_csv("^GSPC", _ohlcv_from_series(ew), data_root)

    growth_cols = [c for c in groups.get("group_a_growth_tech", []) if c in wide.columns]
    growth = wide[growth_cols].mean(axis=1).dropna() if growth_cols else ew
    _write_daily_csv("QQQ", _ohlcv_from_series(growth), data_root)

    ret = ew.pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252) * 100.0
    vol = vol.bfill().fillna(20.0).clip(8.0, 80.0)
    _write_daily_csv("^VIX", _ohlcv_from_series(vol), data_root)

    cfg = build_ashare_adaptive_config(groups, data_root, output_root, max_assets=max_assets)
    cfg_path = root / "src" / "strategies" / "AdaptiveRotationConf_ashare.yaml"
    cfg_path.write_text(_dump_yaml(cfg) + "\n", encoding="utf-8")
    logger.info("adaptive data ready dir=%s groups=%s", data_root, {k: len(v) for k, v in groups.items()})
    return cfg_path, data_root, groups


def run_adaptive_ashare(
    prices: pd.DataFrame,
    industry_map: Optional[Dict[str, str]] = None,
    as_of_date: Optional[str] = None,
    max_per_group: int = 5,
) -> pd.DataFrame:
    cfg_path, data_root, groups = prepare_adaptive_data(
        prices, industry_map=industry_map, max_per_group=max_per_group
    )
    from src.strategies.adaptive_rotation import AdaptiveRotationEngine
    from src.strategies.adaptive_rotation.data_preprocessor import DataPreprocessor
    from src.strategies.adaptive_rotation.config_loader import load_config

    config = load_config(str(cfg_path))
    preprocessor = DataPreprocessor(config)
    preprocessor.load_and_prepare(data_dir=str(data_root))
    engine = AdaptiveRotationEngine(config=str(cfg_path), data_preprocessor=preprocessor)

    if as_of_date is None:
        ends = [df.index.max() for df in preprocessor.daily_data.values()]
        as_of_date = max(ends).strftime("%Y-%m-%d")

    raw = preprocessor.get_data_as_of(as_of_date)
    price_data = {symbol: df["close"] for symbol, df in raw.items()}
    weights_obj, audit = engine.run(price_data=price_data, as_of_date=as_of_date)

    rows = []
    for sym, w in sorted(weights_obj.weights.items(), key=lambda x: x[1], reverse=True):
        if float(w) <= 0:
            continue
        if sym in ("^GSPC", "^VIX", "QQQ"):
            continue
        code = str(sym)
        if code.isdigit():
            code = code.zfill(6)
        rows.append({"gvkey": code, "tic": code, "weight": float(w)})
    if not rows:
        flat = []
        for syms in groups.values():
            flat.extend(syms[:1])
        flat = flat[:8] or list(prices["tic"].astype(str).str.zfill(6).unique()[:8])
        ww = 1.0 / max(1, len(flat))
        rows = [{"gvkey": c, "tic": c, "weight": ww} for c in flat]
    out = pd.DataFrame(rows)
    total = out["weight"].sum()
    if total > 0:
        out["weight"] = out["weight"] / total
    out.attrs["adaptive_as_of"] = as_of_date
    out.attrs["adaptive_regime"] = getattr(weights_obj, "regime_state", None)
    return out
