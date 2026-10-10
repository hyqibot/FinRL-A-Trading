# -*- coding: utf-8 -*-
"""A股策略统一入口：把原项目策略能力接到 ashare 数据上，输出权重 DataFrame。

不改策略核心算法文件；此处只做组装 / 经典配置 / 自定义加载。
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("ashare_strategy_runner")

# README 三种用例（中文标签供 UI）
USE_CASE_PRESETS = {
    "用例1·组合配置范式": {
        "family": "equal_weight",
        "desc": "README Use Case 1：等权/均值方差/最小方差/DRL 配置范式",
        "hint_families": ["equal_weight", "min_variance", "mean_variance", "market_cap", "drl"],
    },
    "用例2·滚动选股+DRL": {
        "family": "ml_then_drl",
        "desc": "README Use Case 2：ML 滚动选股后再 DRL 定权",
        "prediction_mode": "rolling",
        "weight_method": "equal",
        "drl_algo": "ppo",
    },
    "用例3·自适应多资产轮动": {
        "family": "adaptive_rotation",
        "desc": "README Use Case 3：Adaptive Multi-Asset Rotation",
    },
    "用例4·FinRL滚动DRL": {
        "family": "finrl_rolling_drl",
        "desc": "完整复制 US rl_model+fundamental_portfolio_drl：滚动训 A2C/PPO/DDPG/TD3/SAC→验证选优→推理权重并落盘",
    },
}

STRATEGY_FAMILIES = [
    ("equal_weight", "等权"),
    ("min_variance", "最小方差（经典，全池）"),
    ("mean_variance", "均值方差 / 最大夏普（经典，全池）"),
    ("ml", "ML 选股（MLStockSelectionStrategy）"),
    ("sector_neutral_ml", "行业中性 ML（SectorNeutralMLStrategy）"),
    ("ml_bucket", "分桶 ML（A股行业映射四桶）"),
    ("ml_then_drl", "滚动 ML 选股 + DRL 定权"),
    ("drl", "短训DRL配置（ashare_drl_runner / SB3或torch夏普优化）"),
    ("finrl_rolling_drl", "FinRL滚动DRL（完整训推，可落盘）"),
    ("adaptive_rotation", "自适应多资产轮动"),
    ("custom", "自定义策略（Python 类路径）"),
    ("market_cap", "市值加权（通达信 gbbq）"),
]

# 各族一句话说明（供 UI 折叠帮助；与 STRATEGY_FAMILIES 键一一对应）
STRATEGY_FAMILY_ONELINERS = {
    "equal_weight": "股票池内每只股票等权（1/N），对应原 EqualWeightStrategy。",
    "min_variance": "用全池历史收益协方差最小化组合波动（scipy，可调回看期）。",
    "mean_variance": "全池最大夏普（均值-方差，做多约束，scipy）。",
    "ml": "ML 基本面选股（MLStockSelectionStrategy），再按所选方法定权。",
    "sector_neutral_ml": "按行业分别训练再合并（SectorNeutralMLStrategy，需行业映射）。",
    "ml_bucket": "A股无美股 GICS 四桶原数据；此处按行业中性路径落地。",
    "ml_then_drl": "先滚动 ML 选股，再用短训 DRL 定权（README 用例2）。",
    "drl": "短训 DRL 配置权重（ashare_drl_runner；默认约数千步，A2C/PPO/DDPG）。",
    "finrl_rolling_drl": "完整 FinRL 滚动训推（多窗×多算法），耗时长，可落盘续跑。",
    "adaptive_rotation": "自适应多资产轮动（A股代理指数+行业分组，依赖 PyYAML）。",
    "custom": "加载本地 .py 中 BaseStrategy 子类的 generate_weights。",
    "market_cap": "通达信 gbbq 股本×收盘价，按总市值或流通市值加权。",
}

STRATEGY_HELP_DOC_REL = "docs/策略族说明.md"


def get_strategy_help_doc_path() -> Path:
    """项目根下的策略族说明 Markdown 绝对路径。"""
    return Path(__file__).resolve().parents[2] / STRATEGY_HELP_DOC_REL


def format_strategy_family_oneliners_md() -> str:
    """生成各族一句话的 Markdown 列表（按 STRATEGY_FAMILIES 顺序）。"""
    labels = {k: v for k, v in STRATEGY_FAMILIES}
    lines = ["**策略族一览（一句话）**", ""]
    for key, _label in STRATEGY_FAMILIES:
        blurb = STRATEGY_FAMILY_ONELINERS.get(key, "")
        title = labels.get(key, key)
        lines.append(f"- **`{key}`**（{title}）：{blurb}")
    return "\n".join(lines)



def _ensure_logger(strategy) -> None:
    if not hasattr(strategy, "logger") or strategy.logger is None:
        strategy.logger = logging.getLogger(strategy.__class__.__name__)


def _normalize_weights(df: pd.DataFrame) -> pd.DataFrame:
    w = df.copy()
    if "tic" not in w.columns and "gvkey" in w.columns:
        w["tic"] = w["gvkey"].astype(str).str.zfill(6)
    if "gvkey" not in w.columns and "tic" in w.columns:
        w["gvkey"] = w["tic"].astype(str).str.zfill(6)
    if "tic" in w.columns:
        w["tic"] = w["tic"].astype(str).str.zfill(6)
    if "gvkey" in w.columns:
        w["gvkey"] = w["gvkey"].astype(str).str.zfill(6)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    total = w["weight"].sum()
    if total > 0:
        w["weight"] = w["weight"] / total
    return w.sort_values("weight", ascending=False).reset_index(drop=True)


def _panel_returns(prices: pd.DataFrame, lookback: int = 252) -> Tuple[pd.DataFrame, List[str]]:
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.zfill(6)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values(["tic", "date"]).drop_duplicates(["tic", "date"], keep="last")
    wide = df.pivot(index="date", columns="tic", values="close").sort_index()
    cover = wide.notna().mean()
    keep = cover[cover >= 0.6].index.tolist()
    wide = wide[keep].ffill().dropna(how="any")
    if lookback and len(wide) > lookback:
        wide = wide.iloc[-lookback:]
    rets = wide.pct_change().dropna(how="any")
    return rets, list(rets.columns)


def equal_weight_from_universe(fundamentals: pd.DataFrame) -> pd.DataFrame:
    from src.strategies.base_strategy import EqualWeightStrategy, StrategyConfig
    strat = EqualWeightStrategy(StrategyConfig(name="Equal_weight"))
    _ensure_logger(strat)
    result = strat.generate_weights({"fundamentals": fundamentals})
    return _normalize_weights(result.weights)


def classical_min_variance(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    rets, tickers = _panel_returns(prices, lookback=lookback)
    mu = rets.mean().values
    cov = rets.cov().values
    n = len(tickers)

    def port_var(w):
        return float(w @ cov @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else w0
    w = np.clip(w, 0, None)
    w = w / w.sum()
    return _normalize_weights(pd.DataFrame({"gvkey": tickers, "tic": tickers, "weight": w}))


def classical_mean_variance(prices: pd.DataFrame, lookback: int = 252, risk_free: float = 0.0) -> pd.DataFrame:
    """最大夏普（做多约束）。"""
    rets, tickers = _panel_returns(prices, lookback=lookback)
    mu = rets.mean().values
    cov = rets.cov().values
    n = len(tickers)

    def neg_sharpe(w):
        ret = float(w @ mu) - risk_free / 252.0
        vol = float(np.sqrt(max(w @ cov @ w, 1e-18)))
        return -ret / vol

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else w0
    w = np.clip(w, 0, None)
    w = w / w.sum()
    return _normalize_weights(pd.DataFrame({"gvkey": tickers, "tic": tickers, "weight": w}))


def run_ml_strategy(
    ds: Dict[str, pd.DataFrame],
    *,
    sector_neutral: bool = False,
    prediction_mode: str = "single",
    weight_method: str = "min_variance",
    train_quarters: int = 16,
    test_quarters: int = 4,
    top_quantile: float = 0.75,
    confirm_mode: str = "none",
    lookback_periods: int = 8,
) -> pd.DataFrame:
    from src.strategies.base_strategy import StrategyConfig
    from src.strategies.ml_strategy import MLStockSelectionStrategy, SectorNeutralMLStrategy

    cls = SectorNeutralMLStrategy if sector_neutral else MLStockSelectionStrategy
    strat = cls(StrategyConfig(name=cls.__name__))
    _ensure_logger(strat)
    kwargs = dict(
        prediction_mode=prediction_mode,
        weight_method=weight_method,
        train_quarters=int(train_quarters),
        test_quarters=int(test_quarters),
        top_quantile=float(top_quantile),
        confirm_mode=confirm_mode,
        lookback_periods=int(lookback_periods),
    )
    result = strat.generate_weights(ds, **kwargs)
    if result.weights is None or result.weights.empty:
        meta = result.metadata or {}
        raise RuntimeError("ML 未产出权重: %s" % meta.get("error", meta))
    return _normalize_weights(result.weights)


def run_ml_bucket(ds: Dict[str, pd.DataFrame], **ml_kwargs) -> pd.DataFrame:
    """用 A 股 gsector 映射到四桶后，按桶调用 SectorNeutral 逻辑（同行业中性）。"""
    fund = ds["fundamentals"].copy()
    if "gsector" not in fund.columns or fund["gsector"].dropna().empty:
        raise RuntimeError("分桶 ML 需要 gsector（请启用行业映射 / 选择行业中性相关策略）")
    # 直接复用行业中性：每行业独立训练再合并，等价于 bucket 思路的 A 股落地
    return run_ml_strategy(ds, sector_neutral=True, **ml_kwargs)



def run_finrl_rolling(ds, **kwargs):
    """Full FinRL rolling DRL (distinct from short ashare_drl_runner)."""
    from src.strategies.ashare_finrl_drl import generate_weights_from_dataset
    return _normalize_weights(generate_weights_from_dataset(ds, **kwargs))


def run_drl(ds: Dict[str, pd.DataFrame], algo: str = "ppo", **kwargs) -> pd.DataFrame:
    from src.strategies.ashare_drl_runner import generate_drl_weights
    return _normalize_weights(generate_drl_weights(ds["prices"], algo=algo, **kwargs))


def run_ml_then_drl(ds: Dict[str, pd.DataFrame], drl_algo: str = "ppo", **ml_kwargs) -> pd.DataFrame:
    """用例2：先 ML 选出股票，再在子集上 DRL 定权。"""
    ml_w = run_ml_strategy(ds, sector_neutral=False, **ml_kwargs)
    codes = set(ml_w["tic"].astype(str).str.zfill(6).tolist())
    px = ds["prices"].copy()
    px["tic"] = px["tic"].astype(str).str.zfill(6)
    px = px[px["tic"].isin(codes)]
    if px.empty:
        return ml_w
    try:
        drl_w = run_drl({"prices": px, "fundamentals": ds["fundamentals"]}, algo=drl_algo, top_n=None)
        # 只保留 ML 池内
        drl_w = drl_w[drl_w["tic"].isin(codes)]
        if drl_w.empty:
            return ml_w
        return _normalize_weights(drl_w)
    except Exception as e:
        logger.warning("ML+DRL 的 DRL 阶段失败，回退 ML 权重: %s", e)
        return ml_w


def run_adaptive(ds: Dict[str, pd.DataFrame], as_of_date: Optional[str] = None) -> pd.DataFrame:
    from src.strategies.ashare_adaptive_bridge import run_adaptive_ashare
    fund = ds.get("fundamentals", pd.DataFrame())
    ind = {}
    if not fund.empty and "gsector" in fund.columns:
        tmp = fund[["tic", "gsector"]].dropna()
        tmp["tic"] = tmp["tic"].astype(str).str.zfill(6)
        ind = dict(zip(tmp["tic"], tmp["gsector"].astype(str)))
    return _normalize_weights(run_adaptive_ashare(ds["prices"], industry_map=ind, as_of_date=as_of_date))


def load_custom_strategy(path: str, class_name: Optional[str] = None):
    """加载用户 Python 文件中的 BaseStrategy 子类。"""
    path = str(path).strip()
    if not path:
        raise ValueError("请填写自定义策略文件路径")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("自定义策略文件不存在: %s" % path)
    spec = importlib.util.spec_from_file_location("ashare_custom_strategy", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载模块: %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ashare_custom_strategy"] = mod
    spec.loader.exec_module(mod)

    from src.strategies.base_strategy import BaseStrategy, StrategyConfig

    candidates = []
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
            candidates.append((name, obj))
    if class_name:
        for name, obj in candidates:
            if name == class_name:
                inst = obj(StrategyConfig(name=name))
                _ensure_logger(inst)
                return inst
        raise ValueError("未找到类 %s，可选: %s" % (class_name, [c[0] for c in candidates]))
    if not candidates:
        raise ValueError("文件中未找到 BaseStrategy 子类")
    name, obj = candidates[0]
    inst = obj(StrategyConfig(name=name))
    _ensure_logger(inst)
    return inst


def run_custom(ds: Dict[str, pd.DataFrame], path: str, class_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
    strat = load_custom_strategy(path, class_name=class_name)
    result = strat.generate_weights(ds, **kwargs)
    if hasattr(result, "weights"):
        return _normalize_weights(result.weights)
    if isinstance(result, pd.DataFrame):
        return _normalize_weights(result)
    raise RuntimeError("自定义策略返回无法识别的结果")



def classical_market_cap(
    ds: Dict[str, pd.DataFrame],
    *,
    mv_field: str = "total_mv",
    tdx_dir: Optional[str] = None,
    as_of: Optional[str] = None,
) -> pd.DataFrame:
    """Market-cap weights from TDX gbbq shares (wan-gu) x close."""
    from src.data.tdx_gbbq import enrich_dataset_market_cap, latest_market_caps

    mv_field = str(mv_field or "total_mv").strip()
    if mv_field not in ("total_mv", "circ_mv"):
        raise ValueError("mv_field must be total_mv or circ_mv, got: %s" % mv_field)

    prices = ds.get("prices")
    if prices is None or prices.empty:
        raise RuntimeError("market_cap needs prices panel")

    caps = ds.get("market_caps")
    if caps is None or (isinstance(caps, pd.DataFrame) and (caps.empty or mv_field not in getattr(caps, "columns", []))):
        # Prefer lightweight latest caps (no full-panel attach) for weighting.
        codes = prices["tic"].astype(str).str.zfill(6).unique().tolist()
        try:
            caps = latest_market_caps(codes, prices, tdx_dir=tdx_dir, as_of=as_of)
        except Exception:
            ds = enrich_dataset_market_cap(ds, tdx_dir=tdx_dir, attach_to_prices=True)
            caps = ds.get("market_caps")
            prices = ds.get("prices", prices)
            if caps is None or (isinstance(caps, pd.DataFrame) and (caps.empty or mv_field not in caps.columns)):
                caps = latest_market_caps(codes, prices, tdx_dir=tdx_dir, as_of=as_of)

    wdf = caps.copy()
    wdf["tic"] = wdf["tic"].astype(str).str.zfill(6)
    wdf[mv_field] = pd.to_numeric(wdf[mv_field], errors="coerce")
    wdf = wdf[wdf[mv_field].notna() & (wdf[mv_field] > 0)].copy()
    if wdf.empty:
        raise RuntimeError(
            "no positive market caps for weighting (field=%s); check gbbq under tdx_dir and pytdx" % mv_field
        )
    total = float(wdf[mv_field].sum())
    wdf["weight"] = wdf[mv_field] / total
    wdf["gvkey"] = wdf["tic"]
    out = _normalize_weights(wdf[["gvkey", "tic", "weight"]])
    out.attrs["mv_field"] = mv_field
    out.attrs["n_skipped_no_mv"] = int(len(caps) - len(wdf)) if caps is not None else 0
    return out


def generate_weights(
    family: str,
    ds: Dict[str, pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """统一入口。返回 (weights_df, meta)。"""
    params = dict(params or {})
    family = str(family or "ml").strip()
    meta: Dict[str, Any] = {"family": family}

    if family == "market_cap":
        w = classical_market_cap(
            ds,
            mv_field=params.get("mv_field", "total_mv"),
            tdx_dir=params.get("tdx_dir"),
            as_of=params.get("as_of_date"),
        )
        meta["mv_field"] = params.get("mv_field", "total_mv")
        meta["n_skipped_no_mv"] = getattr(w, "attrs", {}).get("n_skipped_no_mv")
        meta["n_names"] = len(w)
        return w, meta


    if family == "equal_weight":
        w = equal_weight_from_universe(ds["fundamentals"])
    elif family == "min_variance":
        w = classical_min_variance(ds["prices"], lookback=int(params.get("lookback_periods", 252)))
    elif family == "mean_variance":
        w = classical_mean_variance(ds["prices"], lookback=int(params.get("lookback_periods", 252)))
    elif family == "ml":
        w = run_ml_strategy(ds, sector_neutral=False, **_ml_params(params))
    elif family == "sector_neutral_ml":
        w = run_ml_strategy(ds, sector_neutral=True, **_ml_params(params))
    elif family == "ml_bucket":
        w = run_ml_bucket(ds, **_ml_params(params))
    elif family == "finrl_rolling_drl":
        finrl_kwargs = {
            "algos": params.get("finrl_algos") or ("a2c", "ppo", "ddpg", "td3", "sac"),
            "testing_window_days": int(params.get("finrl_testing_window_days", 365)),
            "max_rolling_window_days": int(params.get("finrl_max_rolling_window_days", 1095)),
            "lookback": int(params.get("finrl_lookback", 252)),
            "rebalance_freq": params.get("finrl_rebalance_freq", "Q"),
            "seed": int(params.get("finrl_seed", 42)),
            "resume": bool(params.get("finrl_resume", True)),
            "output_dir": params.get("finrl_output_dir") or None,
            "max_stocks": params.get("finrl_max_stocks"),
            "max_windows": params.get("finrl_max_windows"),
            "total_timesteps": params.get("finrl_total_timesteps"),
            "k_eig": params.get("finrl_k_eig"),
        }
        # optional per-algo timesteps override via dict
        if params.get("finrl_timesteps_by_algo"):
            finrl_kwargs["timesteps_by_algo"] = params["finrl_timesteps_by_algo"]
        # Forward UI/CLI progress callback (not part of persisted meta).
        progress_cb = params.get("progress")
        if progress_cb is not None:
            finrl_kwargs["progress"] = progress_cb
        w = run_finrl_rolling(ds, **{k: v for k, v in finrl_kwargs.items() if v is not None})
        meta["finrl_meta"] = getattr(w, "attrs", {}).get("finrl_meta")
        meta["drl_kind"] = "finrl_rolling"
    elif family == "drl":
        w = run_drl(
            ds,
            algo=params.get("drl_algo", "ppo"),
            lookback=int(params.get("lookback_periods", 252)),
            epochs=int(params.get("drl_epochs", 40)),
            timesteps=int(params.get("drl_timesteps", 8000)),
            top_n=params.get("drl_top_n"),
        )
        meta["drl_backend"] = getattr(w, "attrs", {}).get("drl_backend")
        meta["drl_kind"] = "short_ashare_drl_runner"
    elif family == "ml_then_drl":
        w = run_ml_then_drl(
            ds,
            drl_algo=params.get("drl_algo", "ppo"),
            **_ml_params(params),
        )
    elif family == "adaptive_rotation":
        w = run_adaptive(ds, as_of_date=params.get("as_of_date"))
        meta["adaptive_as_of"] = getattr(w, "attrs", {}).get("adaptive_as_of")
        meta["adaptive_regime"] = getattr(w, "attrs", {}).get("adaptive_regime")
    elif family == "custom":
        w = run_custom(
            ds,
            path=params.get("custom_path", ""),
            class_name=params.get("custom_class") or None,
            **{k: v for k, v in params.items() if k.startswith("ml_") or k in (
                "prediction_mode", "weight_method", "train_quarters", "test_quarters",
                "top_quantile", "confirm_mode", "lookback_periods",
            )},
        )
    else:
        raise ValueError("未知策略族: %s" % family)

    meta["n_names"] = len(w)
    return w, meta


def _ml_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        prediction_mode=params.get("prediction_mode", "single"),
        weight_method=params.get("weight_method", "min_variance"),
        train_quarters=int(params.get("train_quarters", 16)),
        test_quarters=int(params.get("test_quarters", 4)),
        top_quantile=float(params.get("top_quantile", 0.75)),
        confirm_mode=params.get("confirm_mode", "none"),
        lookback_periods=int(params.get("lookback_periods", 8)),
    )


def family_needs_industry(family: str) -> bool:
    return family in ("sector_neutral_ml", "ml_bucket", "adaptive_rotation")
