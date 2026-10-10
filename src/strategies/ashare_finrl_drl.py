# -*- coding: utf-8 -*-
"""A-share full port of US FinRL rolling DRL (rl_model.run_models + fundamental_portfolio_drl.main).

Pipeline (parity with US path):
  1) Build OHLCV panel from ashare caches (6-digit tic)
  2) Technical indicators (FinRL INDICATORS names)
  3) Rolling cov/return lookback features
  4) For each rebalance trade_date: train A2C/PPO/DDPG/(TD3/SAC) on train window,
     evaluate on test window, pick best by cumulative return, infer weights on trade window
  5) Persist models, progress, performance CSVs, drl_weight.csv
  6) Export latest weights as ['gvkey','tic','weight'] for ashare_executor

Distinct from ashare_drl_runner (short SB3/torch Sharpe optimizer).
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.strategies.ashare_finrl_env import (
    DEFAULT_TIMESTEPS,
    DEVICE,
    INDICATORS,
    USE_GPU,
    DRLAgent,
    StockPortfolioEnv,
    add_technical_indicators,
    check_sb3_available,
    data_split,
    default_algo_params,
    safe_drl_prediction,
)

logger = logging.getLogger("ashare_finrl_drl")

ProgressCb = Optional[Callable[[str], None]]

# Module-level live progress path (set by generate_finrl_rolling_weights).
_LIVE_PROGRESS_PATH: Optional[Path] = None
_LIVE_PROGRESS_STATE: Dict[str, Any] = {}


def set_live_progress_path(path: Optional[Path]) -> None:
    global _LIVE_PROGRESS_PATH
    _LIVE_PROGRESS_PATH = Path(path) if path else None


def write_live_progress(msg: str, **extra: Any) -> None:
    """Overwrite live_progress.txt + live_progress.json for UI/CLI refresh."""
    global _LIVE_PROGRESS_STATE
    path = _LIVE_PROGRESS_PATH
    if path is None:
        return
    try:
        state = dict(_LIVE_PROGRESS_STATE)
        state.update(extra)
        state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["last_msg"] = msg
        _LIVE_PROGRESS_STATE = state
        txt_path = path if path.suffix.lower() == ".txt" else path.with_suffix(".txt")
        json_path = txt_path.with_name("live_progress.json")
        lines = [
            "updated_at: %s" % state["updated_at"],
            "last_msg: %s" % msg,
        ]
        for k in (
            "status",
            "window_idx",
            "window_total",
            "trade_date",
            "algo",
            "phase",
            "best_algo",
            "output_dir",
        ):
            if k in state and state[k] is not None:
                lines.append("%s: %s" % (k, state[k]))
        if "returns" in state and isinstance(state["returns"], dict):
            lines.append("returns: %s" % state["returns"])
        os.makedirs(txt_path.parent, exist_ok=True)
        tmp = str(txt_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, str(txt_path))
        atomic_write_json(state, str(json_path))
    except Exception:
        pass


def touch_progress_json(path: Path, **fields: Any) -> None:
    """Update progress.json without advancing last_idx (mid-window / per-algo)."""
    try:
        data = load_progress(path)
        data.update(fields)
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(data, str(path))
    except Exception:
        pass


def _log(msg: str, progress: ProgressCb = None, **extra: Any) -> None:
    logger.info(msg)
    print(msg, flush=True)
    write_live_progress(msg, **extra)
    if progress:
        try:
            progress(msg)
        except Exception:
            pass


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir() -> Path:
    return _project_root() / "artifacts" / "ashare_drl"


def atomic_to_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=index, encoding="utf-8-sig")
    os.replace(tmp, path)


def atomic_write_json(obj: dict, path: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def load_progress(path: Path) -> dict:
    if not path.is_file():
        return {"last_idx": -1, "df_dict": {"trade_date": [], "gvkey": [], "weights": []}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_idx": -1, "df_dict": {"trade_date": [], "gvkey": [], "weights": []}}


def save_progress(path: Path, idx: int, trade_date, df_dict: dict) -> None:
    atomic_write_json(
        {
            "last_idx": int(idx),
            "last_trade_date": str(pd.to_datetime(trade_date)),
            "df_dict": df_dict,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        str(path),
    )


def check_per_date_stock_coverage(df_: pd.DataFrame, stock_dim: int) -> pd.DataFrame:
    """Keep only dates where all stock_dim names appear (US helper parity)."""
    if "date" not in df_.columns:
        raise ValueError("check_per_date_stock_coverage needs date column")
    counts = df_.groupby("date").size()
    good_dates = counts[counts == stock_dim].index
    out = df_[df_["date"].isin(good_dates)].copy()
    if out.empty:
        raise RuntimeError("after coverage filter no dates remain (stock_dim=%d)" % stock_dim)
    out = out.sort_values(["date", "tic"], ignore_index=True)
    out.index = out["date"].factorize()[0]
    return out


def compute_and_save_performance(
    df_daily_return: pd.DataFrame,
    df_actions: pd.DataFrame,
    out_prefix: str = "backtest",
    results_dir: str = "results",
    rf_annual: float = 0.02,
    trading_days: int = 252,
) -> pd.DataFrame:
    if not isinstance(df_daily_return, pd.DataFrame) or df_daily_return.empty:
        raise ValueError("df_daily_return is None or empty")
    if "daily_return" not in df_daily_return.columns:
        raise ValueError("df_daily_return must contain daily_return")
    os.makedirs(results_dir, exist_ok=True)

    dr = df_daily_return.copy()
    if "date" in dr.columns:
        dr = dr.sort_values("date").set_index("date")
    rets = pd.to_numeric(dr["daily_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + rets).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    n = max(len(rets), 1)
    ann_return = float((1.0 + total_return) ** (trading_days / n) - 1.0) if n > 1 else total_return
    ann_vol = float(rets.std(ddof=0) * np.sqrt(trading_days))
    sharpe = float((ann_return - rf_annual) / (ann_vol + 1e-12))
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    eq_df = pd.DataFrame({"equity": equity, "drawdown": dd})
    eq_df.to_csv(os.path.join(results_dir, "%s_equity_curve.csv" % out_prefix), encoding="utf-8-sig")

    wsum = df_actions.sum(axis=1) if df_actions is not None and not df_actions.empty else pd.Series(dtype=float)
    wsum.to_csv(os.path.join(results_dir, "%s_weights_sum.csv" % out_prefix), encoding="utf-8-sig")

    if df_actions is not None and len(df_actions) >= 2:
        turnover = df_actions.diff().abs().sum(axis=1).fillna(0.0)
    else:
        turnover = pd.Series(dtype=float)
    turnover.to_csv(os.path.join(results_dir, "%s_turnover.csv" % out_prefix), encoding="utf-8-sig")

    summary_df = pd.DataFrame(
        [
            {
                "total_return": total_return,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "n_days": int(n),
                "avg_turnover": float(turnover.mean()) if len(turnover) else 0.0,
                "avg_weight_sum": float(wsum.mean()) if len(wsum) else 0.0,
            }
        ]
    )
    summary_df.to_csv(
        os.path.join(results_dir, "%s_summary.csv" % out_prefix), index=False, encoding="utf-8-sig"
    )
    return summary_df


def normalize_ashare_prices(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    if "tic" not in df.columns or "close" not in df.columns:
        raise ValueError("prices need tic/close")
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.zfill(6)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df.dropna(subset=["close"])
    df = df.sort_values(["tic", "date"]).drop_duplicates(subset=["tic", "date"], keep="last")
    df["adjcp"] = df["close"]
    return df.reset_index(drop=True)


def build_cov_features(df: pd.DataFrame, lookback: int = 252, progress: ProgressCb = None) -> pd.DataFrame:
    """Attach cov_list / return_list like fundamental_portfolio_drl.main."""
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]
    cov_list = []
    return_list = []
    uniq = df.index.unique()
    _log("building cov features lookback=%d unique_days=%d" % (lookback, len(uniq)), progress)
    for i in range(lookback, len(uniq)):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(index="date", columns="tic", values="close")
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        covs = return_lookback.cov().values
        cov_list.append(covs)
    df_cov = pd.DataFrame(
        {"date": df.date.unique()[lookback:], "cov_list": cov_list, "return_list": return_list}
    )
    out = df.merge(df_cov, on="date")
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    out.index = out.date.factorize()[0]
    return out


def make_trade_dates(
    dates: Sequence[pd.Timestamp],
    freq: str = "Q",
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
) -> List[pd.Timestamp]:
    """Quarterly/monthly rebalance dates from available trading calendar (A-share adaption of US selected_stock dates)."""
    s = pd.Series(pd.to_datetime(list(dates))).drop_duplicates().sort_values()
    if min_date is not None:
        s = s[s >= pd.to_datetime(min_date)]
    if max_date is not None:
        s = s[s <= pd.to_datetime(max_date)]
    if s.empty:
        return []
    # period end labels
    rule = "QE" if str(freq).upper() in ("Q", "QE", "QUARTER", "QUARTERLY") else "ME"
    period_ends = s.groupby(s.dt.to_period("Q" if rule == "QE" else "M")).max()
    return [pd.Timestamp(x) for x in period_ends.tolist()]


def prepare_rolling_train(df, testing_window, max_rolling_window, trade_date):
    return data_split(df, trade_date - max_rolling_window, trade_date - testing_window)


def prepare_rolling_test(df, testing_window, max_rolling_window, trade_date):
    return data_split(df, trade_date - testing_window, trade_date)


def run_models(
    df: pd.DataFrame,
    trade_date,
    env_kwargs: dict,
    testing_window,
    max_rolling_window,
    algos: Sequence[str] = ("a2c", "ppo", "ddpg"),
    timesteps_by_algo: Optional[Dict[str, int]] = None,
    seed: int = 42,
    model_dir: Optional[Path] = None,
    progress: ProgressCb = None,
) -> Tuple[Dict[str, Any], Any, Dict[str, float]]:
    """Parameterized equivalent of rl_model.run_models (train all algos, pick best on test window)."""
    ok, err = check_sb3_available()
    if not ok:
        raise RuntimeError(err)

    timesteps_by_algo = dict(timesteps_by_algo or {})
    algos = [str(a).lower() for a in algos]
    for a in algos:
        if a not in DRLAgent.SUPPORTED:
            raise ValueError("unsupported algo %s" % a)

    X_train = prepare_rolling_train(df, testing_window, max_rolling_window, trade_date)
    X_test = prepare_rolling_test(df, testing_window, max_rolling_window, trade_date)
    if X_train.empty or X_test.empty:
        raise RuntimeError(
            "empty train/test window for trade_date=%s train=%s test=%s"
            % (trade_date, X_train.shape, X_test.shape)
        )

    # ensure coverage
    stock_dim = int(env_kwargs["stock_dim"])
    X_train = check_per_date_stock_coverage(X_train, stock_dim)
    X_test = check_per_date_stock_coverage(X_test, stock_dim)

    results_dir = env_kwargs.get("results_dir", "results")
    e_train_gym = StockPortfolioEnv(df=X_train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)

    trained: Dict[str, Any] = {}
    returns: Dict[str, float] = {}
    best_model = None
    max_return = -np.inf

    e_trade_gym = StockPortfolioEnv(df=X_test, **env_kwargs)

    if model_dir is not None:
        os.makedirs(model_dir, exist_ok=True)

    for algo in algos:
        ts = int(timesteps_by_algo.get(algo, DEFAULT_TIMESTEPS.get(algo, 10000)))
        _log(
            "training %s timesteps=%d trade_date=%s" % (algo, ts, trade_date),
            progress,
            status="running",
            algo=algo,
            phase="training",
            trade_date=str(pd.to_datetime(trade_date)),
        )
        params = default_algo_params(algo, use_gpu=USE_GPU)
        # shrink n_steps if tiny timesteps (smoke)
        if algo in ("a2c", "ppo") and "n_steps" in params:
            params["n_steps"] = int(min(params["n_steps"], max(8, ts // 2)))
        if algo == "ppo" and "batch_size" in params:
            params["batch_size"] = int(min(params["batch_size"], max(8, params["n_steps"])))
        if algo == "sac":
            params["learning_starts"] = int(min(params.get("learning_starts", 100), max(1, ts // 4)))

        model = agent.get_model(algo, model_kwargs=params, seed=seed)
        model = agent.train_model(model=model, total_timesteps=ts, tb_log_name=algo)
        trained[algo] = model

        if model_dir is not None:
            save_path = model_dir / ("%s.zip" % algo)
            try:
                model.save(str(save_path))
                _log("saved model %s" % save_path, progress)
            except Exception as e:
                logger.warning("save model failed %s: %s", save_path, e)

        df_daily_return, df_actions = safe_drl_prediction(model, e_trade_gym)
        cum = float((df_daily_return["daily_return"] + 1).cumprod().iloc[-1])
        returns[algo] = cum
        _log(
            "test cum_return %s=%.6f" % (algo, cum),
            progress,
            status="running",
            algo=algo,
            phase="algo_done",
            returns=dict(returns),
        )
        # Mid-window checkpoint so refresh can see which algo finished.
        try:
            if model_dir is not None:
                touch_progress_json(
                    Path(model_dir).parents[1] / "progress.json",
                    current_algo=algo,
                    current_phase="algo_done",
                    current_returns=dict(returns),
                    current_trade_date=str(pd.to_datetime(trade_date)),
                )
        except Exception:
            pass
        if cum > max_return:
            max_return = cum
            best_model = model

    meta = {"returns": returns, "best_return": max_return, "device": DEVICE}
    return trained, best_model, meta


def _env_kwargs_for(df_: pd.DataFrame, indicators: List[str], results_dir: str, k_eig: Optional[int] = None) -> dict:
    stock_dimension = len(df_.tic.unique())
    # Classic FinRL: state_space == stock_dim (cov block width)
    state_space = stock_dimension
    kw = {
        "hmax": 100,
        "initial_amount": 1000000,
        "transaction_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": indicators,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "results_dir": results_dir,
    }
    if k_eig is not None:
        kw["k_eig"] = int(k_eig)
    return kw


def latest_weights_from_actions(df_actions: pd.DataFrame) -> pd.DataFrame:
    if df_actions is None or df_actions.empty:
        raise RuntimeError("no actions to export")
    last = df_actions.iloc[-1]
    w = last.astype(float)
    w = w.clip(lower=0)
    if w.sum() <= 0:
        w = pd.Series(np.ones(len(w)) / len(w), index=w.index)
    else:
        w = w / w.sum()
    out = pd.DataFrame(
        {
            "gvkey": [str(t).zfill(6) for t in w.index],
            "tic": [str(t).zfill(6) for t in w.index],
            "weight": w.values,
        }
    )
    return out


def generate_finrl_rolling_weights(
    prices: pd.DataFrame,
    *,
    algos: Sequence[str] = ("a2c", "ppo", "ddpg", "td3", "sac"),
    testing_window_days: int = 365,
    max_rolling_window_days: int = 1095,
    lookback: int = 252,
    rebalance_freq: str = "Q",
    timesteps_by_algo: Optional[Dict[str, int]] = None,
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume: bool = True,
    output_dir: Optional[str] = None,
    max_stocks: Optional[int] = None,
    max_windows: Optional[int] = None,
    indicators: Optional[Sequence[str]] = None,
    k_eig: Optional[int] = None,
    selected_by_date: Optional[Dict[Any, Sequence[str]]] = None,
    progress: ProgressCb = None,
) -> pd.DataFrame:
    """Full rolling FinRL DRL; returns latest weight table for executor."""
    ok, err = check_sb3_available()
    if not ok:
        raise RuntimeError(err)

    set_global_seed(seed)
    indicators = list(indicators or INDICATORS)
    out_root = Path(output_dir) if output_dir else default_output_dir()
    results_dir = out_root / "results"
    models_root = out_root / "models"
    progress_path = out_root / "progress.json"
    live_path = out_root / "live_progress.txt"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    set_live_progress_path(live_path)
    _LIVE_PROGRESS_STATE.clear()
    _LIVE_PROGRESS_STATE["output_dir"] = str(out_root)
    _LIVE_PROGRESS_STATE["status"] = "starting"
    _log("live progress -> %s" % live_path, progress, status="starting", output_dir=str(out_root))

    if total_timesteps is not None:
        timesteps_by_algo = {a: int(total_timesteps) for a in algos}
    timesteps_by_algo = dict(timesteps_by_algo or {})

    df = normalize_ashare_prices(prices)
    if max_stocks is not None and int(max_stocks) > 0:
        # pick most liquid / longest history
        counts = df.groupby("tic").size().sort_values(ascending=False)
        keep = counts.head(int(max_stocks)).index.tolist()
        df = df[df["tic"].isin(keep)].copy()
        _log("max_stocks=%d -> %s" % (max_stocks, keep), progress)

    _log("adding technical indicators n=%d rows=%d" % (df["tic"].nunique(), len(df)), progress)
    df = add_technical_indicators(df, indicators=indicators)
    if df.empty:
        raise RuntimeError("no rows after technical indicators")

    # universe coverage: keep tics present on most dates after indicators
    wide = df.pivot_table(index="date", columns="tic", values="close")
    cover = wide.notna().mean()
    keep = cover[cover >= 0.85].index.tolist()
    if len(keep) < 2:
        keep = cover.sort_values(ascending=False).head(max(2, min(5, len(cover)))).index.tolist()
    df = df[df["tic"].isin(keep)].copy()
    _log("coverage filter -> %d stocks" % len(keep), progress)

    # Align to common dates (inner join)
    wide = df.pivot_table(index="date", columns="tic", values="close").dropna(how="any")
    if wide.empty or wide.shape[1] < 2:
        raise RuntimeError("insufficient aligned price panel for FinRL env")
    common_dates = set(wide.index)
    df = df[df["date"].isin(common_dates) & df["tic"].isin(wide.columns)].copy()

    df = build_cov_features(df, lookback=lookback, progress=progress)
    stock_dimension = len(df.tic.unique())
    _log("panel ready stock_dim=%d days=%d" % (stock_dimension, df["date"].nunique()), progress)

    testing_window = pd.Timedelta(days=int(testing_window_days))
    max_rolling_window = pd.Timedelta(days=int(max_rolling_window_days))

    all_dates = sorted(pd.to_datetime(df["date"].unique()))
    trade_dates = make_trade_dates(all_dates, freq=rebalance_freq)
    # need at least train history before first trade
    min_start = all_dates[0] + max_rolling_window
    trade_dates = [d for d in trade_dates if d >= min_start]
    if len(trade_dates) < 2:
        # fallback: use last two available month ends or synthetic windows from calendar
        if len(all_dates) < lookback + 30:
            raise RuntimeError(
                "not enough history for rolling DRL (days=%d lookback=%d)" % (len(all_dates), lookback)
            )
        # create synthetic rebalance points every testing_window
        trade_dates = []
        cur = min_start
        last = all_dates[-1]
        while cur < last:
            # snap to nearest available date <= cur
            cand = [d for d in all_dates if d <= cur]
            if cand:
                trade_dates.append(cand[-1])
            cur = cur + testing_window
        if all_dates[-1] not in trade_dates:
            trade_dates.append(all_dates[-1])
        trade_dates = sorted(set(trade_dates))
    if max_windows is not None and int(max_windows) > 0:
        trade_dates = trade_dates[-(int(max_windows) + 1) :]

    _log("trade_dates=%d first=%s last=%s" % (len(trade_dates), trade_dates[0], trade_dates[-1]), progress)

    prog = load_progress(progress_path) if resume else {"last_idx": -1, "df_dict": {"trade_date": [], "gvkey": [], "weights": []}}
    start_idx = max(1, int(prog.get("last_idx", -1)) + 1) if resume else 1
    df_dict = prog.get("df_dict") or {"trade_date": [], "gvkey": [], "weights": []}
    for k in ("trade_date", "gvkey", "weights"):
        df_dict.setdefault(k, [])

    last_actions = None
    best_name = None

    for idx in range(start_idx, len(trade_dates)):
        current_trade_date = trade_dates[idx - 1]
        next_trade_date = trade_dates[idx]
        _log(
            "window %d/%d trade=%s -> %s" % (idx, len(trade_dates) - 1, current_trade_date.date(), next_trade_date.date()),
            progress,
            status="running",
            window_idx=idx,
            window_total=len(trade_dates) - 1,
            trade_date=str(current_trade_date.date()),
            phase="window_start",
        )
        touch_progress_json(
            progress_path,
            current_window_idx=idx,
            current_window_total=len(trade_dates) - 1,
            current_trade_date=str(current_trade_date),
            current_phase="window_start",
        )

        # optional per-date selection (US all_stocks_info equivalent)
        if selected_by_date:
            key = None
            for k in selected_by_date:
                if pd.to_datetime(k) == current_trade_date:
                    key = k
                    break
            sel = selected_by_date.get(key) if key is not None else None
            if sel:
                sel = [str(x).zfill(6) for x in sel]
                df_win = df[df["tic"].isin(sel)].copy()
            else:
                df_win = df.copy()
        else:
            df_win = df.copy()

        earliest = current_trade_date - max_rolling_window
        df_ = df_win[(df_win["date"] >= earliest) & (df_win["date"] < next_trade_date)].copy()
        if df_.empty:
            _log("skip empty slice %s" % current_trade_date, progress)
            save_progress(progress_path, idx, next_trade_date, df_dict)
            continue

        df_ = df_.sort_values(["date", "tic"], ignore_index=True)
        stock_dimension = len(df_["tic"].unique())
        try:
            df_ = check_per_date_stock_coverage(df_, stock_dimension)
        except Exception as e:
            _log("coverage fail skip: %s" % e, progress)
            save_progress(progress_path, idx, next_trade_date, df_dict)
            continue

        env_kwargs = _env_kwargs_for(df_, indicators, str(results_dir), k_eig=k_eig)
        model_dir = models_root / current_trade_date.strftime("%Y%m%d")

        try:
            trained, best_model, meta = run_models(
                df_,
                current_trade_date,
                env_kwargs,
                testing_window,
                max_rolling_window,
                algos=algos,
                timesteps_by_algo=timesteps_by_algo,
                seed=seed,
                model_dir=model_dir,
                progress=progress,
            )
            atomic_write_json(meta, str(model_dir / "train_meta.json"))
            # pick best name
            best_name = None
            if meta.get("returns"):
                best_name = max(meta["returns"], key=meta["returns"].get)

            trade = data_split(df_, current_trade_date, next_trade_date)
            trade = check_per_date_stock_coverage(trade, env_kwargs["stock_dim"])
            e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)
            model_for_bt = best_model if best_model is not None else next(iter(trained.values()))
            _log("infer with best=%s" % (best_name or type(model_for_bt).__name__), progress)
            df_daily_return, df_actions = safe_drl_prediction(model_for_bt, e_trade_gym)
            last_actions = df_actions

            for i in range(len(df_actions)):
                for j in df_actions.columns:
                    df_dict["trade_date"].append(str(df_actions.index[i]))
                    df_dict["gvkey"].append(str(j).zfill(6))
                    df_dict["weights"].append(float(df_actions.loc[df_actions.index[i], j]))

            out_prefix = "bt_%s_%s" % (
                current_trade_date.strftime("%Y%m%d"),
                next_trade_date.strftime("%Y%m%d"),
            )
            try:
                summary_df = compute_and_save_performance(
                    df_daily_return=df_daily_return,
                    df_actions=df_actions,
                    out_prefix=out_prefix,
                    results_dir=str(results_dir),
                )
                _log("perf %s %s" % (out_prefix, summary_df.to_dict("records")), progress)
            except Exception as perf_err:
                _log("perf error %s: %s" % (out_prefix, perf_err), progress)

        except Exception as e:
            logger.exception("window failed")
            _log("ERROR window %s: %s" % (current_trade_date, e), progress)
        finally:
            save_progress(progress_path, idx, next_trade_date, df_dict)

    df_rl = pd.DataFrame(df_dict)
    weight_path = results_dir / "drl_weight.csv"
    atomic_to_csv(df_rl, str(weight_path), index=False)
    _log("wrote %s rows=%d" % (weight_path, len(df_rl)), progress)

    if last_actions is not None and not last_actions.empty:
        latest = latest_weights_from_actions(last_actions)
    elif not df_rl.empty:
        # average last trade_date weights
        last_d = df_rl["trade_date"].iloc[-1]
        sub = df_rl[df_rl["trade_date"] == last_d]
        latest = sub.groupby("gvkey", as_index=False)["weights"].mean()
        latest = latest.rename(columns={"weights": "weight"})
        latest["tic"] = latest["gvkey"].astype(str).str.zfill(6)
        latest["gvkey"] = latest["tic"]
        s = latest["weight"].sum()
        if s > 0:
            latest["weight"] = latest["weight"] / s
        latest = latest[["gvkey", "tic", "weight"]]
    else:
        raise RuntimeError(
            "FinRL滚动DRL 未产出任何权重（窗口被跳过或训练失败）。请查看 %s 与日志。" % out_root
        )

    latest_path = out_root / "latest_weights.csv"
    atomic_to_csv(latest, str(latest_path), index=False)
    meta_out = {
        "family": "finrl_rolling_drl",
        "output_dir": str(out_root),
        "drl_weight_csv": str(weight_path),
        "latest_weights_csv": str(latest_path),
        "algos": list(algos),
        "best_last": best_name,
        "n_names": len(latest),
        "device": DEVICE,
        "layout": {
            "models": "artifacts/ashare_drl/models/<YYYYMMDD>/<algo>.zip",
            "results": "artifacts/ashare_drl/results/bt_*_{summary,equity_curve,turnover,weights_sum}.csv",
            "progress": "artifacts/ashare_drl/progress.json",
            "live_progress": "artifacts/ashare_drl/live_progress.txt",
            "live_progress_json": "artifacts/ashare_drl/live_progress.json",
            "drl_weight": "artifacts/ashare_drl/results/drl_weight.csv",
            "latest": "artifacts/ashare_drl/latest_weights.csv",
        },
    }
    atomic_write_json(meta_out, str(out_root / "run_meta.json"))
    latest.attrs["finrl_meta"] = meta_out
    _log(
        "latest weights n=%d sum=%.4f" % (len(latest), float(latest["weight"].sum())),
        progress,
        status="done",
        phase="finished",
        best_algo=best_name,
        output_dir=str(out_root),
    )
    return latest


def generate_weights_from_dataset(ds: Dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    prices = ds.get("prices")
    if prices is None or getattr(prices, "empty", True):
        raise RuntimeError("finrl_rolling_drl needs prices panel from ashare_data_adapter")
    return generate_finrl_rolling_weights(prices, **kwargs)
