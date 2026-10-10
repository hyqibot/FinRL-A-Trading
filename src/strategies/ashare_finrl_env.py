# -*- coding: utf-8 -*-
"""FinRL-compatible portfolio env + DRLAgent for A-share (vendored).

Prefer finrl when present; otherwise local StockPortfolioEnv + SB3 DRLAgent
mirroring FinRL train/predict API. US scripts stay untouched.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ashare_finrl_env")

try:
    import gymnasium as _gym
except ImportError:  # pragma: no cover
    _gym = None



INDICATORS: List[str] = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

DEVICE = "cpu"
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    pass
USE_GPU = DEVICE == "cuda"


def data_split(df: pd.DataFrame, start, end, target_date_col: str = "date") -> pd.DataFrame:
    """Mirror finrl.meta.preprocessor.preprocessors.data_split."""
    data = df.copy()
    if target_date_col not in data.columns and "datadate" in data.columns:
        data = data.rename(columns={"datadate": target_date_col})
    data[target_date_col] = pd.to_datetime(data[target_date_col])
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    data = data[(data[target_date_col] >= start) & (data[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def add_technical_indicators(df: pd.DataFrame, indicators: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Add FinRL-named technical columns without requiring stockstats/finrl."""
    indicators = list(indicators or INDICATORS)
    need = {"date", "tic", "close", "high", "low"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError("prices missing columns for indicators: %s" % sorted(missing))

    out_parts = []
    for tic, g in df.groupby("tic", sort=False):
        g = g.sort_values("date").copy()
        c = pd.to_numeric(g["close"], errors="coerce")
        h = pd.to_numeric(g["high"], errors="coerce")
        l = pd.to_numeric(g["low"], errors="coerce")

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        boll_ub = mid + 2 * std
        boll_lb = mid - 2 * std
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(30).mean()
        loss = (-delta.clip(upper=0)).rolling(30).mean()
        rs = gain / (loss + 1e-12)
        rsi_30 = 100 - (100 / (1 + rs))
        tp = (h + l + c) / 3.0
        cci_30 = (tp - tp.rolling(30).mean()) / (0.015 * tp.rolling(30).std() + 1e-12)
        up = h.diff()
        down = -l.diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = pd.Series(tr, index=g.index).rolling(30).mean()
        plus_di = 100 * pd.Series(plus_dm, index=g.index).rolling(30).mean() / (atr + 1e-12)
        minus_di = 100 * pd.Series(minus_dm, index=g.index).rolling(30).mean() / (atr + 1e-12)
        dx_30 = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)
        close_30_sma = c.rolling(30).mean()
        close_60_sma = c.rolling(60).mean()

        mapping = {
            "macd": macd,
            "boll_ub": boll_ub,
            "boll_lb": boll_lb,
            "rsi_30": rsi_30,
            "cci_30": cci_30,
            "dx_30": dx_30,
            "close_30_sma": close_30_sma,
            "close_60_sma": close_60_sma,
        }
        for name in indicators:
            if name not in mapping:
                logger.warning("unknown indicator %s skipped", name)
                continue
            g[name] = mapping[name].values
        out_parts.append(g)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["date", "tic"], ignore_index=True)
    out = out.dropna(subset=list(indicators)).reset_index(drop=True)
    return out


def try_import_finrl():
    try:
        from finrl.agents.stablebaselines3.models import DRLAgent as FinRLAgent
        from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv as FinRLEnv
        from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split as finrl_split

        return FinRLEnv, FinRLAgent, FeatureEngineer, finrl_split, True, None
    except Exception as e:
        return None, None, None, None, False, e


class StockPortfolioEnv(_gym.Env if _gym is not None else object):
    """Local FinRL StockPortfolioEnv (gymnasium)."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        transaction_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        lookback=252,
        day=0,
        k_eig=None,
        results_dir: str = "results",
        **kwargs,
    ):
        try:
            from gymnasium import spaces
            from gymnasium.utils import seeding
        except ImportError as e:
            raise ImportError(
                "gymnasium is required for FinRL rolling DRL. "
                'Install: pip install "gymnasium>=0.29.0"'
            ) from e
        from stable_baselines3.common.vec_env import DummyVecEnv

        super().__init__()
        self._spaces = spaces
        self._seeding = seeding
        self._DummyVecEnv = DummyVecEnv

        self.day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = int(stock_dim)
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = int(state_space)
        self.action_space_dim = int(action_space)
        self.tech_indicator_list = list(tech_indicator_list)
        self.k_eig = k_eig
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        obs_rows = self.stock_dim + len(self.tech_indicator_list)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_rows, self.stock_dim), dtype=np.float32
        )

        self.data = self.df.loc[self.day, :]
        self.covs = self._cov_matrix(self.data)
        self.state = self._build_state(self.data, self.covs)
        self.terminal = False
        self.turbulence_threshold = turbulence_threshold
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1.0 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward = 0.0

    def _cov_matrix(self, data) -> np.ndarray:
        cov = np.asarray(data["cov_list"].values[0], dtype=np.float64)
        if cov.ndim != 2:
            raise ValueError("cov_list must be 2D, got shape %s" % (cov.shape,))
        if cov.shape[0] != self.stock_dim or cov.shape[1] != self.stock_dim:
            m = np.eye(self.stock_dim, dtype=np.float64)
            n = min(self.stock_dim, cov.shape[0], cov.shape[1])
            m[:n, :n] = cov[:n, :n]
            cov = m
        if self.k_eig is not None:
            k = int(self.k_eig)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                idx = np.argsort(eigvals)[::-1][:k]
                cov = (eigvecs[:, idx] * eigvals[idx]) @ eigvecs[:, idx].T
            except Exception:
                pass
        return cov

    def _build_state(self, data, covs) -> np.ndarray:
        tech_rows = [data[tech].values.tolist() for tech in self.tech_indicator_list]
        state = np.append(np.array(covs, dtype=np.float64), tech_rows, axis=0)
        return state.astype(np.float32)

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                df = pd.DataFrame({"daily_return": self.portfolio_return_memory})
                plt.figure()
                plt.plot(df.daily_return.cumsum(), "r")
                plt.savefig(os.path.join(self.results_dir, "cumulative_reward.png"))
                plt.close()
                plt.figure()
                plt.plot(self.portfolio_return_memory, "r")
                plt.savefig(os.path.join(self.results_dir, "rewards.png"))
                plt.close()
            except Exception:
                pass
            return self.state, float(self.reward), True, False, {}

        weights = self.softmax_normalization(actions)
        self.actions_memory.append(weights)
        last_day_memory = self.data
        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.covs = self._cov_matrix(self.data)
        self.state = self._build_state(self.data, self.covs)
        portfolio_return = float(
            np.sum(((self.data.close.values / last_day_memory.close.values) - 1.0) * weights)
        )
        self.portfolio_value = self.portfolio_value * (1.0 + portfolio_return)
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(self.portfolio_value)
        self.reward = float(self.portfolio_value)
        return self.state, self.reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.covs = self._cov_matrix(self.data)
        self.state = self._build_state(self.data, self.covs)
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1.0 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        return self.state, {}

    def render(self, mode="human"):
        return self.state

    def softmax_normalization(self, actions):
        actions = np.asarray(actions, dtype=np.float64).reshape(-1)
        actions = actions - np.max(actions)
        numerator = np.exp(actions)
        denominator = np.sum(numerator) + 1e-12
        return (numerator / denominator).astype(np.float64)

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "daily_return": self.portfolio_return_memory})

    def save_action_memory(self):
        df_actions = pd.DataFrame(self.actions_memory)
        df_actions.columns = list(self.data.tic.values)
        df_actions.index = pd.Index(self.date_memory, name="date")
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = self._seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = self._DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


class DRLAgent:
    """SB3 wrapper matching FinRL DRLAgent API."""

    SUPPORTED = ("a2c", "ppo", "ddpg", "td3", "sac")

    def __init__(self, env):
        self.env = env

    def get_model(self, model_name: str, model_kwargs: Optional[dict] = None, policy="MlpPolicy", seed: Optional[int] = None):
        from stable_baselines3 import A2C, PPO, DDPG, TD3, SAC
        from stable_baselines3.common.noise import NormalActionNoise

        model_name = str(model_name).lower()
        model_kwargs = dict(model_kwargs or {})
        if seed is not None:
            model_kwargs.setdefault("seed", seed)

        if model_name == "a2c":
            return A2C(policy, self.env, verbose=0, **model_kwargs)
        if model_name == "ppo":
            return PPO(policy, self.env, verbose=0, **model_kwargs)
        if model_name in ("ddpg", "td3", "sac"):
            n_actions = self.env.action_space.shape[-1]
            if "action_noise" not in model_kwargs and model_name in ("ddpg", "td3"):
                model_kwargs["action_noise"] = NormalActionNoise(
                    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
                )
            cls = {"ddpg": DDPG, "td3": TD3, "sac": SAC}[model_name]
            return cls(policy, self.env, verbose=0, **model_kwargs)
        raise ValueError("unsupported model_name=%s; choose from %s" % (model_name, self.SUPPORTED))

    def train_model(self, model, total_timesteps: int, tb_log_name: str = "", callback=None):
        kw = {}
        if tb_log_name:
            kw["tb_log_name"] = tb_log_name
        if callback is not None:
            kw["callback"] = callback
        model.learn(total_timesteps=int(total_timesteps), **kw)
        return model

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True):
        return safe_drl_prediction(model, environment, deterministic=deterministic)


def safe_drl_prediction(model, environment, deterministic=True):
    test_env, test_obs = environment.get_sb_env()
    test_env.reset()
    n_steps = len(environment.df.index.unique())
    max_steps = n_steps - 1
    account_memory = None
    actions_memory = None
    for i in range(n_steps):
        action, _ = model.predict(test_obs, deterministic=deterministic)
        step_out = test_env.step(action)
        if len(step_out) == 4:
            test_obs, rewards, dones, info = step_out
        else:
            test_obs, rewards, terminated, truncated, info = step_out
            dones = np.asarray(terminated) | np.asarray(truncated)
        if (i == max_steps) or bool(dones[0]):
            account_memory = test_env.env_method("save_asset_memory")
            actions_memory = test_env.env_method("save_action_memory")
            break
    if account_memory is None:
        account_memory = test_env.env_method("save_asset_memory")
    if actions_memory is None:
        actions_memory = test_env.env_method("save_action_memory")
    return account_memory[0], actions_memory[0]


def default_algo_params(algo: str, use_gpu: bool = False) -> Dict[str, Any]:
    algo = str(algo).lower()
    if algo == "a2c":
        if use_gpu:
            return {"n_steps": 1024, "ent_coef": 0.005, "learning_rate": 0.0002, "device": DEVICE}
        return {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
    if algo == "ppo":
        if use_gpu:
            return {
                "n_steps": 2048,
                "ent_coef": 0.005,
                "learning_rate": 0.0001,
                "batch_size": 1024,
                "device": DEVICE,
            }
        return {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 128}
    if algo == "ddpg":
        if use_gpu:
            return {"batch_size": 1024, "buffer_size": 100000, "learning_rate": 0.001, "device": DEVICE}
        return {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
    if algo == "td3":
        return {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
    if algo == "sac":
        return {
            "batch_size": 128,
            "buffer_size": 100000,
            "learning_rate": 0.0003,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        }
    raise ValueError("unknown algo %s" % algo)


DEFAULT_TIMESTEPS = {
    "a2c": 50000,
    "ppo": 80000,
    "ddpg": 50000,
    "td3": 30000,
    "sac": 50000,
}


def check_sb3_available() -> Tuple[bool, str]:
    try:
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401

        return True, ""
    except Exception as e:
        return False, (
            "FinRL滚动DRL 需要 stable-baselines3 与 gymnasium。请在项目 venv 中执行:\n"
            '  python -m pip install "stable-baselines3>=2.0.0" "gymnasium>=0.29.0" '
            "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "可选完整 FinRL:\n"
            '  python -m pip install "finrl" "stockstats" "pypfopt" '
            "-i https://pypi.tuna.tsinghua.edu.cn/simple\n"
            "原始错误: %s" % e
        )
