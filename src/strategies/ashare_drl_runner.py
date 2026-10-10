# -*- coding: utf-8 -*-
"""A股 DRL 权重生成（不依赖硬编码美股 CSV）。

优先路径：若已安装 stable_baselines3 + gymnasium，用简易组合环境训 PPO/A2C/DDPG。
回退路径：纯 torch 的 softmax 策略梯度（REINFORCE），保证无 finrl 也能出权重。

输入：ashare build_dataset 的 prices 长表 ['date','tic','close',...]
输出：DataFrame columns ['gvkey','tic','weight']
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ashare_drl_runner")


def _pivot_returns(prices: pd.DataFrame, lookback: int = 252) -> Tuple[pd.DataFrame, List[str]]:
    if prices is None or getattr(prices, "empty", True):
        raise ValueError("prices 为空，无法跑 DRL")
    df = prices.copy()
    if "tic" not in df.columns or "close" not in df.columns:
        raise ValueError("prices 需要 tic/close 列")
    df["date"] = pd.to_datetime(df["date"])
    df["tic"] = df["tic"].astype(str).str.zfill(6)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    # 去重
    df = df.sort_values(["tic", "date"]).drop_duplicates(subset=["tic", "date"], keep="last")
    wide = df.pivot(index="date", columns="tic", values="close").sort_index()
    # 至少 60% 非空
    cover = wide.notna().mean()
    keep = cover[cover >= 0.6].index.tolist()
    if len(keep) < 2:
        raise ValueError("有效价格序列不足 2 只，无法 DRL")
    wide = wide[keep].ffill().dropna(how="any")
    if len(wide) < max(60, lookback // 2):
        raise ValueError("交易日过少（%d），无法 DRL" % len(wide))
    if lookback and len(wide) > lookback:
        wide = wide.iloc[-lookback:]
    rets = wide.pct_change().dropna(how="any")
    if rets.empty or rets.shape[1] < 2:
        raise ValueError("收益矩阵无效")
    return rets, list(rets.columns)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


def _torch_reinforce(rets: pd.DataFrame, algo: str = "ppo", epochs: int = 40, seed: int = 42) -> np.ndarray:
    """轻量策略梯度：最大化夏普近似。algo 仅影响学习率/噪声，接口统一。"""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    n = rets.shape[1]
    data = torch.tensor(rets.values, dtype=torch.float32)
    # 简单策略：logits 参数
    logits = nn.Parameter(torch.zeros(n))
    lr = {"a2c": 0.08, "ppo": 0.05, "ddpg": 0.03}.get(str(algo).lower(), 0.05)
    opt = torch.optim.Adam([logits], lr=lr)

    for ep in range(max(1, int(epochs))):
        weights = torch.softmax(logits, dim=0)
        port = data @ weights
        mu = port.mean()
        sigma = port.std(unbiased=False) + 1e-8
        sharpe = mu / sigma
        # 轻微熵正则，避免塌缩到单票
        entropy = -(weights * (weights + 1e-12).log()).sum()
        loss = -sharpe - 0.01 * entropy
        if str(algo).lower() == "ppo":
            # PPO 味：对权重变化加 clip 惩罚（相对上一轮）
            if ep == 0:
                prev = weights.detach().clone()
            else:
                ratio = weights / (prev + 1e-8)
                clip_pen = torch.mean(torch.clamp(ratio, 0.8, 1.2) - ratio).abs()
                loss = loss + 0.05 * clip_pen
                prev = weights.detach().clone()
        elif str(algo).lower() == "ddpg":
            # DDPG 味：平滑目标
            if ep == 0:
                target = logits.detach().clone()
            else:
                target = 0.95 * target + 0.05 * logits.detach()
                loss = loss + 0.01 * torch.mean((logits - target) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.softmax(logits, dim=0).cpu().numpy()
    return w


def _sb3_train(rets: pd.DataFrame, algo: str = "ppo", timesteps: int = 8000, seed: int = 42) -> Optional[np.ndarray]:
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception as e:
        logger.info("gymnasium unavailable: %s", e)
        return None
    try:
        from stable_baselines3 import A2C, PPO, DDPG
        from stable_baselines3.common.noise import NormalActionNoise
    except Exception as e:
        logger.info("stable_baselines3 unavailable: %s", e)
        return None

    arr = rets.values.astype(np.float32)
    n_assets = arr.shape[1]
    window = min(20, max(5, arr.shape[0] // 10))

    class PortfolioEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(n_assets,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(window * n_assets,), dtype=np.float32
            )
            self._i = window
            self._w = np.ones(n_assets, dtype=np.float32) / n_assets

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._i = window
            self._w = np.ones(n_assets, dtype=np.float32) / n_assets
            obs = arr[self._i - window:self._i].reshape(-1)
            return obs.astype(np.float32), {}

        def step(self, action):
            w = np.exp(np.asarray(action, dtype=np.float32))
            w = w / (w.sum() + 1e-8)
            r = float(arr[self._i] @ w)
            self._w = w
            self._i += 1
            done = self._i >= len(arr) - 1
            obs = arr[self._i - window:self._i].reshape(-1).astype(np.float32)
            # reward = return - turnover penalty
            reward = r
            return obs, reward, done, False, {"weights": w}

    env = PortfolioEnv()
    algo_l = str(algo).lower()
    if algo_l == "a2c":
        model = A2C("MlpPolicy", env, verbose=0, seed=seed)
    elif algo_l == "ddpg":
        noise = NormalActionNoise(mean=np.zeros(n_assets), sigma=0.1 * np.ones(n_assets))
        model = DDPG("MlpPolicy", env, action_noise=noise, verbose=0, seed=seed)
    else:
        model = PPO("MlpPolicy", env, verbose=0, seed=seed)
    model.learn(total_timesteps=int(timesteps))
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    w = np.exp(np.asarray(action, dtype=float))
    w = w / (w.sum() + 1e-8)
    return w


def generate_drl_weights(
    prices: pd.DataFrame,
    algo: str = "ppo",
    lookback: int = 252,
    epochs: int = 40,
    timesteps: int = 8000,
    top_n: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """生成 DRL 组合权重。"""
    rets, tickers = _pivot_returns(prices, lookback=lookback)
    # 可选：先按波动/动量筛 top_n 降低维度
    if top_n is not None and int(top_n) > 0 and len(tickers) > int(top_n):
        mom = rets.iloc[-60:].mean() if len(rets) >= 60 else rets.mean()
        keep = mom.sort_values(ascending=False).head(int(top_n)).index.tolist()
        rets = rets[keep]
        tickers = keep
        logger.info("DRL top_n=%d -> %s...", top_n, tickers[:5])

    backend = "torch_reinforce"
    w = None
    try:
        w = _sb3_train(rets, algo=algo, timesteps=timesteps, seed=seed)
        if w is not None:
            backend = "stable_baselines3/%s" % algo
    except Exception as e:
        logger.warning("sb3 train failed, fallback torch: %s", e)
        w = None
    if w is None:
        w = _torch_reinforce(rets, algo=algo, epochs=epochs, seed=seed)
        backend = "torch_reinforce/%s" % algo

    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        w = np.ones(len(tickers)) / len(tickers)
    else:
        w = w / w.sum()
    out = pd.DataFrame({
        "gvkey": [str(t).zfill(6) for t in tickers],
        "tic": [str(t).zfill(6) for t in tickers],
        "weight": w,
    })
    out.attrs["drl_backend"] = backend
    logger.info("DRL weights ready backend=%s n=%d", backend, len(out))
    return out
