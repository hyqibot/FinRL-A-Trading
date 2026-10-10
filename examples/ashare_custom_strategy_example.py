# -*- coding: utf-8 -*-
"""自定义策略示例：等权持有基本面里出现的全部 gvkey。

在 A 股页策略族选「custom」，路径指向本文件即可。
也可写成更复杂的 ML/规则，只要继承 BaseStrategy 并实现 generate_weights。
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.strategies.base_strategy import BaseStrategy, StrategyConfig, StrategyResult


class SimpleEqualWeightCustom(BaseStrategy):
    """最简单的自定义示例。"""

    def generate_weights(self, data: Dict[str, pd.DataFrame], target_date: Optional[str] = None, **kwargs) -> StrategyResult:
        fund = data.get("fundamentals")
        if fund is None or fund.empty:
            raise ValueError("fundamentals 为空")
        col = "gvkey" if "gvkey" in fund.columns else "tic"
        tickers = fund[col].astype(str).str.zfill(6).unique().tolist()
        weight = 1.0 / max(1, len(tickers))
        weights_df = pd.DataFrame({"gvkey": tickers, "tic": tickers, "weight": [weight] * len(tickers)})
        return StrategyResult(strategy_name=self.config.name, weights=weights_df, metadata={"n": len(tickers)})
