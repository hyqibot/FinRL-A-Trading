import os
import pandas as pd
from typing import Dict, Optional, Iterable
from strategies.strategylogger import StrategyLogger
from strategies.base_signal import BaseSignalEngine
class TSMOMSignalEngine(BaseSignalEngine):
    """
    TS-MOM (Moskowitz et al., 2012)
    --------------------------------
    严格使用“月度价格”计算信号：
        ret_12m = P(t-1m) / P(t-12m) - 1
    信号是月度频率（M），最终会在 BaseSignalEngine 中扩展成 daily。
    """

    def __init__(
        self,
        strategy_name="tsmom",
        col_map=None,
        universe_mgr=None,
        logger=None,
        chunk_size=200000,
        multi_file=True,
        lookback_months=12,      # lookback 按月
        neutral_band=0.10,       # 信号区间
        # === NEW: 信号时间区间 ===
        signal_start_date=None,
        signal_end_date=None,
        data_start_date=None,
        data_end_date=None
    ):
        # === FIX: 你原来 super 传参错位置，这里纠正 ===
        super().__init__(
            strategy_name=strategy_name,
            col_map=col_map,
            universe_mgr=universe_mgr,
            logger=logger,
            chunk_size=chunk_size,
            multi_file=multi_file,
            # === NEW: 信号时间区间传入基类 ===
            signal_start_date=signal_start_date,
            signal_end_date=signal_end_date,
            data_start_date=data_start_date,
            data_end_date=data_end_date
        )

        self.lookback_months = lookback_months
        self.neutral_band = neutral_band

        # === NEW: data_end_date 默认等于 signal_end_date ===
        if self.data_end_date is None:
            self.data_end_date = self.signal_end_date

        # === NEW: logger record ===
        if self.logger:
            self.logger.log_error(
                f"[TSMOM INIT] signal=[{self.signal_start_date} ~ {self.signal_end_date}], "
                f"data=[{self.data_start_date} ~ {self.data_end_date}], "
                f"lookback_months={self.lookback_months}"
            )

    # =====================================================
    # === NEW: 告诉 BaseSignalEngine 我是月度频率 (M) ===
    # =====================================================
    def get_signal_frequency(self):
        return "M"

    # =====================================================
    # 单股票的月度信号生成
    # =====================================================
    def generate_signal_one_ticker(self, df):

        # === NEW: 数据时间过滤 (data_start / data_end) ===
        if self.data_start_date is not None:
            df = df[df["date"] >= self.data_start_date]
        if self.data_end_date is not None:
            df = df[df["date"] <= self.data_end_date]

        df = df.sort_values("date").copy()

        # ========================
        # ① 按月取最后一天价格
        # ========================
        df_m = (
            df.resample("M", on="date")
              .last()[["close"]]
              .dropna()
        )

        # ========================
        # ② 计算 12 个月动量
        # ========================
        df_m["ret_12m"] = (
            df_m["close"].shift(1) / df_m["close"].shift(self.lookback_months) - 1
        )

        # ========================
        # ③ 生成月度信号
        # ========================
        sig = pd.Series(0, index=df_m.index)

        sig[df_m["ret_12m"] > +self.neutral_band] = 1
        sig[df_m["ret_12m"] < -self.neutral_band] = -1

        sig.index.name = "date"

        # === NEW: 信号窗口过滤 ===
        if self.signal_start_date is not None:
            sig = sig[sig.index >= self.signal_start_date]
        if self.signal_end_date is not None:
            sig = sig[sig.index <= self.signal_end_date]

        return sig
