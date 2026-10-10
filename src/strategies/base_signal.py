import pandas_market_calendars as mcal
import pandas as pd
import random 
import numpy as np
import os
from typing import Dict, Optional, Iterable
from .universe_manager import UniverseManager
from .strategylogger import StrategyLogger

class BaseSignalEngine:
    """
    BaseSignalEngine (Rebuilt Clean Version)
    ---------------------------------------
    负责：
        ✓ 多文件/单文件读取
        ✓ chunk 加载
        ✓ 字段映射 col_map
        ✓ 为每个 tic 调用 generate_signal_one_ticker()
        ✓ 与 Universe / Position 做基本过滤
    """

    def __init__(
        self,
        strategy_name="default",
        col_map=None,
        universe_mgr=None,
        logger=None,
        chunk_size=200000,
        multi_file=True,
        #signal are generated in this period
        signal_start_date=None,
        signal_end_date=None,

        # data are read in this period
        data_start_date=None,
        data_end_date=None
    ):
        self.strategy_name = strategy_name
        self.universe_mgr = universe_mgr
        self.chunk_size = chunk_size
        self.multi_file = multi_file
        #  time parameters are parsed in advance
        self.signal_start_date = pd.to_datetime(signal_start_date) if signal_start_date else None
        self.signal_end_date   = pd.to_datetime(signal_end_date) if signal_end_date else None
        self.data_start_date   = pd.to_datetime(data_start_date) if data_start_date else None
        self.data_end_date     = pd.to_datetime(data_end_date) if data_end_date else None

        # 统一内部列名
        self.col_map = col_map or {
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "tic": "tic"
        }

        self.logger = logger or StrategyLogger(strategy_name)

    # ===============================================================
    # 多文件模式：每个股票一个 CSV
    # ===============================================================
    def load_price_data_multi_file(self, folder, tics):
        price_dict = {}

        for tic in tics:
            path = os.path.join(folder, f"{tic}_daily.csv")
            if not os.path.exists(path):
                self.logger.log_error(f"[WARN] Missing file: {path}")
                continue

            print(f"[READ] {path} ...")

            chunks = []
            for chunk in pd.read_csv(path, chunksize=self.chunk_size):

                # rename_map：内部名 ← 文件名
                rename_map = {
                    file_col: internal_col
                    for internal_col, file_col in self.col_map.items()
                    if file_col in chunk.columns
                }
                chunk = chunk.rename(columns=rename_map)

                # 强制加入 tic
                chunk["tic"] = tic

                # 统一 datetime
                if "datetime" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["datetime"])
                    chunk.drop(columns=["datetime"], inplace=True)   # === NEW === 删除冗余列
                elif "date" in chunk.columns:
                    chunk["date"] = pd.to_datetime(chunk["date"])
                else:
                    raise ValueError(f"{path} 缺少 date/datetime 列")
                # === data time filter ===
                if self.data_start_date is not None:
                    chunk = chunk[chunk["date"] >= self.data_start_date]
                if self.data_end_date is not None:
                    chunk = chunk[chunk["date"] <= self.data_end_date]

                # === skip empty chunk ===
                if chunk.empty:
                    continue

                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            df = df.sort_values("date")

            price_dict[tic] = df
            print(f"      ✓ loaded {len(df)} rows for {tic}")

        return price_dict

    # ===============================================================
    # 单文件模式（很少使用）
    # ===============================================================
    def load_price_data_single_file(self, filepath):
        print(f"[READ] Big file in chunks: {filepath}")

        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
            rename_map = {
                file_col: internal_col
                for internal_col, file_col in self.col_map.items()
                if file_col in chunk.columns
            }
            chunk = chunk.rename(columns=rename_map)

            if "datetime" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["datetime"])
                chunk.drop(columns=["datetime"], inplace=True)   # === NEW === 删除冗余列

            elif "date" in chunk.columns:
                chunk["date"] = pd.to_datetime(chunk["date"])
            else:
                raise ValueError("文件缺少 date/datetime 列")

            # === data time filter ===
            if self.data_start_date is not None:
                chunk = chunk[chunk["date"] >= self.data_start_date]
            if self.data_end_date is not None:
                chunk = chunk[chunk["date"] <= self.data_end_date]

            # === skip empty chunk ===
            if chunk.empty:
                continue


            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        df = df.sort_values(["tic", "date"])
        return df
        
    # expand signal to daily, weekly, monthly

    def _expand_signal_to_daily(self, signal_df):
        freq = self.get_signal_frequency()

        # 需要 trading calendar，由 UniverseManager 提供
        cal = pd.DatetimeIndex(self.universe_mgr.trading_calendar)

        # -------- 日频：不需要扩展 --------
        if freq == "D":
            return signal_df.reindex(cal).fillna(0)

        # -------- 周频：覆盖至下次周信号 --------
        if freq == "W":
            idx = signal_df.index
            next_idx = list(idx[1:]) + [idx[-1] + pd.Timedelta(days=7)]
            
            records = []
            for start, end in zip(idx, next_idx):
                mask = (cal >= start) & (cal < end)
                for d in cal[mask]:
                    s = signal_df.loc[start]
                    records.append( (d, s) )

            out = pd.DataFrame({"date": [r[0] for r in records]}).set_index("date")
            for col in signal_df.columns:
                out[col] = [r[1][col] for r in records]
            return out

        # -------- 月频：覆盖至下次月信号 --------
        if freq == "M":
            idx = signal_df.index
            next_idx = list(idx[1:]) + [idx[-1] + pd.offsets.MonthEnd(1)]

            records = []
            for start, end in zip(idx, next_idx):
                mask = (cal >= start) & (cal < end)
                for d in cal[mask]:
                    s = signal_df.loc[start]
                    records.append((d, s))

            out = pd.DataFrame({"date": [r[0] for r in records]}).set_index("date")
            for col in signal_df.columns:
                out[col] = [r[1][col] for r in records]
            return out

        raise ValueError(f"Unsupported signal freq: {freq}")

    # ===============================================================
    # 主方法：生成 signal_df（date × tic）
    # ===============================================================
    def compute_signals(self, price_source, tics, position_df=None):

        # ---- Step 1: 读入 ----
        if self.multi_file:
            price_dict = self.load_price_data_multi_file(price_source, tics)
            full_df = pd.concat(price_dict.values(), ignore_index=True)
        else:
            full_df = self.load_price_data_single_file(price_source)
        # === data time filter ===
        if self.data_start_date is not None:
            full_df = full_df[full_df["date"] >= self.data_start_date]
        if self.data_end_date is not None:
            full_df = full_df[full_df["date"] <= self.data_end_date]

        # ---- Step 2: 当前持仓 ----
        positions = {}
        if position_df is not None and len(position_df) > 0:
            positions = dict(zip(position_df["tic"], position_df["weight"]))

        # ---- Step 3: 为每只股票生成信号 ----
        signal_list = []
        for tic in tics:
            sub = full_df[full_df["tic"] == tic]
            if sub.empty:
                continue

            sig = self.generate_signal_one_ticker(sub)

            # === NEW: 信号时间过滤 ===
            if self.signal_start_date is not None:
                sig = sig[sig.index >= self.signal_start_date]
            if self.signal_end_date is not None:
                sig = sig[sig.index <= self.signal_end_date]
            sig.name = tic
            signal_list.append(sig)
            self.logger.log_raw_signal(tic, sig)

        signal_df = pd.concat(signal_list, axis=1).fillna(0)
        signal_df.to_csv("./log/signal_df.csv")
        final_df = self._expand_signal_to_daily(signal_df)
        # =========================================================
        # filter daily signals by universe 
        # =========================================================
        if self.universe_mgr is not None:
            univ_mgr = self.universe_mgr

            # get all trading dates
            dates = final_df.index

            # get all columns (tic)
            all_tics = final_df.columns

            # build a mask for each date, set the signal of stocks not in the universe to 0
            mask_matrix = []
            for d in dates:
                todays_universe = univ_mgr.get_universe(d)
                mask = all_tics.isin(todays_universe)  # True=keep，False=0
                
                if hasattr(mask, 'values'):
                    mask_matrix.append(mask.values)
                else:
                    # 如果 mask 已经是 numpy array，直接 append
                    mask_matrix.append(mask)
            mask_matrix = np.vstack(mask_matrix)  # shape=(n_dates, n_tics)

            # use mask to filter the signal of stocks not in the universe
            final_df = final_df.where(mask_matrix, 0)
        return final_df
    def get_signal_frequency(self) -> str:
        """
        返回策略生成信号的频率：
            "D": 日度
            "W": 周度
            "M": 月度
        子类应该覆盖。
        """
        return "D"  # 默认日度
