import pandas_market_calendars as mcal
import pandas as pd
import random 
import numpy as np
from strategies.strategylogger import StrategyLogger

class UniverseManager:
    """
    UniverseManager (Final Version, Logger-compatible)
    --------------------------------------------------
    * 自动从季度选股生成日度股票池
    * 不关心持仓
    * 只负责 in_universe 的判断
    * 事件日志兼容增强版 StrategyLogger
    """

    def __init__(
        self,
        stock_selection_df,
        col_map,
        trading_calendar,
        logger=None,
        backtest_start=None,
        backtest_end=None,
    ):
        self.logger = logger
        self.trading_calendar = pd.DatetimeIndex(sorted(trading_calendar))

        # === save backtest start and end ===
        self.backtest_start = pd.to_datetime(backtest_start) if backtest_start else None
        self.backtest_end   = pd.to_datetime(backtest_end) if backtest_end else None

        # -----------------------------
        # map column names
        # -----------------------------
        df = stock_selection_df.copy()
        df = df.rename(columns={
            col_map["tic_name"]: "tic_name",
            col_map["trade_date"]: "trade_date"
        })
        df["trade_date"] = pd.to_datetime(df["trade_date"])

        # === select backtest period ===
        if self.backtest_start is not None:
            df = df[df["trade_date"] >= self.backtest_start]

        if self.backtest_end is not None:
            df = df[df["trade_date"] <= self.backtest_end]


        # -----------------------------
        # build daily universe_df
        # -----------------------------
        self.universe_df = self._build_universe(df)

        # -----------------------------
        # build fast index
        # -----------------------------
        self.universe_map = self._build_fast_index(self.universe_df)

        # save yesterday's universe, for IN / OUT judgment
        self.prev_universe = set()
        
        # === log feedback ===
        if self.logger:
            self.logger.log_error(
                f"[UniverseManager] Loaded {len(self.universe_df)} daily rows, "
                f"backtest=[{self.backtest_start} ~ {self.backtest_end}]"
            )

    # ============================================================
    # Internal Helpers
    # ============================================================

    def _next_trade_date(self, date):
        date = pd.Timestamp(date)
        pos = self.trading_calendar.searchsorted(date, side="right")
        if pos >= len(self.trading_calendar):
            return None
        return self.trading_calendar[pos]

    def _build_universe(self, df):
        df = df.copy()
        df["activate_date"] = df["trade_date"].apply(self._next_trade_date)
        df = df.dropna(subset=["activate_date"])

        df = df.sort_values(["trade_date", "tic_name"])
        quarters = df.groupby("trade_date")

        trade_dates = sorted(df["trade_date"].unique())
        activate_dates = [self._next_trade_date(d) for d in trade_dates]

        deactivate_map = {}
        for i in range(len(activate_dates)-1):
            deactivate_map[activate_dates[i]] = activate_dates[i+1]

        max_date = self.trading_calendar.max() + pd.Timedelta(days=1)
        deactivate_map[activate_dates[-1]] = max_date

        # ⬇ build daily universe
        records = []

        for trade_date, group in quarters:
            act_date = self._next_trade_date(trade_date)
            deact_date = deactivate_map[act_date]

            tics = group["tic_name"].tolist()
            mask = (self.trading_calendar >= act_date) & (self.trading_calendar < deact_date)
            active_days = self.trading_calendar[mask]

            for d in active_days:
                for tic in tics:
                    records.append({
                        "date": d,
                        "tic_name": tic,
                        "in_universe": 1
                    })

        universe_df = (
            pd.DataFrame(records)
              .drop_duplicates(["date", "tic_name"])
              .sort_values(["date", "tic_name"])
        )
        return universe_df

    def _build_fast_index(self, universe_df):
        fast = {}
        for date, grp in universe_df.groupby("date"):
            fast[pd.Timestamp(date)] = set(grp["tic_name"].tolist())
        return fast

    # ============================================================
    # Public API
    # ============================================================

    def is_in_universe(self, tic_name, date):
        date = pd.Timestamp(date)
        tics = self.universe_map.get(date)
        if tics is None:
            return False
        return tic_name in tics

    def get_universe(self, date):
        date = pd.Timestamp(date)
        return self.universe_map.get(date, set())

    # ============================================================
    # Universe Logging (IN / OUT)
    # ============================================================

    def log_universe_events_for_date(self, date):
        """
        仅记录股票池的 IN / OUT（Execution 决定 close-only）
        """
        if self.logger is None:
            return

        date = pd.Timestamp(date)
        today_u = self.get_universe(date)

        added = today_u - self.prev_universe
        removed = self.prev_universe - today_u

        # --- modified: use logger's compatible signature ---
        for tic in sorted(added):
            self.logger.log_universe(
                date=date,
                symbol=tic,
                in_universe=1,
                close_only=False,
                has_position=False
            )

        for tic in sorted(removed):
            self.logger.log_universe(
                date=date,
                symbol=tic,
                in_universe=0,
                close_only=False,
                has_position=False
            )
        # --- end of modification ---

        self.prev_universe = today_u.copy()
