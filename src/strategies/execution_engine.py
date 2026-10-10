import pandas as pd
from typing import Dict, Optional
import pandas_market_calendars as mcal
import pandas as pd
import random 
import numpy as np
from strategies.strategylogger import StrategyLogger
from strategies.universe_manager import UniverseManager
from strategies.base_signal import BaseSignalEngine
class ExecutionManager:
    def __init__(
        self,
        universe_mgr,
        max_positions: int = 20,
        max_weight: float = 0.20,
        min_weight: float = 0.05,
        weight_step: float = 0.05,
        allow_short: bool = True,
        gross_leverage: float = 1.0,
        cooling_days: int = 0,
        rebalance_freq: str = "D",  # "D"（日频）或者 "M"（月频）,"W"（周频），未来可扩展
        logger: Optional[object] = None,
        ratio: float = 1.0,           # 最大可用资金比例
        seed: int = 42                # 固定随机性
    ):
        """
        Parameters
        ----------
        universe_mgr : UniverseManager
            已初始化好的 UniverseManager，用于判断股票是否在池子
        max_positions : int
            最大持仓股票数（按 |weight| > 0 计）
        max_weight : float
            单个股票最大绝对权重（如 0.2 = 20%）
        min_weight : float
            单个股票最小非零绝对权重（如 0.05 = 5%），低于此绝对值直接视为 0
        weight_step : float
            每次调整权重的步长（如 0.05）
        allow_short : bool
            是否允许做空（weight < 0）
        gross_leverage : float
            组合总 |weight| 之和上限（如 1.0 = 100%）
        cooling_days : int
            冷静期天数：卖出 / 平仓后需要等待的交易日数，期间禁止重新开仓
        rebalance_freq : str
            调仓频率：
              - "D"：每日调仓
              - "M"：每月调仓（默认用月内第二个交易日）
              - "W"：每周调仓（默认用周内第一个交易日）
              - 可预留扩展 "W"、"intraday" 等
        logger : object or None
            可选日志对象，需实现 log_signal(...)
        """
        self.universe_mgr = universe_mgr
        self.max_positions = int(max_positions)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.weight_step = float(weight_step)
        self.allow_short = allow_short
        self.gross_leverage = float(gross_leverage)
        self.cooling_days = int(cooling_days)
        self.rebalance_freq = rebalance_freq.upper()
        self.logger = logger
        self.ratio = ratio
        random.seed(seed)

        # 当前目标权重：tic_name -> weight (可为负，表示空头)
        self.current_weights: Dict[str, float] = {}

        # 冷静期计数器：tic_name -> 剩余冷静天数
        self.cooldown: Dict[str, int] = {}

        # 上一日日期，用于 close-only 判断
        self.prev_date: Optional[pd.Timestamp] = None

    def set_rebalance_frequency(self, freq: str):
        """
        freq: 'D' / 'W' / 'M'
        """
        self.rebalance_freq = freq.upper()
    # =========================================================
    # 公共主入口：根据 signal_df 生成全历史权重矩阵
    # =========================================================
    def generate_weight_matrix(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        generate a weight matrix (index=date, columns=tic_name, value=weight)
        based on the daily signal_df (index=date, columns=tic_name, value=-1/0/1)

        Parameters
        ----------
        signal_df : pd.DataFrame
            index：日期（DatetimeIndex 或可转为 Timestamp）
            columns：tic_name
            values：-1 / 0 / 1

        Returns
        -------
        weights_df : pd.DataFrame
            index：日期
            columns：tic_name
            values：权重（float）
        """
        dates = sorted(pd.to_datetime(signal_df.index.unique()))
        all_tics = sorted(signal_df.columns.unique())

        records = []


        for dt in dates:
            # get the signal of the day
            row = signal_df.loc[dt]
            if isinstance(row, pd.DataFrame):
                # if there are multiple rows (uncommon), take the first row
                signal_series = row.iloc[0]
            else:
                signal_series = row

            self.step(dt, signal_series)

            # record the weights of the day
            row_weights = {tic: self.current_weights.get(tic, 0.0) for tic in all_tics}
            row_weights["date"] = pd.Timestamp(dt)
            records.append(row_weights)
        #  calculate the target weight matrix

        weights_df = pd.DataFrame(records).set_index("date").sort_index()

        if hasattr(self, "_compute_target_weights"):
            try:
                target_df = self._compute_target_weights(signal_df)
                # align the index and columns (take the intersection to avoid column inconsistency)
                target_df = target_df.reindex_like(weights_df).fillna(0.0)
                # use the target weights to cover the current weights matrix
                weights_df.update(target_df)
                if self.logger:
                    self.logger.log_info("[ExecutionManager] Applied _compute_target_weights successfully.")
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"[ExecutionManager] _compute_target_weights failed: {e}")
                else:
                    print(f"[WARN] _compute_target_weights failed: {e}")

        return weights_df

    # frequency control of rebalance
    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        """
        based on the rebalance_freq, determine if the current date needs to rebalance.
        currently supported:
          - "D"：Day 
          - "M"：Month 
        """
        date = pd.Timestamp(date)

        if self.rebalance_freq == "D":
            return True

        if self.rebalance_freq == "W":
            cal = self.universe_mgr.trading_calendar
            # find the trading days of the week
            week_dates = [d for d in cal
                          if d.isocalendar()[1] == date.isocalendar()[1]
                          and d.year == date.year]
            if not week_dates:
                return False
            week_dates = sorted(week_dates)
            return date.normalize() == week_dates[0].normalize()

        if self.rebalance_freq == "M":
            cal = self.universe_mgr.trading_calendar
            month_dates = [d for d in cal if d.year == date.year and d.month == date.month]
            if not month_dates:
                return False
            month_dates = sorted(month_dates)
            # 第二个交易日 & 月末最后一个交易日
            second_day = month_dates[1] if len(month_dates) >= 2 else month_dates[0]
            last_day = month_dates[-1]
            return date.normalize() in [
                pd.Timestamp(second_day).normalize(),
                pd.Timestamp(last_day).normalize()
            ]

    # daily execution logic: update self.current_weights
    def step(self, date, signal_series: pd.Series):
        """
        single day execution logic:
          1. Decrement cooldown period for each stock.
          2. Check if today is a rebalance day based on the strategy's settings.
          3. If today is a rebalance day:
             - Adjust weights according to Universe membership, signals, close-only rule, and cooldown status
             - Apply portfolio constraints such as max_positions and gross_leverage
        """
        date = pd.Timestamp(date)
        signals = signal_series.to_dict()  # tic -> -1/0/1

        #  decrement the cooldown period for each stock
        for tic in list(self.cooldown.keys()):
            if self.cooldown[tic] > 0:
                self.cooldown[tic] -= 1

        #  update prev_date (for close-only judgment)
        prev_date = self.prev_date
        self.prev_date = date

        # if not a rebalance day, do not change the weights (cooldown period still decrements)
        if not self._should_rebalance(date):
            return

        # the universe of stocks that are allowed to open new positions today
        today_universe = self.universe_mgr.get_universe(date)

        current_positions = {tic for tic, w in self.current_weights.items() if abs(w) > 0}

        # all the stocks that need to be considered: have signal or have positions
        all_tics = sorted(set(signals.keys()) | current_positions)

        new_weights = self.current_weights.copy()

        for tic in all_tics:
            old_w = float(self.current_weights.get(tic, 0.0))
            sig = int(signals.get(tic, 0))

            # cooldown status
            cd = int(self.cooldown.get(tic, 0))
            has_pos = abs(old_w) > 0

            in_uni_today = tic in today_universe
            in_uni_yday = False
            if prev_date is not None:
                in_uni_yday = self.universe_mgr.is_in_universe(tic, prev_date)

            # close-only: yesterday in the pool & today not in the pool & still have positions
            close_only = in_uni_yday and (not in_uni_today) and has_pos

            # if no positions and in cooldown period, do not open new positions (regardless of the signal)
            if (not has_pos) and cd > 0:
                effective_sig = 0
            else:
                effective_sig = sig

            # decide the target direction today (0/+1/-1)
            if effective_sig == 0:
                # signal is 0: immediately close
                new_w = 0.0

            elif close_only:
                # close-only: do not open new positions; only keep the original position
                # if the signal turns to 0, close (already covered above)
                new_w = old_w
                print(f"[CLOSE-ONLY] {tic} keep position {old_w:.2f} (still have positions in the pool)")

            elif effective_sig > 0 and in_uni_today:
                target_sign = 1
                new_w = self._update_weight_one_name(
                    old_weight=old_w, target_sign=target_sign, close_only=False, target_weight=self.max_weight, 
                )

            elif effective_sig < 0 and in_uni_today and self.allow_short:
                target_sign = -1
                new_w = self._update_weight_one_name(
                    old_weight=old_w, target_sign=target_sign, close_only=False,target_weight=self.max_weight, 
                )

            else:
                # not in the universe today & no positions → force 0
                new_w = 0.0

            # update the weight of the day
            new_weights[tic] = new_w

            # if the position changes from non-zero to 0 → start the cooldown period
            if (abs(old_w) > 0) and (abs(new_w) == 0) and (self.cooling_days > 0):
                self.cooldown[tic] = self.cooling_days

            # log
            if self.logger is not None:
                if abs(old_w - new_w) > 1e-8:
                    action = "HOLD"
                    if old_w == 0 and new_w != 0:
                        action = "OPEN_LONG" if new_w > 0 else "OPEN_SHORT"
                    elif old_w != 0 and new_w == 0:
                        action = "CLOSE"
                    elif old_w * new_w < 0:
                        action = "FLIP"
                    else:
                        action = "ADJUST"

                    self.logger.log_signal(
                        date=date,
                        symbol=tic,
                        signal=effective_sig,
                        action=action,
                        old_weight=old_w,
                        new_weight=new_w,
                        close_only=close_only,
                        cooldown_left=self.cooldown.get(tic, 0),
                    )

        # ========= portfolio level constraints =========

        # 1) limit the number of positions
        nz = [(tic, w) for tic, w in new_weights.items() if abs(w) > 0]
        if len(nz) > self.max_positions:
            # sort by |weight| in descending order, keep the top max_positions
            nz_sorted = sorted(nz, key=lambda x: abs(x[1]), reverse=True)
            keep = {tic for tic, _ in nz_sorted[: self.max_positions]}
            for tic, w in nz:
                if tic not in keep:
                    new_weights[tic] = 0.0

        # 2) limit the total leverage (by the sum of absolute values)
        gross = sum(abs(w) for w in new_weights.values())
        if gross > 0 and gross > self.gross_leverage:
            scale = self.gross_leverage / gross
            for tic in new_weights:
                new_weights[tic] *= scale

        self.current_weights = new_weights

    # single stock weight adjustment logic
    def _update_weight_one_name(
        self,
        old_weight: float,
        target_sign: int,
        close_only: bool,
        target_weight: float,
    ) -> float:
        w = float(old_weight)

        # in close-only mode, only reduce the position
        if close_only and target_sign != 0:
            return w  # do not open new positions

        if target_sign == 0:
            # immediately close
            return 0.0

        # open new positions or add positions or flip positions: directly set the target position
        return target_sign * target_weight

    def _apply_min_weight_threshold(self, w: float) -> float:
        """
            when the absolute value is less than min_weight, directly treat it as 0 (close),
            to prevent "dirty positions" like 0.01.
        """
        if abs(w) < self.min_weight:
            return 0.0
        return w
    def _compute_target_weights(self, signal_df: pd.DataFrame) -> pd.DataFrame:
        """
        根据 signal_df 计算目标权重矩阵，遵守以下约束：
        1. 单股最大仓位 ≤ 20%
        2. 单股最小仓位 ≥ 2%
        3. 卖空仓位也计算在总仓位中（取绝对值）
        4. ratio 控制总可用仓位比例（默认 1.0）
        5. 最多 20 只股票（超出则随机抽取）
        6. 新股票只用剩余仓位买入，不调整已有仓位
        7. 每月最后一个交易日做等权 Rebalance
        """
        random.seed(42)
        max_weight = self.max_weight
        min_weight = self.min_weight
        ratio = getattr(self, "ratio", 1.0)
        max_positions = self.max_positions

        dates = sorted(pd.to_datetime(signal_df.index.unique()))
        all_tics = signal_df.columns
        weights_target = pd.DataFrame(index=dates, columns=all_tics, dtype=float).fillna(0.0)

        current_holdings = set()
        last_weights = pd.Series(0.0, index=all_tics)   #


        for date in dates:
            cal = self.universe_mgr.trading_calendar
            month_dates = [d for d in cal if d.year == date.year and d.month == date.month]
            if not month_dates:
                continue
            month_dates = sorted(month_dates)

            # 第二个交易日与月底
            signal_day = month_dates[1] if len(month_dates) >= 2 else month_dates[0]
            last_day = month_dates[-1]
            is_signal_day = date.normalize() == pd.Timestamp(signal_day).normalize()
            is_month_end = date.normalize() == pd.Timestamp(last_day).normalize()

            # --- 每月第二个交易日：根据信号建仓 ---
            if is_signal_day:
                row = signal_df.loc[date] if date in signal_df.index else None
                if row is None:
                    weights_target.loc[date] = last_weights
                    continue

                active_tics = [tic for tic, sig in row.items() if sig != 0]
                if not active_tics:
                    weights_target.loc[date] = last_weights
                    continue

                if len(active_tics) > max_positions:
                    active_tics = random.sample(active_tics, max_positions)

                equal_w = min(max_weight, max(min_weight, ratio / len(active_tics)))
                new_weights = pd.Series(0.0, index=all_tics)
                for tic in active_tics:
                    new_weights[tic] = row[tic] * equal_w

                last_weights = new_weights.copy()
                weights_target.loc[date] = last_weights
                current_holdings = set(active_tics)

            # --- 月底 Rebalance ---
            elif is_month_end and len(current_holdings) > 0:
                equal_w = min(max_weight, max(min_weight, ratio / len(current_holdings)))
                new_weights = pd.Series(0.0, index=all_tics)
                for tic in current_holdings:
                    sig = 1 if last_weights.get(tic, 0) >= 0 else -1
                    new_weights[tic] = sig * equal_w

                last_weights = new_weights.copy()
                weights_target.loc[date] = last_weights

            # --- 其他日期：延续上次仓位 ---
            else:
                weights_target.loc[date] = last_weights

        # 最后再清理
        weights_target = weights_target.fillna(0.0)
        return weights_target