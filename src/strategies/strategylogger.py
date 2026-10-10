import os
import threading
import queue
import pandas as pd
from datetime import datetime


# ============================================================
#  Async Writer Thread
# ============================================================

class AsyncWriterThread(threading.Thread):
    """
    后台异步写日志线程：
        - 主线程不断 push log 到队列
        - 异步线程后台 flush 到文件
    """

    def __init__(self, log_queue, flush_interval, base_dir):
        super().__init__(daemon=True)
        self.log_queue = log_queue
        self.flush_interval = flush_interval
        self.base_dir = base_dir
        self.running = True

        self.buffer = []         # 临时日志缓存
        self.buffer_count = 0    # 累计条数，用来触发 flush

    def write_to_disk(self, logs):
        if not logs:
            return

        df = pd.DataFrame(logs)

        # 旋转文件夹
        dstr = datetime.utcnow().strftime("%Y-%m-%d")
        day_dir = os.path.join(self.base_dir, dstr)
        os.makedirs(day_dir, exist_ok=True)

        fname = os.path.join(day_dir, "async_logs.csv")

        # 写入模式：追加（append）
        df.to_csv(fname, mode='a', header=not os.path.exists(fname), index=False)

    def run(self):
        while self.running:
            try:
                log = self.log_queue.get(timeout=1)

                if log == "__FLUSH__":
                    # 强制 flush
                    self.write_to_disk(self.buffer)
                    self.buffer = []
                    self.buffer_count = 0
                    continue

                # 写入缓存
                self.buffer.append(log)
                self.buffer_count += 1

                # 超过 interval → flush
                if self.buffer_count >= self.flush_interval:
                    self.write_to_disk(self.buffer)
                    self.buffer = []
                    self.buffer_count = 0

            except queue.Empty:
                # 没有新任务，检查是否需要 flush
                if self.buffer_count > 0:
                    self.write_to_disk(self.buffer)
                    self.buffer = []
                    self.buffer_count = 0

    def stop(self):
        self.running = False
        # Flush before exit
        self.write_to_disk(self.buffer)
        self.buffer = []


# ============================================================
#  StrategyLogger Advanced
# ============================================================

class StrategyLogger:
    """
    StrategyLogger (Advanced Version)
    ---------------------------------
    扩展功能:
        ✔ flush_interval (避免内存暴涨)
        ✔ log rotation (按日生成文件夹)
        ✔ async 异步写入 (不阻塞主策略)
    """

    def __init__(
        self,
        strategy_name="strategy",
        log_dir="./log",
        async_mode=True,
        flush_interval=5000
    ):
        self.strategy_name = strategy_name
        self.log_dir = os.path.join(log_dir, f"strategy_{strategy_name}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Async 模式
        self.async_mode = async_mode
        self.flush_interval = flush_interval

        # 队列: 主线程 push, 异步线程 pull
        self.log_queue = queue.Queue()

        # 启动异步线程
        if self.async_mode:
            self.writer_thread = AsyncWriterThread(
                log_queue=self.log_queue,
                flush_interval=self.flush_interval,
                base_dir=self.log_dir
            )
            self.writer_thread.start()

        # Memory logs（不用于写文件，仅用于快速调试）
        self.signal_logs = []
        self.universe_logs = []
        self.portfolio_logs = []
        self.error_logs = []
        self.feature_logs = {}
        self.raw_signal_logs = {}
        self.filtered_signal_logs = {}

    # ============================================================
    # Unified Log Method → push to queue
    # ============================================================

    def _push_log(self, log_dict, category="generic"):
        """
        push log to async queue
        """
        if self.async_mode:
            log_dict["category"] = category
            self.log_queue.put(log_dict)
        else:
            # fallback to memory logs only
            if category == "signal":
                self.signal_logs.append(log_dict)
            elif category == "portfolio":
                self.portfolio_logs.append(log_dict)
            elif category == "universe":
                self.universe_logs.append(log_dict)
            elif category == "error":
                self.error_logs.append(log_dict)

    # ============================================================
    # Logging Methods
    # ============================================================
    def log_signal(self, date, symbol, signal, action, old_weight, new_weight, close_only=False, cooldown_left=0):
        self._push_log({
            "date": pd.Timestamp(date),
            "symbol": symbol,
            "signal": signal,
            "action": action,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "close_only": close_only,
            "cooldown_left": cooldown_left
        }, category="signal")

    def log_portfolio(self, date, portfolio_dict):
        self._push_log({
            "date": pd.Timestamp(date),
            **portfolio_dict
        }, category="portfolio")

    def log_universe(self, date, symbol, in_universe, close_only=False, has_position=False):
        self._push_log({
            "date": pd.Timestamp(date),
            "symbol": symbol,
            "in_universe": int(in_universe),
            "close_only": int(close_only),
            "has_position": int(has_position)
        }, category="universe")

    def log_error(self, msg):
        self._push_log({"error": msg}, category="error")

    # Feature logs are stored in-memory only (not frequent)
    def log_feature(self, tic, df):
        self.feature_logs[tic] = df.copy()

    def log_raw_signal(self, tic, sig):
        self.raw_signal_logs[tic] = sig.copy()

    def log_filtered_signal(self, df):
        self.filtered_signal_logs["signal_df"] = df.copy()

    # ============================================================
    # Force flush
    # ============================================================
    def flush(self):
        """
        强制刷写日志到文件
        """
        if self.async_mode:
            self.log_queue.put("__FLUSH__")

    # ============================================================
    # Terminate writer thread
    # ============================================================
    def close(self):
        if self.async_mode:
            self.writer_thread.stop()

