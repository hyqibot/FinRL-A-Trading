# -*- coding: utf-8 -*-
"""
FinRL Trading Dashboard（A股主界面）
================================

Streamlit 入口。侧边栏各页已按 A 股口径改造；「A股实盘」仍走 ashare_dashboard。
美股 Alpaca / WRDS 等模块软失败，不影响 A 股页加载。
"""

import streamlit as st
import logging
from pathlib import Path
import sys

_WEB = Path(__file__).resolve().parent
_SRC = _WEB.parent
_ROOT = _SRC.parent
for _p in (str(_ROOT), str(_SRC), str(_WEB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _load(mod, names):
    import importlib
    try:
        m = importlib.import_module(mod)
        return {n: getattr(m, n) for n in names}
    except Exception as exc:
        logging.getLogger(__name__).warning("skip %s: %s", mod, exc)
        return {n: None for n in names}


_loaded = {}
for _mod, _names in [
    ("config.settings", ("get_config",)),
    ("data.data_store", ("get_data_store",)),
    ("strategies.base_strategy", ("create_strategy", "StrategyConfig")),
    ("backtest.backtest_engine", ("BacktestEngine", "BacktestConfig")),
    ("trading.alpaca_manager", ("create_alpaca_account_from_env",)),
    ("trading.trade_executor", ("TradeExecutor", "ExecutionConfig")),
    ("utils.logging_utils", ("setup_logging",)),
]:
    _loaded.update(_load(_mod, _names))

get_config = _loaded["get_config"]
get_data_store = _loaded["get_data_store"]
create_strategy = _loaded["create_strategy"]
StrategyConfig = _loaded["StrategyConfig"]
BacktestEngine = _loaded["BacktestEngine"]
BacktestConfig = _loaded["BacktestConfig"]
create_alpaca_account_from_env = _loaded["create_alpaca_account_from_env"]
TradeExecutor = _loaded["TradeExecutor"]
ExecutionConfig = _loaded["ExecutionConfig"]
setup_logging = _loaded["setup_logging"]

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="FinRL A股交易面板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not st.session_state.get("config") and get_config:
    try:
        st.session_state.config = get_config()
    except Exception:
        st.session_state.config = None
if not st.session_state.get("data_store") and get_data_store:
    try:
        st.session_state.data_store = get_data_store()
    except Exception:
        st.session_state.data_store = None

# page helpers
try:
    from web.ashare_pages import (
        display_quick_stats_ashare,
        show_overview_ashare,
        show_data_management_ashare,
        show_strategy_backtesting_ashare,
        show_portfolio_analysis_ashare,
        show_settings_ashare,
    )
except ImportError:
    from ashare_pages import (
        display_quick_stats_ashare,
        show_overview_ashare,
        show_data_management_ashare,
        show_strategy_backtesting_ashare,
        show_portfolio_analysis_ashare,
        show_settings_ashare,
    )


PAGE_OVERVIEW = "总览"
PAGE_DATA = "数据管理"
PAGE_BT = "策略回测"
PAGE_LIVE = "A股实盘"
PAGE_PERF = "绩效分析"
PAGE_SETTINGS = "设置"


def main():
    st.title("FinRL A股交易面板")
    st.markdown("通达信行情 · akshare 基本面 · 策略选股定权 · easytrader 通道（实盘需确认）")

    with st.sidebar:
        st.header("导航")
        page = st.selectbox(
            "选择页面",
            [PAGE_OVERVIEW, PAGE_DATA, PAGE_BT, PAGE_LIVE, PAGE_PERF, PAGE_SETTINGS],
        )
        st.divider()
        display_quick_stats_ashare()

    if page == PAGE_OVERVIEW:
        show_overview_ashare()
    elif page == PAGE_DATA:
        show_data_management_ashare()
    elif page == PAGE_BT:
        show_strategy_backtesting_ashare()
    elif page == PAGE_LIVE:
        show_live_trading()
    elif page == PAGE_PERF:
        show_portfolio_analysis_ashare()
    elif page == PAGE_SETTINGS:
        show_settings_ashare()


def show_live_trading():
    """A股实盘：委托 ashare_dashboard，不在此改财联社逻辑。"""
    try:
        from .ashare_dashboard import show_ashare_live
    except ImportError:
        from ashare_dashboard import show_ashare_live
    show_ashare_live()


if __name__ == "__main__":
    main()
