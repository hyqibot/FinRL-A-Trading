# -*- coding: utf-8 -*-
"""A股实盘页：策略用例 + 全策略能力 + 财联社情绪 + 一键启动。默认预览，实盘需勾选确认。"""
import logging
import os
from pathlib import Path
import sys
import time

import pandas as pd
import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("ashare_dashboard")

# FinRL rolling DRL long-run status (session + on-disk live_progress.*)
_FINRL_LIVE_TXT = os.path.join(REPO_ROOT, "artifacts", "ashare_drl", "live_progress.txt")
_FINRL_LIVE_JSON = os.path.join(REPO_ROOT, "artifacts", "ashare_drl", "live_progress.json")
_FINRL_PROGRESS_JSON = os.path.join(REPO_ROOT, "artifacts", "ashare_drl", "progress.json")


def _read_finrl_live_progress(output_dir=None):
    """Read live_progress.txt/json written by ashare_finrl_drl (refresh-safe)."""
    root = output_dir or os.path.join(REPO_ROOT, "artifacts", "ashare_drl")
    txt = os.path.join(root, "live_progress.txt")
    js = os.path.join(root, "live_progress.json")
    data = {}
    if os.path.isfile(js):
        try:
            import json
            with open(js, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
    raw = ""
    if os.path.isfile(txt):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                raw = f.read()
        except Exception:
            raw = ""
    if not data and raw:
        data = {"last_msg": raw.strip().splitlines()[-1] if raw.strip() else "", "raw": raw}
    elif raw:
        data.setdefault("raw", raw)
    return data


def _show_finrl_run_status(force_output_dir=None):
    """Show in-progress / last live progress when form was not re-submitted."""
    running = bool(st.session_state.get("ashare_finrl_running"))
    last_msg = st.session_state.get("ashare_finrl_last_msg") or ""
    out_dir = (
        force_output_dir
        or st.session_state.get("ashare_finrl_output_dir")
        or os.path.join(REPO_ROOT, "artifacts", "ashare_drl")
    )
    live = _read_finrl_live_progress(out_dir)
    disk_msg = (live or {}).get("last_msg") or ""
    disk_status = (live or {}).get("status") or ""
    if not running and disk_status not in ("running", "starting") and not last_msg and not disk_msg:
        return False
    st.warning("训练进行中（或上次未完整结束的进度）")
    st.info(last_msg or disk_msg or "正在运行，请稍候…")
    msgs = st.session_state.get("ashare_finrl_msgs") or []
    if msgs:
        st.code("\n".join(msgs[-20:]), language="text")
    elif live.get("raw"):
        st.code(str(live.get("raw"))[-4000:], language="text")
    w_idx = live.get("window_idx") or st.session_state.get("ashare_finrl_window_idx")
    w_tot = live.get("window_total") or st.session_state.get("ashare_finrl_window_total")
    if w_idx and w_tot:
        try:
            st.progress(min(1.0, float(w_idx) / float(w_tot)))
            st.caption("窗口 %s / %s · 算法=%s · 阶段=%s" % (
                w_idx, w_tot, live.get("algo") or "-", live.get("phase") or "-",
            ))
        except Exception:
            pass
    st.caption(
        "产物目录：%s ｜ live_progress.txt / progress.json 可刷新查看。"
        "长跑建议用命令行：python -m src.ashare_finrl_cli --resume" % out_dir
    )
    if st.button("清除「训练进行中」标记", key="ashare_finrl_clear_running"):
        st.session_state["ashare_finrl_running"] = False
        st.session_state["ashare_finrl_last_msg"] = ""
        st.session_state["ashare_finrl_msgs"] = []
        st.rerun()
    return True


def _make_ui_progress(status_slot, log_slot, bar_slot=None):
    """Progress callback: update Streamlit widgets + session_state."""
    import re

    def _cb(msg):
        msg = str(msg)
        st.session_state["ashare_finrl_running"] = True
        st.session_state["ashare_finrl_last_msg"] = msg
        hist = list(st.session_state.get("ashare_finrl_msgs") or [])
        hist.append(msg)
        st.session_state["ashare_finrl_msgs"] = hist[-40:]
        try:
            status_slot.info(msg)
        except Exception:
            pass
        try:
            log_slot.code("\n".join(st.session_state["ashare_finrl_msgs"][-20:]), language="text")
        except Exception:
            pass
        m = re.search(r"window\s+(\d+)/(\d+)", msg)
        if m:
            try:
                i, n = int(m.group(1)), int(m.group(2))
                st.session_state["ashare_finrl_window_idx"] = i
                st.session_state["ashare_finrl_window_total"] = n
                if bar_slot is not None and n > 0:
                    bar_slot.progress(min(1.0, float(i) / float(n)), text="窗口 %d/%d" % (i, n))
            except Exception:
                pass

    return _cb



def _default_tdx_dir():
    from src.config.ashare_tdx import get_tdx_dir
    return get_tdx_dir()


def _persist_tdx_dir(path):
    from src.config.ashare_tdx import save_tdx_dir
    return save_tdx_dir(path)


def _default_start_year():
    from src.config.ashare_tdx import get_start_year
    return get_start_year()


def _persist_start_year(year):
    from src.config.ashare_tdx import save_start_year
    return save_start_year(year)


def _cache_path_for_year(cache, start_year, need_ind=False):
    """Append _y{year} when start_year != 2018 so caches do not mix."""
    from src.config.ashare_tdx import DEFAULT_START_YEAR

    cache_path = str(cache) if cache else cache
    if cache_path and int(start_year) != int(DEFAULT_START_YEAR):
        suffix = "_y%d" % int(start_year)
        if not str(cache_path).endswith(suffix):
            cache_path = str(cache_path) + suffix
    if need_ind and cache_path and not str(cache_path).endswith("_ind"):
        cache_path = str(cache_path) + "_ind"
    return cache_path


def _load_codes(universe_path, manual):
    if manual and manual.strip():
        return [c.strip().zfill(6) for c in manual.replace("，", ",").replace(" ", ",").split(",") if c.strip()]
    codes = []
    with open(universe_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(",")[0].strip()
            if line and line[0].isdigit():
                codes.append(line.zfill(6))
    return codes


def _clean_weights(weights, min_weight):
    w = weights.copy()
    if "tic" not in w.columns and "gvkey" in w.columns:
        w["tic"] = w["gvkey"].astype(str).str.zfill(6)
    if "tic" in w.columns:
        w["tic"] = w["tic"].astype(str).str.zfill(6)
    if "gvkey" in w.columns:
        w["gvkey"] = w["gvkey"].astype(str).str.zfill(6)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce").fillna(0.0)
    if "tic" in w.columns:
        w = w.sort_values("weight", ascending=False).drop_duplicates(subset="tic")
    w = w[w["weight"] > min_weight].copy()
    total = w["weight"].sum()
    if total > 0:
        w["weight"] = w["weight"] / total
    return w.sort_values("weight", ascending=False).reset_index(drop=True)


def run_pipeline(
    codes,
    cache,
    min_weight,
    out_csv,
    live,
    broker,
    cash_override,
    progress,
    tdx_dir=None,
    strategy_family="ml",
    strategy_params=None,
    start_year=None,
):
    from src.data.ashare_data_adapter import build_dataset
    from src.strategies.ashare_strategy_runner import generate_weights, family_needs_industry
    from ashare_executor import AShareExecutor

    strategy_params = dict(strategy_params or {})
    need_ind = family_needs_industry(strategy_family)
    from src.config.ashare_tdx import get_start_year, DEFAULT_START_YEAR

    if start_year is None:
        start_year = get_start_year()
    start_year = int(start_year)
    cache_path = _cache_path_for_year(cache, start_year, need_ind=need_ind)

    if int(start_year) != int(DEFAULT_START_YEAR):
        progress(
            "正在构建数据集（起始年=%d；缓存前缀已加 _y%d，与默认2018缓存分开）"
            % (start_year, start_year)
        )
    else:
        progress("正在构建数据集（起始年=%d；有缓存会直接读；要行业时用带行业缓存）" % start_year)
    ds = build_dataset(
        codes,
        start_year=start_year,
        with_industry=need_ind,
        cache_path=cache_path,
        tdx_dir=tdx_dir,
    )
    if ds["fundamentals"].empty and strategy_family not in ("drl", "finrl_rolling_drl", "adaptive_rotation", "min_variance", "mean_variance", "market_cap", "equal_weight"):
        raise RuntimeError("基本面为空，无法选股")
    if ds["prices"].empty and strategy_family in ("drl", "finrl_rolling_drl", "adaptive_rotation", "min_variance", "mean_variance", "ml_then_drl", "market_cap"):
        raise RuntimeError("价格为空，无法做组合优化")

    progress("正在生成权重（策略：%s）" % strategy_family)
    # Forward progress into strategy params so finrl_rolling_drl can stream logs to UI.
    strategy_params = dict(strategy_params or {})
    strategy_params["progress"] = progress
    weights, meta = generate_weights(strategy_family, ds, strategy_params)
    weights = _clean_weights(weights, min_weight)

    weights.to_csv(out_csv, index=False, encoding="utf-8-sig")

    progress("正在计算委托")
    ex = AShareExecutor(dry_run=not live, cash_override=cash_override)
    if live:
        progress("正在连接券商")
        ex.connect_broker(broker)
    orders, total_value = ex.calculate_orders(ex.load_target_weights(out_csv))
    positions = ex.get_current_positions() if live else {}
    if live and orders:
        progress("正在提交委托")
        ex.execute(out_csv, dry_run=False)
    return {
        "weights": weights,
        "orders": orders,
        "total_value": total_value,
        "positions": positions,
        "submitted": bool(live and orders),
        "meta": meta,
    }


UNIVERSE_PRESETS = {
    "沪深300": (
        os.path.join(REPO_ROOT, "hs300.txt"),
        os.path.join(REPO_ROOT, "data", "cache_hs300"),
    ),
    "全部A股": (
        os.path.join(REPO_ROOT, "all_a.txt"),
        os.path.join(REPO_ROOT, "data", "cache_all_a"),
    ),
}


def _six_code(raw):
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except Exception:
        return None
    text = str(raw).strip()
    if text.lower() in ("nan", "none", ""):
        return None
    if text.endswith(".0"):
        text = text[:-2]
    head = text.split(",")[0].strip().split(".")[0].strip()
    if head.isdigit():
        if len(head) > 6:
            return None
        return head.zfill(6)
    digits = "".join(ch for ch in head if ch.isdigit())
    if len(digits) == 6:
        return digits
    return None


def _column_codes(df, names):
    col = None
    for name in names:
        if name in df.columns:
            col = name
            break
    if col is None:
        for name in df.columns:
            label = str(name)
            if "代码" in label and "名" not in label:
                col = name
                break
    if col is None:
        raise RuntimeError("返回结果里没有代码列: %s" % ",".join(str(c) for c in df.columns))
    codes = []
    seen = set()
    for raw_code in df[col].tolist():
        code = _six_code(raw_code)
        if not code or code in seen:
            continue
        seen.add(code)
        codes.append(code)
    if not codes:
        raise RuntimeError("接口没有返回股票代码")
    return codes


def _with_retries(fn, tries=3, delay=1.5, label=""):
    last = None
    wait = delay
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            logger.warning("retry %s (%d/%d): %s", label, i + 1, tries, str(e)[:160])
            if i + 1 < tries:
                time.sleep(wait)
                wait *= 1.5
    raise last


def fetch_hs300_codes():
    import akshare as ak
    errors = []

    def from_sina_cons():
        df = ak.index_stock_cons(symbol="000300")
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("index_stock_cons(000300) empty")
        return _column_codes(df, ("品种代码", "成分券代码"))

    def from_csindex():
        df = ak.index_stock_cons_csindex(symbol="000300")
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("index_stock_cons_csindex(000300) empty")
        return _column_codes(df, ("成分券代码", "品种代码", "成分股代码"))

    def from_sina_node():
        df = ak.index_stock_cons_sina(symbol="000300")
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("index_stock_cons_sina(000300) empty")
        return _column_codes(df, ("code", "代码", "品种代码", "成分券代码"))

    sources = (
        ("新浪成分", from_sina_cons),
        ("中证成分", from_csindex),
        ("新浪节点成分", from_sina_node),
    )
    for label, fn in sources:
        try:
            codes = _with_retries(fn, tries=3, delay=1.5, label="hs300/%s" % label)
            if not codes:
                raise RuntimeError("empty")
            return codes, label
        except Exception as e:
            errors.append("%s: %s" % (label, e))
    raise RuntimeError("HS300 all sources failed: " + " | ".join(errors))


def fetch_all_a_codes():
    import akshare as ak
    errors = []

    def from_em_spot():
        df = ak.stock_zh_a_spot_em()
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("stock_zh_a_spot_em empty")
        return _column_codes(df, ("代码",))

    def from_exchange_list():
        df = ak.stock_info_a_code_name()
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("stock_info_a_code_name empty")
        return _column_codes(df, ("证券代码", "代码", "code"))

    def from_sina_spot():
        df = ak.stock_zh_a_spot()
        if df is None or getattr(df, "empty", True):
            raise RuntimeError("stock_zh_a_spot empty")
        return _column_codes(df, ("代码", "code"))

    sources = (
        ("东财现货", from_em_spot),
        ("交易所代码表", from_exchange_list),
        ("新浪现货", from_sina_spot),
    )
    for label, fn in sources:
        try:
            codes = _with_retries(fn, tries=3, delay=1.5, label="all_a/%s" % label)
            if not codes:
                raise RuntimeError("empty")
            return codes, label
        except Exception as e:
            errors.append("%s: %s" % (label, e))
    raise RuntimeError(
        "全部A股 akshare 源均失败（东财现货 / stock_info_a_code_name / 新浪现货）: "
        + " | ".join(errors)
    )


def write_universe_file(path, codes):
    if not codes:
        raise RuntimeError("拒绝写入空股票池")
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    payload = (chr(13) + chr(10)).join(codes).encode("utf-8")
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(payload)
    os.replace(tmp, path)
    return len(codes)


def _render_cls_results_block():
    """Show latest per-ticker CLS news results from disk (survives Streamlit rerun)."""
    from src.data.cls_news import get_cls_config, load_latest_news_df, load_latest_summary

    err = st.session_state.get("cls_last_error")
    if err:
        st.error("新闻拉取失败：%s" % err)

    flash = st.session_state.get("cls_last_flash")
    if flash:
        level = flash.get("level") or "info"
        msg = flash.get("message") or ""
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.info(msg)

    summary = load_latest_summary() or st.session_state.get("cls_last_summary")
    news_df = load_latest_news_df()
    cfg = get_cls_config()
    latest_csv = Path(cfg["cache_dir"]) / "cls_news_latest.csv"
    summary_json = Path(cfg["cache_dir"]) / "cls_sentiment_latest.json"

    if summary is None and (news_df is None or news_df.empty):
        st.info("还没有新闻结果。在上方填写股票代码与日期后，点「拉取新闻并分析情绪」。")
        return news_df, summary

    st.markdown("#### 最近一次新闻情绪结果")
    if summary:
        n = int(summary.get("n", 0) or 0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("新闻条数", n)
        m2.metric("正/中/负", "%s/%s/%s" % (
            summary.get("positive", 0),
            summary.get("neutral", 0),
            summary.get("negative", 0),
        ))
        m3.metric("标注方式", str(summary.get("sentiment_backend") or "-"))
        tickers = summary.get("tickers") or []
        m4.metric("请求股票数", len(tickers) if isinstance(tickers, list) else "-")
        st.caption(
            "来源=%s | 区间=%s~%s | 文件=%s | 摘要=%s"
            % (
                summary.get("source", ""),
                summary.get("start_date", ""),
                summary.get("end_date", ""),
                latest_csv,
                summary_json,
            )
        )
        per = summary.get("per_ticker_counts") or {}
        if isinstance(per, dict) and per:
            rows = [{"代码": k, "命中条数": int(v)} for k, v in sorted(per.items(), key=lambda x: (-int(x[1]), x[0]))]
            st.markdown("##### 各股命中条数")
            st.dataframe(pd.DataFrame(rows), width="stretch", height=min(280, 48 + 28 * max(len(rows), 1)))
        if n <= 0:
            st.warning("拉取完成但没有命中新闻（电报在日期内未出现该股全称或代码）。可加大翻页数或放宽日期。")
    else:
        st.warning("有新闻 CSV 但缺少摘要 JSON，仅展示表格。")

    if news_df is not None and not news_df.empty:
        show_cols = [
            c for c in [
                "ticker", "ticker_name", "time", "title", "sentiment",
                "sentiment_confidence", "sentiment_model", "content", "url",
            ]
            if c in news_df.columns
        ]
        preview = news_df[show_cols].head(50) if show_cols else news_df.head(50)
        st.dataframe(preview, width="stretch", height=420)
        with st.expander("查看全部 %d 条" % len(news_df), expanded=False):
            st.dataframe(news_df[show_cols] if show_cols else news_df, width="stretch", height=520)
        try:
            csv_bytes = latest_csv.read_bytes() if latest_csv.is_file() else news_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载最新 CSV",
                data=csv_bytes,
                file_name="cls_news_latest.csv",
                mime="text/csv",
                key="cls_download_latest_csv",
            )
        except Exception as e:
            st.caption("下载按钮不可用：%s" % e)
    return news_df, summary


def _default_cls_tickers(universe_path: str) -> str:
    try:
        if universe_path and os.path.isfile(universe_path):
            codes = _load_codes(universe_path, "")
            return ",".join(codes[:20])
    except Exception:
        pass
    return "600519"


def _render_cls_panel(universe_path: str = ""):
    """Align with original: per-ticker fetch + annotate only; not used in weights."""
    from datetime import date, timedelta
    from src.data.cls_news import (
        fetch_cls_news_many,
        get_cls_config,
        load_latest_summary,
        save_cls_config,
    )

    st.subheader("财联社新闻情绪（按股票标注，对齐原项目）")
    st.caption(
        "用法与原项目 fetch_news(ticker, start, end, analyze_sentiment=True) 一致："
        "按你指定的股票代码拉新闻，给每条新闻标 positive/neutral/negative。"
        "新闻源为财联社电报（替代 FMP）。策略选股/权重不会读取这些情绪。"
        "有 OPENAI_API_KEY 时优先 GPT；否则用中文词典兜底。"
    )
    cfg = get_cls_config()
    default_tickers = st.session_state.get("cls_tickers_input") or _default_cls_tickers(universe_path)
    tickers_text = st.text_area(
        "股票代码（逗号/换行分隔）",
        value=default_tickers,
        height=80,
        key="cls_tickers_input",
        help="默认取当前股票池文件前 20 只，可改成任意列表，例如 600519,000858",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        end_d = st.date_input("结束日期", value=date.today(), key="cls_end_date")
    with c2:
        start_d = st.date_input(
            "开始日期",
            value=date.today() - timedelta(days=3),
            key="cls_start_date",
        )
    with c3:
        page_size = st.number_input("每页条数", min_value=10, max_value=100, value=int(cfg.get("page_size", 50)), key="cls_page_size")
    max_pages = st.number_input("翻页数（回溯电报）", min_value=1, max_value=30, value=int(cfg.get("max_pages", 5)), key="cls_max_pages")
    use_openai = st.checkbox(
        "用 OpenAI 标注情绪（需 OPENAI_API_KEY；否则中文词典）",
        value=False,
        key="cls_use_openai",
    )
    analyze = st.checkbox("分析情绪（analyze_sentiment）", value=True, key="cls_analyze")

    if st.button("拉取新闻并分析情绪", key="cls_fetch_btn"):
        st.session_state.pop("cls_last_error", None)
        st.session_state.pop("cls_last_flash", None)
        try:
            codes = [
                c.strip().zfill(6)
                for c in tickers_text.replace("，", ",").replace("\n", ",").replace(" ", ",").split(",")
                if c.strip()
            ]
            if not codes:
                raise RuntimeError("请至少填写一个股票代码")
            save_cls_config(page_size=int(page_size), max_pages=int(max_pages))
            with st.spinner("正在按股票拉取财联社新闻并标注情绪…"):
                result = fetch_cls_news_many(
                    codes,
                    start_date=str(start_d),
                    end_date=str(end_d),
                    analyze_sentiment=bool(analyze),
                    use_openai=bool(use_openai),
                    page_size=int(page_size),
                    max_pages=int(max_pages),
                    save=True,
                )
            summary = load_latest_summary() or result.get("summary") or {}
            n = int(summary.get("n", 0) or 0)
            msg = (
                "已按 %d 只股票拉取，命中 %d 条新闻（来源 %s；标注=%s；正/中/负=%d/%d/%d）。文件：%s"
                % (
                    len(codes),
                    n,
                    summary.get("source", ""),
                    summary.get("sentiment_backend", ""),
                    int(summary.get("positive", 0) or 0),
                    int(summary.get("neutral", 0) or 0),
                    int(summary.get("negative", 0) or 0),
                    result.get("latest_csv") or (Path(get_cls_config()["cache_dir"]) / "cls_news_latest.csv"),
                )
            )
            st.session_state["cls_last_summary"] = summary
            st.session_state["cls_last_csv"] = result.get("latest_csv")
            if n <= 0:
                st.session_state["cls_last_flash"] = {"level": "warning", "message": msg}
            else:
                st.session_state["cls_last_flash"] = {"level": "success", "message": msg}
        except Exception as e:
            st.session_state["cls_last_error"] = str(e)
            st.session_state["cls_last_flash"] = None
            logger.exception("cls fetch failed")

    _render_cls_results_block()


def _strategy_controls():
    from src.strategies.ashare_strategy_runner import USE_CASE_PRESETS, STRATEGY_FAMILIES

    st.subheader("策略选择（README 三种用例）")
    use_case = st.radio(
        "内置用例预设",
        ["（不使用预设，手动选策略）"] + list(USE_CASE_PRESETS.keys()),
        horizontal=False,
        key="ashare_use_case",
    )
    if use_case in USE_CASE_PRESETS:
        st.caption(USE_CASE_PRESETS[use_case]["desc"])
        default_family = USE_CASE_PRESETS[use_case]["family"]
    else:
        default_family = "ml"

    family_labels = {k: v for k, v in STRATEGY_FAMILIES}
    family_keys = [k for k, _ in STRATEGY_FAMILIES]
    # 预设锁定默认族，仍允许改
    try:
        default_idx = family_keys.index(default_family)
    except ValueError:
        default_idx = family_keys.index("ml")

    family = st.selectbox(
        "策略族",
        family_keys,
        index=default_idx,
        format_func=lambda k: "%s — %s" % (k, family_labels.get(k, k)),
        key="ashare_strategy_family",
    )
    from src.web.ashare_strategy_help import render_strategy_family_help
    render_strategy_family_help(st, key_prefix="ashare_live_strategy_help")
    if family == "market_cap":
        st.caption("市值来自通达信 gbbq 股本(万股)×收盘价；默认总市值，可改流通市值。")

    params = {}
    with st.expander("策略参数", expanded=True):
        if family in ("ml", "sector_neutral_ml", "ml_bucket", "ml_then_drl", "custom"):
            c1, c2, c3 = st.columns(3)
            with c1:
                params["prediction_mode"] = st.selectbox("预测模式", ["single", "rolling"], index=1 if default_family == "ml_then_drl" else 0)
                params["weight_method"] = st.selectbox("ML 定权", ["min_variance", "equal"])
                params["confirm_mode"] = st.selectbox("当日确认", ["none", "today"])
            with c2:
                params["train_quarters"] = st.number_input("训练季度数", min_value=4, max_value=40, value=16)
                params["test_quarters"] = st.number_input("测试季度数", min_value=1, max_value=12, value=4)
                params["top_quantile"] = st.number_input("入选分位数阈值", min_value=0.5, max_value=0.95, value=0.75, step=0.05)
            with c3:
                params["lookback_periods"] = st.number_input("协方差回溯（季/日）", min_value=4, max_value=504, value=8)

        if family in ("drl", "ml_then_drl") or (use_case == "用例1·组合配置范式" and family == "drl"):
            c1, c2, c3 = st.columns(3)
            with c1:
                params["drl_algo"] = st.selectbox("DRL 算法", ["ppo", "a2c", "ddpg"], index=0)
            with c2:
                params["drl_epochs"] = st.number_input("Torch 训练轮数", min_value=5, max_value=200, value=40)
            with c3:
                params["drl_timesteps"] = st.number_input("SB3 步数（若已装）", min_value=1000, max_value=100000, value=8000, step=1000)
            params["drl_top_n"] = st.number_input("DRL 标的上限（0=不限）", min_value=0, max_value=300, value=50)
            if params["drl_top_n"] == 0:
                params["drl_top_n"] = None

        if family == "finrl_rolling_drl":
            st.warning(
                "FinRL滚动DRL 是完整训推（可能很久）。默认 dry-run 下单；产物落在 artifacts/ashare_drl/。"
                "短训 DRL 请选「drl」。"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                algo_opts = st.multiselect(
                    "训练算法",
                    ["a2c", "ppo", "ddpg", "td3", "sac"],
                    default=["a2c", "ppo", "ddpg"],
                    key="ashare_finrl_algos",
                )
                params["finrl_algos"] = tuple(algo_opts) if algo_opts else ("a2c", "ppo", "ddpg")
                params["finrl_rebalance_freq"] = st.selectbox("再平衡频率", ["Q", "M"], index=0, key="ashare_finrl_freq")
            with c2:
                params["finrl_testing_window_days"] = st.number_input("测试窗(天)", 30, 750, 365, key="ashare_finrl_test")
                params["finrl_max_rolling_window_days"] = st.number_input("训练回看(天)", 60, 2000, 1095, key="ashare_finrl_roll")
                params["finrl_lookback"] = st.number_input("协方差lookback", 20, 504, 252, key="ashare_finrl_lb")
            with c3:
                params["finrl_total_timesteps"] = st.number_input(
                    "每算法timesteps(0=用FinRL默认)", 0, 500000, 0, step=1000, key="ashare_finrl_ts"
                )
                if params["finrl_total_timesteps"] == 0:
                    params["finrl_total_timesteps"] = None
                params["finrl_seed"] = st.number_input("随机种子", 0, 999999, 42, key="ashare_finrl_seed")
                params["finrl_resume"] = st.checkbox("断点续跑(progress.json)", value=True, key="ashare_finrl_resume")
            params["finrl_output_dir"] = st.text_input(
                "输出目录",
                value=os.path.join(REPO_ROOT, "artifacts", "ashare_drl"),
                key="ashare_finrl_outdir",
            )
            params["finrl_max_stocks"] = st.number_input("最多股票数(0=不限，调试可设5)", 0, 500, 0, key="ashare_finrl_maxn")
            if params["finrl_max_stocks"] == 0:
                params["finrl_max_stocks"] = None
            params["finrl_max_windows"] = st.number_input("最多滚动窗(0=不限)", 0, 100, 0, key="ashare_finrl_maxw")
            if params["finrl_max_windows"] == 0:
                params["finrl_max_windows"] = None


        if family == "market_cap":
            params["mv_field"] = st.radio(
                "市值口径",
                ["total_mv", "circ_mv"],
                format_func=lambda k: "总市值" if k == "total_mv" else "流通市值",
                horizontal=True,
                key="ashare_mv_field",
            )
            st.caption("公式：市值 = 收盘价 × 股本(万股) × 10000；股本来自 <tdx_dir>/T0002/hq_cache/gbbq")

        if family in ("min_variance", "mean_variance"):
            params["lookback_periods"] = st.number_input("价格回溯交易日", min_value=60, max_value=756, value=252)

        if family == "adaptive_rotation":
            params["as_of_date"] = st.text_input("决策日（可空=数据最新）", "")
            st.caption("将用股票池价格生成 A 股版 AdaptiveRotationConf，并代理 ^GSPC/^VIX/QQQ。需要 PyYAML。")

        if family == "custom":
            params["custom_path"] = st.text_input(
                "自定义策略 .py 路径",
                os.path.join(REPO_ROOT, "examples", "ashare_custom_strategy_example.py"),
            )
            params["custom_class"] = st.text_input("类名（可空=文件中第一个 BaseStrategy 子类）", "")
            st.caption("文件内需定义继承 BaseStrategy 的类，实现 generate_weights(data, **kwargs)->StrategyResult。")

        if family == "sector_neutral_ml":
            st.caption("将自动 with_industry=True，按 gsector 分行业训练再合并。")
        if family == "ml_bucket":
            st.caption("A股无美股 GICS 四桶原始数据；此处按行业中性路径落地（行业映射四组思想）。")

    # 用例预设覆盖
    if use_case == "用例2·滚动选股+DRL":
        params.setdefault("prediction_mode", "rolling")
        params.setdefault("drl_algo", "ppo")
        if family != "ml_then_drl":
            st.info("用例2建议策略族选「ml_then_drl」。当前族=%s" % family)

    return family, params, use_case


def show_ashare_live():
    st.header("A股实盘")
    st.caption(
        "点一次「启动」：读股票池 → 按所选策略生成权重 → 生成委托。默认只预览；勾选确认后才真实下单。"
        "策略覆盖 README 三种用例及原项目可选策略族；新闻情绪用财联社（非 FMP）。"
    )

    choice = st.radio(
        "股票池",
        ["沪深300", "全部A股"],
        horizontal=True,
        key="ashare_universe_choice",
    )
    applied = st.session_state.get("_ashare_universe_applied")
    if applied != choice or "ashare_universe_path" not in st.session_state:
        path, cache_prefix = UNIVERSE_PRESETS[choice]
        st.session_state["ashare_universe_path"] = path
        st.session_state["ashare_cache_prefix"] = cache_prefix
        st.session_state["_ashare_universe_applied"] = choice
    st.caption("切换选项会更新文件和缓存前缀；路径仍可手改。行业策略会额外使用 *_ind 缓存。")

    c1, c2 = st.columns(2)
    with c1:
        universe = st.text_input("股票池文件", key="ashare_universe_path")
    with c2:
        cache = st.text_input("数据缓存前缀", key="ashare_cache_prefix")

    if st.button("生成股票池"):
        try:
            if choice == "沪深300":
                generated, source = fetch_hs300_codes()
            elif choice == "全部A股":
                generated, source = fetch_all_a_codes()
            else:
                raise RuntimeError("未知股票池: %s" % choice)
            written = write_universe_file(universe, generated)
            st.success("已写入 %d 只（来源: %s）：%s" % (written, source, universe))
        except Exception as e:
            st.error("生成股票池失败：%s" % e)
            logger.exception("generate universe failed")

    if universe and os.path.isfile(universe):
        try:
            st.caption("当前股票池文件 %d 只" % len(_load_codes(universe, "")))
        except Exception as e:
            st.caption("当前股票池文件读不了：%s" % e)
    else:
        st.caption("股票池文件还不存在，点「生成股票池」会按当前选项拉取并写入。")

    if "ashare_tdx_dir" not in st.session_state:
        st.session_state["ashare_tdx_dir"] = _default_tdx_dir()
    tdx_dir = st.text_input(
        "通达信目录",
        key="ashare_tdx_dir",
        help="通达信安装根目录（其下应有 vipdoc）。优先序：本框 > 环境变量 TDX_DIR > user_config.json 的 TDX_CONFIG.tdx_dir。",
    )

    if "ashare_start_year" not in st.session_state:
        st.session_state["ashare_start_year"] = int(_default_start_year())
    start_year = st.number_input(
        "数据起始年份",
        min_value=2005,
        max_value=2026,
        value=int(st.session_state.get("ashare_start_year", 2018)),
        step=1,
        key="ashare_start_year",
        help="基本面/数据集起始年。默认 2018；写入 user_config.json 的 DATA_CONFIG.start_year。非 2018 时缓存前缀会自动加 _y年份，避免和默认缓存混用；改年份后如需全量重刷请到「数据管理」重建。",
    )
    st.caption(
        "起始年=%d。默认年(2018)用原缓存前缀；其它年份自动使用「缓存前缀_y年份」。改年份若仍看到旧数，请重建数据集。"
        % int(start_year)
    )

    family, strategy_params, use_case = _strategy_controls()
    _render_cls_panel(universe)

    with st.form("ashare_run"):
        c1, c2 = st.columns(2)
        with c1:
            manual = st.text_input("或手填代码（逗号分隔，填了就不用上面的文件）", "")
            out_csv = st.text_input("权重输出", os.path.join(REPO_ROOT, "a_share_weights.csv"))
        with c2:
            cash = st.number_input("预览资金（元，实盘时改用账户可用资金）", min_value=10000.0, value=1000000.0, step=10000.0)
            min_weight = st.number_input("最小权重", min_value=0.0, value=0.0001, format="%.4f")

        st.subheader("券商")
        kind = st.selectbox("通道", ["仅预览", "同花顺", "雪球", "miniQMT"])
        ths_path = st.text_input("同花顺下单程序", r"D:\同花顺软件\同花顺\xiadan.exe")
        xq_json = st.text_input("雪球配置", os.path.join(".", "xq.json"))
        qmt_path = st.text_input("miniQMT 路径", r"D:\GJQMT\userdata_mini")
        qmt_account = st.text_input("资金账号（miniQMT）", "")
        confirm = st.checkbox("确认实盘下单（不勾选则只预览，不会发出委托）")
        go = st.form_submit_button("启动")

    if not go:
        _show_finrl_run_status(
            st.session_state.get("ashare_finrl_output_dir")
            or (strategy_params or {}).get("finrl_output_dir")
        )
        return

    live = confirm and kind != "仅预览"
    if confirm and kind == "仅预览":
        st.warning("勾了实盘但通道是「仅预览」，这次只出权重和委托单，不会下单。")
    broker = {"kind": {"同花顺": "ths", "雪球": "xq", "miniQMT": "miniqmt"}.get(kind, "ths")}
    if broker["kind"] == "ths":
        broker["ths_path"] = ths_path
    elif broker["kind"] == "xq":
        broker["prepare_file"] = xq_json
    else:
        broker["miniqmt_path"] = qmt_path
        broker["stock_account"] = qmt_account

    # 用例2强制族
    if use_case == "用例2·滚动选股+DRL":
        family = "ml_then_drl"
        strategy_params["prediction_mode"] = strategy_params.get("prediction_mode") or "rolling"
    elif use_case == "用例3·自适应多资产轮动":
        family = "adaptive_rotation"
    elif use_case == "用例4·FinRL滚动DRL" or (isinstance(use_case, str) and "FinRL" in use_case and "滚动" in use_case):
        family = "finrl_rolling_drl"

    elif use_case == "用例1·组合配置范式" and family not in ("equal_weight", "min_variance", "mean_variance", "market_cap", "drl"):
        st.warning("用例1通常选等权/最小方差/均值方差/DRL；继续用当前族=%s" % family)

    status = st.empty()
    bar_slot = st.empty()
    log_slot = st.empty()
    ui_progress = _make_ui_progress(status, log_slot, bar_slot)
    try:
        codes = _load_codes(universe, manual)
        status.info("股票池 %d 只 | 用例=%s | 策略=%s" % (len(codes), use_case, family))
        try:
            _persist_tdx_dir(tdx_dir)
        except Exception as e:
            logger.warning("persist tdx_dir failed: %s", e)
        try:
            _persist_start_year(int(start_year))
        except Exception as e:
            logger.warning("persist start_year failed: %s", e)
        strategy_params = dict(strategy_params or {})
        strategy_params["tdx_dir"] = tdx_dir
        if family == "finrl_rolling_drl":
            out_dir = strategy_params.get("finrl_output_dir") or os.path.join(REPO_ROOT, "artifacts", "ashare_drl")
            st.session_state["ashare_finrl_running"] = True
            st.session_state["ashare_finrl_output_dir"] = out_dir
            st.session_state["ashare_finrl_last_msg"] = "启动 FinRL 滚动 DRL…"
            st.session_state["ashare_finrl_msgs"] = ["启动 FinRL 滚动 DRL…"]
            st.caption("长跑产物：%s （live_progress.txt / progress.json）" % out_dir)
        result = run_pipeline(
            codes,
            cache,
            min_weight,
            out_csv,
            live,
            broker,
            None if live else cash,
            progress=ui_progress,
            tdx_dir=tdx_dir,
            strategy_family=family,
            strategy_params=strategy_params,
            start_year=int(start_year),
        )
        st.session_state["ashare_finrl_running"] = False
        st.session_state["ashare_finrl_last_msg"] = "已完成"
    except Exception as e:
        st.session_state["ashare_finrl_running"] = False
        st.session_state["ashare_finrl_last_msg"] = "失败：%s" % e
        status.empty()
        st.error("没跑完：%s" % e)
        logger.exception("ashare pipeline failed")
        return

    status.empty()
    try:
        bar_slot.empty()
    except Exception:
        pass
    meta = result.get("meta") or {}
    if result["submitted"]:
        st.success("已向券商提交委托。可投资总额约 %s 元，目标 %d 只。" % (format(result["total_value"], ",.0f"), len(result["weights"])))
    else:
        st.success("预览完成，没有下单。可投资总额约 %s 元，目标 %d 只。" % (format(result["total_value"], ",.0f"), len(result["weights"])))
    if meta:
        st.caption("策略元数据：%s" % meta)

    st.subheader("权重")
    show = result["weights"].copy()
    if "weight" in show.columns:
        show["weight"] = (show["weight"] * 100).round(2)
    st.dataframe(show, width='stretch')

    st.subheader("委托")
    if result["orders"]:
        st.dataframe(pd.DataFrame(result["orders"]), width='stretch')
    else:
        st.info("没有需要下的单（已对齐，或都不够一手）。")

    if result["positions"]:
        st.subheader("当前持仓")
        rows = []
        for k, v in result["positions"].items():
            row = {"代码": k}
            row.update(v)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width='stretch')
