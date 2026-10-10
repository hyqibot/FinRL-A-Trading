# -*- coding: utf-8 -*-
"""A股口径的总览 / 数据管理 / 策略回测 / 绩效分析 / 设置。不碰 A股实盘页的财联社逻辑。"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    return REPO_ROOT


def _user_config_path() -> Path:
    return _root() / "user_config.json"


def _read_user_config() -> Dict[str, Any]:
    p = _user_config_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_user_config(cfg: Dict[str, Any]) -> None:
    p = _user_config_path()
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _universe_presets() -> Dict[str, Tuple[str, str]]:
    r = str(_root())
    return {
        "沪深300": (os.path.join(r, "hs300.txt"), os.path.join(r, "data", "cache_hs300")),
        "全部A股": (os.path.join(r, "all_a.txt"), os.path.join(r, "data", "cache_all_a")),
    }


def _count_universe(path: str) -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = "".join(ch for ch in line.strip() if ch.isdigit())
        if len(s) >= 6:
            n += 1
    return n


def _cache_exists(prefix: str) -> Dict[str, Any]:
    fund = Path(str(prefix) + ".fund.parquet")
    px = Path(str(prefix) + ".px.parquet")
    info: Dict[str, Any] = {
        "fund_path": str(fund),
        "px_path": str(px),
        "fund_ok": fund.is_file(),
        "px_ok": px.is_file(),
        "fund_size": fund.stat().st_size if fund.is_file() else 0,
        "px_size": px.stat().st_size if px.is_file() else 0,
    }
    return info


def _load_cached_ds(prefix: str) -> Optional[Dict[str, pd.DataFrame]]:
    info = _cache_exists(prefix)
    if not (info["fund_ok"] and info["px_ok"]):
        return None
    try:
        return {
            "fundamentals": pd.read_parquet(info["fund_path"]),
            "prices": pd.read_parquet(info["px_path"]),
        }
    except Exception as exc:
        logger.warning("cache read failed: %s", exc)
        return None


def _mb(n: int) -> str:
    return "%.1f MB" % (n / (1024 * 1024.0))


def _latest_weights_path() -> Optional[Path]:
    candidates = [
        _root() / "a_share_weights.csv",
        _root() / "a_share_weights_clean.csv",
        _root() / "data" / "backtest_last_weights.csv",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _load_weights_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    tic_col = cols.get("tic") or cols.get("code") or cols.get("symbol")
    w_col = cols.get("weight") or cols.get("wt") or cols.get("w")
    if tic_col is None or w_col is None:
        raise ValueError("权重文件需要 tic/weight 列: %s" % path)
    out = pd.DataFrame({
        "tic": df[tic_col].astype(str).str.zfill(6),
        "weight": pd.to_numeric(df[w_col], errors="coerce"),
    }).dropna()
    s = out["weight"].sum()
    if s > 0:
        out["weight"] = out["weight"] / s
    return out


def display_quick_stats_ashare() -> None:
    st.subheader("快捷状态")
    presets = _universe_presets()
    try:
        tdx = None
        try:
            from src.config.ashare_tdx import get_tdx_dir, validate_tdx_dir
            tdx = get_tdx_dir()
            try:
                validate_tdx_dir(tdx, require_vipdoc=True)
                tdx_ok = True
            except Exception:
                tdx_ok = False
        except Exception:
            tdx_ok = False
            tdx = "(未配置)"

        hs = _count_universe(presets["沪深300"][0])
        aa = _count_universe(presets["全部A股"][0])
        cinfo = _cache_exists(presets["沪深300"][1])
        st.caption("通达信: %s%s" % (tdx, "" if tdx_ok else "（无效或不存在）"))
        c1, c2 = st.columns(2)
        with c1:
            st.metric("沪深300池", "%d 只" % hs)
        with c2:
            st.metric("全部A股池", "%d 只" % aa)
        st.caption(
            "沪深300缓存: 基本面 %s / 行情 %s"
            % (
                _mb(cinfo["fund_size"]) if cinfo["fund_ok"] else "无",
                _mb(cinfo["px_size"]) if cinfo["px_ok"] else "无",
            )
        )
        w = _latest_weights_path()
        if w:
            st.caption("最近权重: %s" % w.name)
    except Exception as e:
        st.caption("状态读取失败: %s" % e)


def show_overview_ashare() -> None:
    st.header("总览（A股）")
    st.caption("面向 A 股通达信行情、akshare 基本面、easytrader 通道；货币单位人民币。")

    presets = _universe_presets()
    hs_path, hs_cache = presets["沪深300"]
    aa_path, aa_cache = presets["全部A股"]
    hs_n, aa_n = _count_universe(hs_path), _count_universe(aa_path)
    hs_c, aa_c = _cache_exists(hs_cache), _cache_exists(aa_cache)

    try:
        from src.config.ashare_tdx import get_tdx_dir, validate_tdx_dir
        tdx = get_tdx_dir()
        try:
            validate_tdx_dir(tdx, require_vipdoc=True)
            tdx_msg = tdx
        except Exception as e:
            tdx_msg = "%s（校验失败: %s）" % (tdx, e)
    except Exception as e:
        tdx_msg = "读取失败: %s" % e

    wpath = _latest_weights_path()
    w_n, w_sum = 0, 0.0
    if wpath:
        try:
            wdf = _load_weights_csv(wpath)
            w_n, w_sum = len(wdf), float(wdf["weight"].sum())
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("沪深300股票池", "%d 只" % hs_n)
    with c2:
        st.metric("全部A股股票池", "%d 只" % aa_n)
    with c3:
        st.metric("最近目标权重", "%d 只" % w_n, "%.3f 合计" % w_sum if w_n else None)
    with c4:
        st.metric("沪深300行情缓存", _mb(hs_c["px_size"]) if hs_c["px_ok"] else "无")

    st.subheader("环境与数据")
    st.write("- 通达信目录：`%s`" % tdx_msg)
    st.write(
        "- 沪深300缓存：基本面 %s，行情 %s"
        % (
            "有 (%s)" % _mb(hs_c["fund_size"]) if hs_c["fund_ok"] else "无",
            "有 (%s)" % _mb(hs_c["px_size"]) if hs_c["px_ok"] else "无",
        )
    )
    st.write(
        "- 全部A股缓存：基本面 %s，行情 %s"
        % (
            "有 (%s)" % _mb(aa_c["fund_size"]) if aa_c["fund_ok"] else "无",
            "有 (%s)" % _mb(aa_c["px_size"]) if aa_c["px_ok"] else "无",
        )
    )
    st.write("- 权重文件：%s" % (str(wpath) if wpath else "尚未生成（可在「A股实盘」或「策略回测」产出）"))

    st.subheader("最近活动（本地文件）")
    rows = []
    for label, path in [
        ("沪深300股票池", hs_path),
        ("全部A股股票池", aa_path),
        ("沪深300行情缓存", hs_c["px_path"]),
        ("沪深300基本面缓存", hs_c["fund_path"]),
        ("权重 CSV", str(wpath) if wpath else ""),
        ("财联社最新情绪", str(_root() / "data" / "cls_news" / "cls_sentiment_latest.json")),
        ("回测结果", str(_root() / "data" / "backtest_last.json")),
    ]:
        p = Path(path) if path else None
        if p and p.is_file():
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            rows.append({"项目": label, "路径": str(p), "更新时间": mtime, "大小": _mb(p.stat().st_size)})
        else:
            rows.append({"项目": label, "路径": path or "-", "更新时间": "-", "大小": "-"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("组合净值示意（基于最近权重 × 沪深300缓存，若可用）")
    ds = _load_cached_ds(hs_cache)
    if wpath and ds is not None and not ds["prices"].empty:
        try:
            wdf = _load_weights_csv(wpath)
            nav = _simulate_nav(ds["prices"], wdf, initial_capital=1_000_000.0)
            if nav is not None and not nav.empty:
                fig = px.line(nav, x="date", y="nav", title="示意净值（人民币，非实盘）")
                fig.update_layout(xaxis_title="日期", yaxis_title="组合净值（元）")
                st.plotly_chart(fig, use_container_width=True)
                st.caption("用最近一次权重在缓存行情上做持有期示意，不等于实盘账户。")
            else:
                st.info("权重与缓存行情交集不足，无法画示意净值。")
        except Exception as e:
            st.warning("示意净值失败：%s" % e)
    else:
        st.info("需要权重 CSV 与沪深300行情缓存后，才会显示示意净值曲线。")


def _prices_to_wide(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    if "tic" not in df.columns:
        raise ValueError("prices 缺少 tic")
    date_col = "date" if "date" in df.columns else ("datadate" if "datadate" in df.columns else None)
    if date_col is None:
        raise ValueError("prices 缺少 date")
    close_col = "close" if "close" in df.columns else ("adj_close" if "adj_close" in df.columns else None)
    if close_col is None:
        raise ValueError("prices 缺少 close")
    df[date_col] = pd.to_datetime(df[date_col])
    df["tic"] = df["tic"].astype(str).str.zfill(6)
    wide = df.pivot_table(index=date_col, columns="tic", values=close_col, aggfunc="last").sort_index()
    return wide


def _simulate_nav(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    wide = _prices_to_wide(prices)
    if start:
        wide = wide.loc[wide.index >= pd.to_datetime(start)]
    if end:
        wide = wide.loc[wide.index <= pd.to_datetime(end)]
    if wide.empty:
        return None
    w = weights.copy()
    w["tic"] = w["tic"].astype(str).str.zfill(6)
    wmap = dict(zip(w["tic"], w["weight"]))
    cols = [c for c in w["tic"].tolist() if c in wide.columns]
    if not cols:
        return None
    px = wide[cols].copy()
    # 去掉区间内完全无行情的列，再前向/后向填充，避免单列空洞导致整表 dropna 变空
    px = px.dropna(axis=1, how="all")
    if px.shape[1] == 0:
        return None
    px = px.ffill().bfill()
    # 仍全空的行去掉
    px = px.dropna(how="all")
    # 若仍有零星 NaN，用列中位数填充；再不行就丢列
    if px.isna().any().any():
        px = px.apply(lambda s: s.fillna(s.median()), axis=0)
        px = px.dropna(axis=1, how="any")
    px = px.dropna(how="any")
    if len(px) < 2 or px.shape[1] == 0:
        return None
    cols = list(px.columns)
    vec = np.array([float(wmap.get(c, 0.0)) for c in cols], dtype=float)
    s = vec.sum()
    if s <= 0:
        return None
    vec = vec / s
    rets = px.pct_change().fillna(0.0)
    port_ret = rets.values @ vec
    nav = initial_capital * np.cumprod(1.0 + port_ret)
    nav[0] = initial_capital
    return pd.DataFrame({"date": px.index, "nav": nav, "ret": port_ret})



def _backtest_metrics(nav_df: pd.DataFrame) -> Dict[str, float]:
    nav = nav_df["nav"].astype(float)
    rets = nav.pct_change().dropna()
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) else 0.0
    ann = float(rets.mean() * 252) if len(rets) else 0.0
    vol = float(rets.std() * np.sqrt(252)) if len(rets) else 0.0
    sharpe = float(ann / vol) if vol > 1e-12 else 0.0
    cum = (1.0 + rets).cumprod()
    dd = float((cum / cum.cummax() - 1.0).min()) if len(cum) else 0.0
    return {
        "final_value": float(nav.iloc[-1]) if len(nav) else 0.0,
        "total_return": total_return,
        "annual_return": ann,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "n_days": int(len(nav)),
    }


def show_data_management_ashare() -> None:
    st.header("数据管理（A股）")
    st.caption("股票池文件、通达信行情缓存、akshare 基本面缓存。美股 WRDS / FMP / S&P500 流程已停用。")

    tab1, tab2, tab3, tab4 = st.tabs(["股票池与缓存", "重建数据集", "本地上传", "数据质量"])

    presets = _universe_presets()
    with tab1:
        st.subheader("股票池文件")
        for name, (path, cache) in presets.items():
            info = _cache_exists(cache)
            st.markdown("**%s**" % name)
            st.code(path)
            st.write("代码数：%d；缓存前缀：`%s`" % (_count_universe(path), cache))
            st.write(
                "基本面缓存：%s；行情缓存：%s"
                % (
                    _mb(info["fund_size"]) if info["fund_ok"] else "无",
                    _mb(info["px_size"]) if info["px_ok"] else "无",
                )
            )
            if Path(path).is_file():
                sample = []
                for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = "".join(ch for ch in line.strip() if ch.isdigit())
                    if len(s) >= 6:
                        sample.append(s[-6:].zfill(6))
                    if len(sample) >= 10:
                        break
                st.caption("样例：" + ", ".join(sample))

        st.subheader("通达信目录")
        try:
            from src.config.ashare_tdx import get_tdx_dir, validate_tdx_dir, save_tdx_dir
            cur = get_tdx_dir()
            new_tdx = st.text_input("通达信安装根目录（含 vipdoc）", value=cur, key="dm_tdx_dir")
            if st.button("保存通达信目录", key="dm_save_tdx"):
                save_tdx_dir(new_tdx)
                try:
                    validate_tdx_dir(new_tdx, require_vipdoc=True)
                    st.success("已保存并校验通过：%s" % new_tdx)
                except Exception as e:
                    st.warning("已保存，但校验未通过：%s" % e)
        except Exception as e:
            st.error("通达信配置模块不可用：%s" % e)

        st.subheader("缓存文件列表（data/）")
        data_dir = _root() / "data"
        rows = []
        if data_dir.is_dir():
            for p in sorted(data_dir.glob("cache_*.parquet")):
                rows.append({"文件": p.name, "大小": _mb(p.stat().st_size), "修改时间": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("data/ 下还没有 cache_*.parquet")

        if st.button("删除全部A股缓存文件（cache_all_a*）", key="dm_del_alla"):
            removed = []
            for p in data_dir.glob("cache_all_a*"):
                if p.is_file():
                    p.unlink()
                    removed.append(p.name)
            st.success("已删除：%s" % (", ".join(removed) if removed else "无"))

    with tab2:
        st.subheader("重建 A 股数据集")
        st.warning("会读通达信日线并联网拉 akshare 基本面，耗时可能很长。建议先用沪深300；全部A股请谨慎。")
        choice = st.radio("股票池", ["沪深300", "全部A股"], horizontal=True, key="dm_rebuild_choice")
        path, cache = presets[choice]
        try:
            from src.config.ashare_tdx import get_start_year as _get_sy
            _dm_default_year = int(_get_sy())
        except Exception:
            _dm_default_year = 2018
        start_year = st.number_input(
            "数据起始年份",
            min_value=2005,
            max_value=2026,
            value=_dm_default_year,
            key="dm_start_year",
            help="与 A股实盘 页相同；写入构建数据集的 start_year。非 2018 时缓存前缀加 _y年份。",
        )
        with_ind = st.checkbox("附带行业映射（行业中性/分桶策略需要）", value=False, key="dm_with_ind")
        max_codes = st.number_input("最多使用股票数（0=全部）", min_value=0, max_value=6000, value=0, key="dm_max_codes")
        if st.button("开始重建缓存", key="dm_rebuild_btn"):
            try:
                from src.data.ashare_data_adapter import build_dataset
                from src.config.ashare_tdx import get_tdx_dir
                codes = []
                for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
                    s = "".join(ch for ch in line.strip() if ch.isdigit())
                    if len(s) >= 6:
                        codes.append(s[-6:].zfill(6))
                if not codes:
                    raise RuntimeError("股票池文件为空或不存在，请先在「A股实盘」生成股票池")
                if max_codes and max_codes > 0:
                    codes = codes[: int(max_codes)]
                cache_path = cache
                if int(start_year) != 2018:
                    cache_path = cache_path + ("_y%d" % int(start_year))
                if with_ind:
                    cache_path = cache_path + "_ind"
                with st.spinner("正在构建 %d 只…" % len(codes)):
                    # force rebuild by removing old cache first
                    for suffix in (".fund.parquet", ".px.parquet"):
                        old = Path(cache_path + suffix)
                        if old.is_file():
                            old.unlink()
                    try:
                        from src.config.ashare_tdx import save_start_year
                        save_start_year(int(start_year))
                    except Exception:
                        pass
                    ds = build_dataset(
                        codes,
                        start_year=int(start_year),
                        with_industry=bool(with_ind),
                        cache_path=cache_path,
                        tdx_dir=get_tdx_dir(),
                    )
                st.success(
                    "完成：基本面 %s，行情 %s，缓存前缀 %s"
                    % (ds["fundamentals"].shape, ds["prices"].shape, cache_path)
                )
            except Exception as e:
                st.error("重建失败：%s" % e)
                logger.exception("rebuild dataset failed")

        st.divider()
        st.subheader("美股 / WRDS / FMP")
        st.info("原 Overview 数据管理中的 S&P500、WRDS、FMP 拉取已在本页停用。A 股请用上方重建或「A股实盘」生成股票池。")

    with tab3:
        st.subheader("上传本地 CSV")
        uploaded = st.file_uploader("上传 CSV（可预览）", type=["csv"], key="dm_upload")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("行数：%d，列：%s" % (len(df), list(df.columns)))
            st.dataframe(df.head(50), use_container_width=True)
            save_name = st.text_input("保存到 data/ 的文件名", value="uploaded_ashare.csv", key="dm_upload_name")
            if st.button("保存到 data/", key="dm_upload_save"):
                out = _root() / "data" / Path(save_name).name
                df.to_csv(out, index=False, encoding="utf-8-sig")
                st.success("已保存：%s" % out)

    with tab4:
        st.subheader("缓存质量快检")
        choice2 = st.selectbox("检查哪套缓存", list(presets.keys()), key="dm_quality_choice")
        _, cache2 = presets[choice2]
        ds = _load_cached_ds(cache2)
        if ds is None:
            st.warning("缓存不完整：%s" % cache2)
        else:
            fund, px = ds["fundamentals"], ds["prices"]
            rows = [
                {"指标": "基本面行数", "值": len(fund)},
                {"指标": "基本面列数", "值": fund.shape[1]},
                {"指标": "行情行数", "值": len(px)},
                {"指标": "行情股票数", "值": px["tic"].nunique() if "tic" in px.columns else "-"},
                {"指标": "基本面股票数", "值": fund["tic"].nunique() if "tic" in fund.columns else "-"},
                {"指标": "基本面缺失率", "值": "%.2f%%" % (100.0 * float(fund.isna().mean().mean()))},
                {"指标": "行情缺失率", "值": "%.2f%%" % (100.0 * float(px.isna().mean().mean()))},
            ]
            if "date" in px.columns:
                rows.append({"指标": "行情日期起", "值": str(pd.to_datetime(px["date"]).min().date())})
                rows.append({"指标": "行情日期止", "值": str(pd.to_datetime(px["date"]).max().date())})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if "tic" in fund.columns:
                st.caption("基本面样例代码：" + ", ".join(fund["tic"].astype(str).str.zfill(6).unique()[:12].tolist()))


def show_strategy_backtesting_ashare() -> None:
    st.header("策略回测（A股）")
    st.caption("使用 ashare_strategy_runner 与本地缓存；净值单位人民币。不会连接券商下单。")

    from src.strategies.ashare_strategy_runner import STRATEGY_FAMILIES

    presets = _universe_presets()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("回测配置")
        uni = st.selectbox("股票池缓存", list(presets.keys()), key="bt_universe")
        path, cache = presets[uni]
        family_keys = [k for k, _ in STRATEGY_FAMILIES]
        family_labels = {k: v for k, v in STRATEGY_FAMILIES}
        family = st.selectbox(
            "策略族",
            family_keys,
            index=family_keys.index("equal_weight") if "equal_weight" in family_keys else 0,
            format_func=lambda k: "%s — %s" % (k, family_labels.get(k, k)),
            key="bt_family",
        )
        from src.web.ashare_strategy_help import render_strategy_family_help
        render_strategy_family_help(st, key_prefix="bt_strategy_help")
        start_date = st.date_input("回测开始", datetime(2024, 1, 1), key="bt_start")
        end_date = st.date_input("回测结束", datetime(2026, 9, 30), key="bt_end")
        initial_capital = st.number_input("初始资金（元）", value=1_000_000, step=100_000, key="bt_cash")

        params: Dict[str, Any] = {}
        with st.expander("策略参数", expanded=False):
            if family in ("ml", "sector_neutral_ml", "ml_bucket", "ml_then_drl", "custom"):
                params["prediction_mode"] = st.selectbox("预测模式", ["single", "rolling"], key="bt_pred")
                params["weight_method"] = st.selectbox("ML 定权", ["min_variance", "equal"], key="bt_wm")
                params["top_quantile"] = st.slider("入选分位阈值", 0.5, 0.95, 0.75, 0.05, key="bt_q")
                params["train_quarters"] = st.number_input("训练季度", 4, 40, 16, key="bt_tq")
                params["test_quarters"] = st.number_input("测试季度", 1, 12, 4, key="bt_testq")
            
            if family == "finrl_rolling_drl":
                st.caption("完整 FinRL 滚动训推；耗时长。短训请选 drl。")
                params["finrl_algos"] = tuple(st.multiselect("算法", ["a2c","ppo","ddpg","td3","sac"], default=["a2c","ppo","ddpg"], key="bt_finrl_algos") or ["a2c","ppo","ddpg"])
                params["finrl_testing_window_days"] = st.number_input("测试窗天", 30, 750, 365, key="bt_finrl_test")
                params["finrl_max_rolling_window_days"] = st.number_input("训练回看天", 60, 2000, 1095, key="bt_finrl_roll")
                params["finrl_lookback"] = st.number_input("lookback", 20, 504, 252, key="bt_finrl_lb")
                ts = st.number_input("timesteps(0=默认)", 0, 500000, 5000, step=500, key="bt_finrl_ts")
                params["finrl_total_timesteps"] = None if ts == 0 else int(ts)
                params["finrl_max_stocks"] = st.number_input("max_stocks(0不限)", 0, 100, 10, key="bt_finrl_ms")
                if params["finrl_max_stocks"] == 0:
                    params["finrl_max_stocks"] = None
                params["finrl_max_windows"] = st.number_input("max_windows(0不限)", 0, 20, 2, key="bt_finrl_mw")
                if params["finrl_max_windows"] == 0:
                    params["finrl_max_windows"] = None
                params["finrl_resume"] = st.checkbox("resume", value=False, key="bt_finrl_resume")
                params["finrl_output_dir"] = st.text_input("output_dir", value=str(_root() / "artifacts" / "ashare_drl"), key="bt_finrl_out")
                params["finrl_seed"] = 42
            if family in ("drl", "ml_then_drl"):
                params["drl_algo"] = st.selectbox("DRL 算法", ["ppo", "a2c", "ddpg"], key="bt_algo")
            if family in ("min_variance", "mean_variance", "ml", "sector_neutral_ml", "ml_bucket", "ml_then_drl"):
                params["lookback_periods"] = st.number_input("回看期", 4, 504, 252 if family in ("min_variance", "mean_variance") else 8, key="bt_lb")
            if family == "market_cap":
                params["mv_field"] = st.selectbox("市值口径", ["total_mv", "circ_mv"], format_func=lambda x: "总市值" if x == "total_mv" else "流通市值", key="bt_mv")
            if family == "custom":
                params["custom_path"] = st.text_input("自定义策略 .py", value=str(_root() / "examples" / "ashare_custom_strategy_example.py"), key="bt_custom")
                params["custom_class"] = st.text_input("类名（可空）", value="", key="bt_custom_cls")

        run = st.button("运行回测", key="bt_run")

    with col2:
        st.subheader("回测结果")
        result = st.session_state.get("ashare_backtest_result")
        if result:
            m = result["metrics"]
            mc = st.columns(4)
            mc[0].metric("期末净值（元）", "%.0f" % m["final_value"])
            mc[1].metric("总收益", "%.2f%%" % (100 * m["total_return"]))
            mc[2].metric("年化收益", "%.2f%%" % (100 * m["annual_return"]))
            mc[3].metric("夏普", "%.2f" % m["sharpe"])
            mc2 = st.columns(3)
            mc2[0].metric("波动率", "%.2f%%" % (100 * m["volatility"]))
            mc2[1].metric("最大回撤", "%.2f%%" % (100 * m["max_drawdown"]))
            mc2[2].metric("交易日数", "%d" % m["n_days"])
            nav = result.get("nav")
            if nav is not None and not nav.empty:
                fig = px.line(nav, x="date", y="nav", title="组合净值（人民币）")
                fig.update_layout(xaxis_title="日期", yaxis_title="净值（元）")
                st.plotly_chart(fig, use_container_width=True)
            wdf = result.get("weights")
            if wdf is not None and not wdf.empty:
                st.markdown("**目标权重（回测用，前 30）**")
                st.dataframe(wdf.head(30), use_container_width=True)
            st.caption(result.get("note", ""))
        else:
            st.info("配置左侧参数后点击「运行回测」。")

    if run:
        try:
            from src.strategies.ashare_strategy_runner import generate_weights, family_needs_industry
            from src.config.ashare_tdx import get_tdx_dir

            need_ind = family_needs_industry(family)
            cache_path = cache + ("_ind" if need_ind else "")
            ds = _load_cached_ds(cache_path)
            if ds is None and need_ind:
                # fall back to non-ind cache for families that can run without industry when cache missing
                ds = _load_cached_ds(cache)
            if ds is None:
                raise RuntimeError("缓存不存在：%s。请先到「数据管理」重建或「A股实盘」跑过一次。" % cache_path)

            with st.spinner("生成权重并回测…"):
                params = dict(params)
                params["tdx_dir"] = get_tdx_dir()
                if family == "custom":
                    params["path"] = params.pop("custom_path", "")
                    cls = params.pop("custom_class", "") or None
                    params["class_name"] = cls
                weights, meta = generate_weights(family, ds, params)
                if weights is None or weights.empty:
                    raise RuntimeError("策略未产出权重：%s" % meta)
                # normalize columns
                if "tic" not in weights.columns:
                    raise RuntimeError("权重缺少 tic 列")
                wcol = "weight" if "weight" in weights.columns else [c for c in weights.columns if c != "tic"][0]
                wdf = pd.DataFrame({
                    "tic": weights["tic"].astype(str).str.zfill(6),
                    "weight": pd.to_numeric(weights[wcol], errors="coerce"),
                }).dropna()
                s = wdf["weight"].sum()
                if s > 0:
                    wdf["weight"] = wdf["weight"] / s

                nav = _simulate_nav(
                    ds["prices"],
                    wdf,
                    initial_capital=float(initial_capital),
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                )
                if nav is None or nav.empty:
                    raise RuntimeError("净值序列为空：权重与行情无交集，或日期区间无数据")
                metrics = _backtest_metrics(nav)
                note = "策略元数据：%s；缓存：%s；说明：单期权重持有期内按日盯市（简化回测，非逐季换仓）。" % (meta, cache_path)
                payload = {
                    "metrics": metrics,
                    "nav": nav,
                    "weights": wdf,
                    "meta": meta,
                    "family": family,
                    "universe": uni,
                    "note": note,
                    "initial_capital": float(initial_capital),
                    "start": start_date.strftime("%Y-%m-%d"),
                    "end": end_date.strftime("%Y-%m-%d"),
                }
                st.session_state["ashare_backtest_result"] = payload
                # persist
                out_dir = _root() / "data"
                out_dir.mkdir(parents=True, exist_ok=True)
                wdf.to_csv(out_dir / "backtest_last_weights.csv", index=False, encoding="utf-8-sig")
                nav.to_csv(out_dir / "backtest_last_nav.csv", index=False, encoding="utf-8-sig")
                summary = {
                    "metrics": metrics,
                    "family": family,
                    "universe": uni,
                    "meta": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v) for k, v in (meta or {}).items()},
                    "start": payload["start"],
                    "end": payload["end"],
                    "initial_capital": payload["initial_capital"],
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                (out_dir / "backtest_last.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            st.success("回测完成")
            st.rerun()
        except Exception as e:
            st.error("回测失败：%s" % e)
            logger.exception("ashare backtest failed")


def show_portfolio_analysis_ashare() -> None:
    st.header("绩效分析（A股）")
    st.caption("读取最近一次 A 股回测/权重结果；单位人民币，代码为 6 位 A 股。")

    # hydrate from disk if session empty
    if "ashare_backtest_result" not in st.session_state:
        nav_p = _root() / "data" / "backtest_last_nav.csv"
        w_p = _root() / "data" / "backtest_last_weights.csv"
        j_p = _root() / "data" / "backtest_last.json"
        if nav_p.is_file() and w_p.is_file():
            try:
                nav = pd.read_csv(nav_p, parse_dates=["date"])
                wdf = _load_weights_csv(w_p)
                meta = {}
                family = universe = ""
                metrics = _backtest_metrics(nav)
                if j_p.is_file():
                    summary = json.loads(j_p.read_text(encoding="utf-8"))
                    metrics = summary.get("metrics", metrics)
                    family = summary.get("family", "")
                    universe = summary.get("universe", "")
                    meta = summary.get("meta", {})
                st.session_state["ashare_backtest_result"] = {
                    "metrics": metrics,
                    "nav": nav,
                    "weights": wdf,
                    "meta": meta,
                    "family": family,
                    "universe": universe,
                    "note": "从 data/backtest_last_* 载入",
                }
            except Exception as e:
                st.warning("载入本地回测结果失败：%s" % e)

    result = st.session_state.get("ashare_backtest_result")
    tab1, tab2, tab3, tab4 = st.tabs(["收益表现", "风险分析", "持仓归因", "基准对比"])

    if not result:
        st.info("还没有回测结果。请先到「策略回测」运行，或在「A股实盘」生成权重后回到本页查看持仓结构。")
        wpath = _latest_weights_path()
        if wpath:
            try:
                wdf = _load_weights_csv(wpath)
                st.subheader("最近权重文件：%s" % wpath.name)
                st.dataframe(wdf.sort_values("weight", ascending=False).head(50), use_container_width=True)
                fig = px.pie(wdf.head(20), names="tic", values="weight", title="权重前 20（示意）")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))
        return

    m = result["metrics"]
    nav = result.get("nav")
    wdf = result.get("weights")

    with tab1:
        st.write("策略：`%s`　股票池：`%s`" % (result.get("family", "-"), result.get("universe", "-")))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总收益", "%.2f%%" % (100 * m.get("total_return", 0)))
        c2.metric("年化收益", "%.2f%%" % (100 * m.get("annual_return", 0)))
        c3.metric("波动率", "%.2f%%" % (100 * m.get("volatility", 0)))
        c4.metric("夏普", "%.2f" % m.get("sharpe", 0))
        if nav is not None and not nav.empty:
            fig = px.line(nav, x="date", y="nav", title="组合净值（元）")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if nav is not None and not nav.empty:
            rets = nav["nav"].pct_change().dropna()
            cum = (1.0 + rets).cumprod()
            dd = cum / cum.cummax() - 1.0
            fig = px.line(x=nav["date"].iloc[1:len(dd)+1] if len(nav) > len(dd) else nav["date"].iloc[1:], y=dd.values, title="回撤")
            fig.update_layout(yaxis_tickformat=".1%", xaxis_title="日期", yaxis_title="回撤")
            st.plotly_chart(fig, use_container_width=True)
            var95 = float(np.percentile(rets, 5)) if len(rets) else 0.0
            cvar95 = float(rets[rets <= var95].mean()) if len(rets) else 0.0
            c1, c2, c3 = st.columns(3)
            c1.metric("最大回撤", "%.2f%%" % (100 * m.get("max_drawdown", 0)))
            c2.metric("VaR 95%", "%.2f%%" % (100 * var95))
            c3.metric("CVaR 95%", "%.2f%%" % (100 * cvar95))
        else:
            st.info("无净值序列")

    with tab3:
        if wdf is not None and not wdf.empty:
            show = wdf.sort_values("weight", ascending=False).copy()
            show["weight_pct"] = show["weight"] * 100
            st.dataframe(show.head(50), use_container_width=True)
            fig = px.bar(show.head(25), x="tic", y="weight", title="权重贡献（前 25）")
            fig.update_layout(xaxis_title="代码", yaxis_title="权重", yaxis_tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无权重表")

    with tab4:
        st.subheader("相对沪深300等权示意")
        st.caption("基准为缓存内全部股票等权净值（非官方沪深300指数点位）。")
        presets = _universe_presets()
        ds = _load_cached_ds(presets["沪深300"][1])
        if nav is None or nav.empty or ds is None:
            st.info("需要回测净值与沪深300缓存才能对比。")
        else:
            try:
                wide = _prices_to_wide(ds["prices"])
                # align dates
                nav2 = nav.copy()
                nav2["date"] = pd.to_datetime(nav2["date"])
                start, end = nav2["date"].min(), nav2["date"].max()
                wide = wide.loc[(wide.index >= start) & (wide.index <= end)].ffill().dropna(how="all")
                # equal weight of available names
                er = wide.pct_change().fillna(0.0).mean(axis=1)
                bench_nav = float(nav2["nav"].iloc[0]) * np.cumprod(1.0 + er.reindex(nav2["date"]).fillna(0.0).values)
                bench_nav[0] = float(nav2["nav"].iloc[0])
                cmp_df = pd.DataFrame({
                    "date": nav2["date"],
                    "组合": nav2["nav"].values,
                    "缓存等权基准": bench_nav,
                })
                fig = px.line(cmp_df, x="date", y=["组合", "缓存等权基准"], title="组合 vs 缓存等权基准（元）")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning("基准对比失败：%s" % e)


def show_settings_ashare() -> None:
    st.header("设置（A股）")
    cfg = _read_user_config()
    tdx_cfg = dict(cfg.get("TDX_CONFIG") or {})
    cls_cfg = dict(cfg.get("CLS_CONFIG") or {})
    broker_cfg = dict(cfg.get("BROKER_CONFIG") or {})

    tab1, tab2, tab3, tab4 = st.tabs(["通达信与路径", "券商", "财联社 / OpenAI", "日志"])

    with tab1:
        st.subheader("通达信")
        try:
            from src.config.ashare_tdx import get_tdx_dir, save_tdx_dir, validate_tdx_dir
            cur = tdx_cfg.get("tdx_dir") or get_tdx_dir()
        except Exception:
            cur = tdx_cfg.get("tdx_dir") or r"D:\dgzq"
            save_tdx_dir = None
            validate_tdx_dir = None
        tdx_dir = st.text_input("通达信目录", value=str(cur), key="set_tdx")
        data_dir = st.text_input("数据目录", value=str(_root() / "data"), key="set_data_dir")
        if st.button("保存路径设置", key="set_save_paths"):
            try:
                if save_tdx_dir:
                    save_tdx_dir(tdx_dir)
                else:
                    cfg.setdefault("TDX_CONFIG", {})["tdx_dir"] = tdx_dir
                    _write_user_config(cfg)
                if validate_tdx_dir:
                    try:
                        validate_tdx_dir(tdx_dir, require_vipdoc=True)
                        st.success("已保存，通达信目录校验通过")
                    except Exception as e:
                        st.warning("已保存，校验未通过：%s" % e)
                else:
                    st.success("已保存")
            except Exception as e:
                st.error(str(e))
        st.caption("环境变量 TDX_DIR 优先于本页写入前的默认值；保存后写入 user_config.json。")
        st.text_input("数据目录（只读展示）", value=data_dir, disabled=True)

    with tab2:
        st.subheader("券商路径（仅保存配置，本页不下单）")
        ths = st.text_input("同花顺下单程序", value=broker_cfg.get("ths_path", r"D:\同花顺软件\同花顺\xiadan.exe"), key="set_ths")
        xq = st.text_input("雪球配置 json", value=broker_cfg.get("xq_json", r".\xq.json"), key="set_xq")
        qmt = st.text_input("miniQMT userdata_mini", value=broker_cfg.get("qmt_path", r"D:\GJQMT\userdata_mini"), key="set_qmt")
        if st.button("保存券商路径", key="set_save_broker"):
            cfg["BROKER_CONFIG"] = {"ths_path": ths, "xq_json": xq, "qmt_path": qmt}
            _write_user_config(cfg)
            st.success("已写入 user_config.json 的 BROKER_CONFIG")
        st.info("实盘下单只在「A股实盘」页勾选确认后执行。Alpaca 美股接口不再作为主配置。")

    with tab3:
        st.subheader("财联社")
        page_size = st.number_input("每页条数", min_value=10, max_value=100, value=int(cls_cfg.get("page_size", 50)), key="set_cls_ps")
        max_pages = st.number_input("拉取页数", min_value=1, max_value=20, value=int(cls_cfg.get("max_pages", 3)), key="set_cls_mp")
        cache_dir = st.text_input("新闻缓存目录", value=str(cls_cfg.get("cache_dir", _root() / "data" / "cls_news")), key="set_cls_cache")
        openai_model = st.text_input("OpenAI 模型（情绪标注）", value=str(cls_cfg.get("openai_model", "gpt-4o-mini")), key="set_cls_model")
        if st.button("保存财联社设置", key="set_save_cls"):
            cfg["CLS_CONFIG"] = {
                "enabled": bool(cls_cfg.get("enabled", True)),
                "page_size": int(page_size),
                "max_pages": int(max_pages),
                "openai_model": openai_model,
                "cache_dir": cache_dir,
            }
            _write_user_config(cfg)
            st.success("已保存 CLS_CONFIG")

        st.subheader("OpenAI API Key")
        st.caption("用于财联社情绪 GPT 标注。不会在页面回显已有密钥。可写系统环境变量 OPENAI_API_KEY，或点下方写入用户环境（当前进程+用户级）。")
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        st.write("当前进程是否已有 OPENAI_API_KEY：%s" % ("是" if has_key else "否"))
        new_key = st.text_input("设置 OPENAI_API_KEY", type="password", value="", key="set_openai")
        if st.button("应用到当前进程", key="set_openai_apply"):
            if not new_key.strip():
                st.warning("未填写，未修改")
            else:
                os.environ["OPENAI_API_KEY"] = new_key.strip()
                st.success("已写入当前 Python 进程环境（刷新页面后若由新进程启动需再设或用系统环境变量）")
        st.caption("美股 Alpaca / FMP Key 不再作为 A 股主路径；如仍需可自行在系统环境配置，本页不收集。")

    with tab4:
        level = st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1, key="set_log")
        if st.button("应用日志级别", key="set_log_btn"):
            logging.getLogger().setLevel(getattr(logging, level))
            st.success("根日志级别：%s" % level)
