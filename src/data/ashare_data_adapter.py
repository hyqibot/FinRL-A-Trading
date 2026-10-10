# -*- coding: utf-8 -*-
"""
A股数据适配层 for FinRL-Trading
================================
通达信本地行情 + akshare 多源基本面，产出 FinRL 策略层
直接可用的 fundamentals / prices 两张表，策略代码一行不改。

数据来源（多源，不偷工减料）:
  - 历史价: 通达信本地日线 (TDX .day, <tdx_dir>/vipdoc; tdx_dir 见 src.config.ashare_tdx)  —— 通达信 .day 定长解析
  - 完整基本面: akshare 新浪接口 stock_financial_analysis_indicator (86字段/季)为主力
               东方财富 stock_financial_analysis_indicator_em (若该版akshare支持) 补缺
               同花顺 stock_financial_abstract_ths 兜底
  - 估值 pe/pb/ps/peg/股息率: 用通达信季末价 × 财报自算
  - 行业 gsector: akshare 行业板块成分
  - y_return: 用通达信季末对齐价算 ln(下季/本季)，与 FinRL 原逻辑一致

FinRL ML 策略契约（已核对 src/strategies/ml_strategy.py）:
  data['fundamentals'] 必含: gvkey, datadate, y_return, 以及若干数值特征列
     (特征 = 所有数值列，排除 gvkey/tic/gsector/datadate/y_return/prccd/ajexdi/adj_close/adj_close_q)
     可选: gsector / sector（行业中性）、adj_close_q（最小方差季度价）
  data['prices'] (可选): ['date','tic'/'gvkey','close']，用于最小方差与当日gap调整
"""
import os
import math
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import akshare as ak
except Exception as _e:  # pragma: no cover
    ak = None

logger = logging.getLogger("ashare_adapter")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# TDX root: override arg > env TDX_DIR > user_config.json TDX_CONFIG.tdx_dir > default D:\dgzq
from src.config.ashare_tdx import get_tdx_dir, validate_tdx_dir  # noqa: E402

def _resolved_tdx_dir(tdx_dir=None):
    return get_tdx_dir(tdx_dir)

import time as _time


def _retry(fn, tries=4, delay=2.0, backoff=1.6, label=""):
    """akshare 接口偶发限流/瞬断，自动重试+退避。"""
    last = None
    d = delay
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            logger.debug(f"重试 {label} ({i+1}/{tries}): {msg[:80]}")
            _time.sleep(d)
            d *= backoff
    if last:
        raise last


# =====================================================================
# 1) 通达信本地日线读取（纯函数解析 .day）
# =====================================================================
def format_stock_code(code):
    code = str(code).zfill(6)
    if code.startswith(("688", "605", "603", "601", "600", "689", "787", "789")):
        return f"sh{code}"
    elif code.startswith(("301", "300", "003", "002", "001", "000")):
        return f"sz{code}"
    elif code.startswith(("430", "830", "831", "832", "833", "834", "835", "836",
                           "837", "838", "839", "870", "871", "872", "873", "920")):
        return f"bj{code}"
    return code


def get_market_path(code):
    code = str(code).zfill(6)
    if code.startswith(("688", "605", "603", "601", "600", "689", "787", "789")):
        return "sh"
    elif code.startswith(("301", "300", "003", "002", "001", "000")):
        return "sz"
    elif code.startswith(("430", "920")):
        return "bj"
    return "sh"


def read_tdx_day_file(file_path):
    """通达信日线 .day：每条 32 字节定长记录。"""
    try:
        with open(file_path, "rb") as f:
            buf = f.read()
    except Exception as e:
        logger.warning(f"读取失败 {file_path}: {e}")
        return None
    n = len(buf) // 32
    rows = []
    for i in range(n):
        p = i * 32
        d = int.from_bytes(buf[p:p + 4], "little")
        if d <= 0:
            continue
        o = int.from_bytes(buf[p + 4:p + 8], "little") / 100.0
        h = int.from_bytes(buf[p + 8:p + 12], "little") / 100.0
        low = int.from_bytes(buf[p + 12:p + 16], "little") / 100.0
        c = int.from_bytes(buf[p + 16:p + 20], "little") / 100.0
        amt = int.from_bytes(buf[p + 20:p + 24], "little")
        vol = int.from_bytes(buf[p + 24:p + 28], "little")
        rows.append((d, o, h, low, c, vol, amt))
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_tdx_daily(symbol, tdx_dir=None):
    """读取单只股票的完整通达信日线历史。"""
    tdx_dir = _resolved_tdx_dir(tdx_dir)
    code = str(symbol).zfill(6)
    path = os.path.join(tdx_dir, "vipdoc", get_market_path(code), "lday", f"{format_stock_code(code)}.day")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
    df = read_tdx_day_file(path)
    return df if df is not None else pd.DataFrame()


def get_price_panel(tics, start=None, end=None, tdx_dir=None):
    """返回长表 ['date','tic','close','open','high','low','volume']。"""
    frames = []
    for t in tics:
        df = get_tdx_daily(t, tdx_dir=tdx_dir)
        if df is None or df.empty:
            logger.warning(f"通达信无数据: {t}")
            continue
        df = df.copy()
        df["tic"] = str(t).zfill(6)
        frames.append(df[["date", "tic", "close", "open", "high", "low", "volume"]])
    if not frames:
        return pd.DataFrame(columns=["date", "tic", "close", "open", "high", "low", "volume"])
    panel = pd.concat(frames, ignore_index=True)
    if start:
        panel = panel[panel["date"] >= pd.to_datetime(start)]
    if end:
        panel = panel[panel["date"] <= pd.to_datetime(end)]
    return panel.sort_values(["tic", "date"]).reset_index(drop=True)


def quarter_end_price(price_df, tic, as_of):
    """某股票在 as_of（报告期末）当日或之前最近交易日的收盘价。"""
    sub = price_df[(price_df["tic"] == str(tic).zfill(6)) & (price_df["date"] <= pd.to_datetime(as_of))]
    if sub.empty:
        return np.nan
    return float(sub.sort_values("date")["close"].iloc[-1])


# =====================================================================
# 2) 多源基本面
# =====================================================================
# 新浪 86 字段中，挑选并重命名为干净英文（其余数值字段也会保留为特征，不丢数据）
SINA_RENAME = {
    "日期": "datadate",
    "摊薄每股收益(元)": "EPS",
    "加权每股收益(元)": "EPS_weighted",
    "扣除非经常性损益后的每股收益(元)": "EPS_deducted",
    "每股净资产_调整后(元)": "BPS",
    "每股净资产_调整前(元)": "BPS_unadj",
    "每股经营性现金流(元)": "ocf_per_share",
    "每股资本公积金(元)": "capital_reserve_ps",
    "每股未分配利润(元)": "undistributed_profit_ps",
    "总资产利润率(%)": "roa_profit",
    "主营业务利润率(%)": "main_op_margin",
    "总资产净利润率(%)": "roa",
    "成本费用利润率(%)": "cost_expense_margin",
    "营业利润率(%)": "operating_margin",
    "主营业务成本率(%)": "main_cost_ratio",
    "销售净利率(%)": "net_income_ratio",
    "股本报酬率(%)": "capital_return",
    "净资产报酬率(%)": "equity_return",
    "资产报酬率(%)": "asset_return",
    "销售毛利率(%)": "gross_margin",
    "三项费用比重": "three_expense_ratio",
    "非主营比重": "non_main_ratio",
    "主营利润比重": "main_profit_ratio",
    "股息发放率(%)": "dividend_payout",
    "投资收益率(%)": "investment_return",
    "净资产收益率(%)": "roe",
    "加权净资产收益率(%)": "roe_weighted",
    "扣除非经常性损益后的净利润(元)": "net_profit_deducted",
    "主营业务收入增长率(%)": "revenue_growth",
    "净利润增长率(%)": "net_profit_growth",
    "净资产增长率(%)": "equity_growth",
    "总资产增长率(%)": "asset_growth",
    "应收账款周转率(次)": "acc_rec_turnover",
    "应收账款周转天数(天)": "acc_rec_days",
    "存货周转天数(天)": "inventory_days",
    "存货周转率(次)": "inventory_turnover",
    "固定资产周转率(次)": "fixed_asset_turnover",
    "总资产周转率(次)": "asset_turnover",
    "总资产周转天数(天)": "asset_turnover_days",
    "流动资产周转率(次)": "cur_asset_turnover",
    "流动资产周转天数(天)": "cur_asset_days",
    "股东权益周转率(次)": "equity_turnover",
    "流动比率": "cur_ratio",
    "速动比率": "quick_ratio",
    "现金比率(%)": "cash_ratio",
    "利息支付倍数": "interest_coverage",
    "长期债务与营运资金比率(%)": "lt_debt_to_wc",
    "股东权益比率(%)": "equity_ratio",
    "长期负债比率(%)": "lt_debt_ratio",
    "股东权益与固定资产比率(%)": "equity_to_fixed_asset",
    "负债与所有者权益比率(%)": "debt_to_equity",
    "长期资产与长期资金比率(%)": "lt_asset_to_lt_fund",
    "资本化比率(%)": "capitalization_ratio",
    "固定资产净值率(%)": "fixed_asset_net_ratio",
    "资本固定化比率(%)": "capital_fixation_ratio",
    "产权比率(%)": "debt_to_equity_ratio",
    "清算价值比率(%)": "liquidation_value_ratio",
    "固定资产比重(%)": "fixed_asset_proportion",
    "资产负债率(%)": "debt_ratio",
    "总资产(元)": "total_assets",
    "经营现金净流量对销售收入比率(%)": "ocf_to_sales",
    "资产的经营现金流量回报率(%)": "ocf_return_on_assets",
    "经营现金净流量与净利润的比率(%)": "ocf_to_net_profit",
    "经营现金净流量对负债比率(%)": "ocf_to_debt",
    "现金流量比率(%)": "ocf_ratio",
}


def _to_num(s):
    return pd.to_numeric(
        pd.Series(s).astype(str).str.replace(",", "", regex=False).replace({"--": None, "": None, "nan": None}),
        errors="coerce",
    )


def _fetch_sina(symbol, start_year):
    """新浪财务指标分析：主力源，86字段/季。"""
    code = str(symbol).zfill(6)
    df = _retry(lambda: ak.stock_financial_analysis_indicator(symbol=code, start_year=str(start_year)), label=f"sina {code}")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={k: v for k, v in SINA_RENAME.items() if k in df.columns})
    # 所有非 id 列尽量转数值；datadate 转日期
    if "datadate" in df.columns:
        df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    for c in df.columns:
        if c == "datadate":
            continue
        df[c] = _to_num(df[c])
    df = df.dropna(subset=["datadate"]).copy()
    df["_src"] = "sina"
    return df


def _fetch_em(symbol):
    """东方财富财务分析指标（部分 akshare 版本才有）：补缺。"""
    if ak is None or not hasattr(ak, "stock_financial_analysis_indicator_em"):
        return pd.DataFrame()
    code = str(symbol).zfill(6)
    if code.startswith("920"):
        em = f"{code}.BJ"
    elif code.startswith(("5", "6", "9")):
        em = f"{code}.SH"
    elif code.startswith(("0", "3")):
        em = f"{code}.SZ"
    else:
        em = f"{code}.SZ"
    try:
        df = _retry(lambda: ak.stock_financial_analysis_indicator_em(symbol=em, indicator="按报告期"), tries=2, label=f"em {code}")
    except Exception as e:
        logger.debug(f"em 接口失败 {code}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 统一日期列
    for cand in ["REPORT_DATE", "报告期", "日期"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "datadate"})
            break
    if "datadate" not in df.columns:
        return pd.DataFrame()
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["datadate"]).copy()
    for c in df.columns:
        if c in ("datadate",):
            continue
        df[c] = _to_num(df[c])
    df["_src"] = "em"
    return df


def _fetch_ths(symbol):
    """同花顺财务摘要：兜底源。"""
    if ak is None or not hasattr(ak, "stock_financial_abstract_ths"):
        return pd.DataFrame()
    code = str(symbol).zfill(6)
    try:
        df = _retry(lambda: ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期"), tries=2, label=f"ths {code}")
    except Exception as e:
        logger.debug(f"ths 接口失败 {code}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    for cand in ["报告期", "日期"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "datadate"})
            break
    if "datadate" not in df.columns:
        return pd.DataFrame()
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")
    df = df.dropna(subset=["datadate"]).copy()
    for c in df.columns:
        if c in ("datadate",):
            continue
        df[c] = _to_num(df[c])
    df["_src"] = "ths"
    return df


def get_fundamentals_raw(symbol, start_year=2016):
    """多源基本面：新浪为主，东财/同花顺补缺（按 datadate 对齐，新浪缺的列用其它源填）。"""
    sina = _fetch_sina(symbol, start_year)
    base = sina.copy()
    for fetch in (_fetch_em, _fetch_ths):
        other = fetch(symbol)
        if other is None or other.empty:
            continue
        other = other.drop(columns=[c for c in ["_src"] if c in other.columns])
        if base.empty:
            base = other.copy()
            continue
        # 对齐 datadate，只拿 base 里没有的列来补
        new_cols = [c for c in other.columns if c not in base.columns and c != "datadate"]
        if not new_cols:
            continue
        base = base.merge(other[["datadate"] + new_cols], on="datadate", how="left")
    if base.empty:
        return base
    base["gvkey"] = str(symbol).zfill(6)
    base["tic"] = str(symbol).zfill(6)
    return base.sort_values("datadate").reset_index(drop=True)


# =====================================================================
# 3) 估值自算（价 × 财报），y_return，行业
# =====================================================================
def _fetch_dps(symbol):
    """用东财分红送配(stock_fhps_detail_em)取每股现金分红(元/股) -> {年份: 当年每股派息合计}。
    现金分红比例为“每10股派X元(含税)”，每股派息 = X/10；按报告期所属年份汇总（可含中期分红）。"""
    sym = str(symbol).zfill(6)
    try:
        df = _retry(lambda: ak.stock_fhps_detail_em(symbol=sym), label=f"fhps {sym}")
    except Exception as e:
        logger.warning(f"分红获取失败 {sym}: {e}")
        return {}
    if df is None or df.empty:
        return {}
    col_ratio = "现金分红-现金分红比例"
    col_rep = "报告期"
    if col_ratio not in df.columns or col_rep not in df.columns:
        return {}
    d = df[[col_rep, col_ratio]].copy()
    d[col_rep] = pd.to_datetime(d[col_rep], errors="coerce")
    d[col_ratio] = _to_num(d[col_ratio])
    d = d.dropna(subset=[col_rep])
    d["dps"] = d[col_ratio].fillna(0.0) / 10.0
    d["year"] = d[col_rep].dt.year
    d = d[d["dps"] > 0]  # 只保留真实有派息的报告期，避免“当年仅中期报告、未宣派年度分红”被算成0
    if d.empty:
        return {}
    return d.groupby("year")["dps"].sum().to_dict()


def add_quarter_price_and_valuations(fund_df, price_df, dps_map=None):
    """用通达信季末价补 adj_close_q，并自算 pe/pb/ps/peg/股息率。
    dps_map: {年份: 每股派息} 时按真实分红算股息率；否则回退到 股息发放率×EPS_ttm。"""
    if fund_df.empty:
        return fund_df
    df = fund_df.copy()
    df["adj_close_q"] = df.apply(lambda r: quarter_end_price(price_df, r["tic"], r["datadate"]), axis=1)

    def ttm(series_cum, dates):
        # 新浪每股指标为年内累计(YTD)，用累计差分还原单季再滚动4季得到TTM
        s = pd.Series(series_cum.values, index=pd.to_datetime(dates)).sort_index()
        yr = s.index.year
        single = s.groupby(yr).diff()
        # 每年Q1(无前值)的单季=累计值本身
        first_mask = s.groupby(yr).cumcount() == 0
        single[first_mask.values] = s[first_mask.values]
        return single.rolling(4).sum()

    # EPS TTM -> pe；BPS 为时点值 -> pb；营收/股暂用销售净利率反推不可靠，ps 用 市值/营收，缺营收则跳过
    if "EPS" in df.columns:
        eps_ttm = ttm(df["EPS"], df["datadate"]).reindex(pd.to_datetime(df["datadate"]).values)
        df["EPS_ttm"] = eps_ttm.values if hasattr(eps_ttm, "values") else eps_ttm
        df["pe"] = df["adj_close_q"] / df["EPS_ttm"].replace(0, np.nan)
    if "BPS" in df.columns:
        df["pb"] = df["adj_close_q"] / df["BPS"].replace(0, np.nan)
    # peg = pe / 净利润增长率(%)
    if "pe" in df.columns and "net_profit_growth" in df.columns:
        df["peg"] = df["pe"] / df["net_profit_growth"].replace(0, np.nan)
    # 股息率：优先用真实现金分红(每股派息/价)，取 datadate 年份“上一完整年度”的每股派息；缺则回退 股息发放率×EPS_ttm
    if dps_map:
        def _trailing_dps(dt):
            y = pd.to_datetime(dt).year
            # 上一年度分红通常在本年实施；取 <= 该年份的最近一个有派息的年份
            cand = [yy for yy in dps_map.keys() if yy <= y]
            if not cand:
                return np.nan
            return dps_map[max(cand)]
        dps_series = df["datadate"].apply(_trailing_dps)
        df["dps"] = dps_series.values
        df["dividend_yield"] = df["dps"] / df["adj_close_q"].replace(0, np.nan) * 100.0
    elif "dividend_payout" in df.columns and "EPS_ttm" in df.columns:
        df["dividend_yield"] = (df["dividend_payout"] / 100.0) * df["EPS_ttm"] / df["adj_close_q"].replace(0, np.nan) * 100.0
    # ps = 价 / 每股营收；每股营收≈ EPS_ttm / 销售净利率
    if "EPS_ttm" in df.columns and "net_income_ratio" in df.columns:
        rev_ps = df["EPS_ttm"] / (df["net_income_ratio"] / 100.0).replace(0, np.nan)
        df["ps"] = df["adj_close_q"] / rev_ps.replace(0, np.nan)
    return df


def add_y_return(fund_df):
    """y_return = ln(下一报告期 adj_close_q / 本期 adj_close_q)，按股票分组。"""
    if fund_df.empty or "adj_close_q" not in fund_df.columns:
        fund_df["y_return"] = np.nan
        return fund_df
    df = fund_df.sort_values(["gvkey", "datadate"]).copy()
    df["_next_px"] = df.groupby("gvkey")["adj_close_q"].shift(-1)
    df["y_return"] = np.log(df["_next_px"] / df["adj_close_q"].replace(0, np.nan))
    df = df.drop(columns=["_next_px"])
    return df


_INDUSTRY_CACHE = {}


def get_industry_map(tics):
    """用 akshare 行业板块成分构建 code->行业 映射（一次性，省调用）。"""
    global _INDUSTRY_CACHE
    want = set(str(t).zfill(6) for t in tics)
    if _INDUSTRY_CACHE and want.issubset(set(_INDUSTRY_CACHE.keys())):
        return {k: _INDUSTRY_CACHE[k] for k in want}
    mapping = {}
    try:
        boards = ak.stock_board_industry_name_em()
        name_col = "板块名称" if "板块名称" in boards.columns else boards.columns[0]
        for bname in boards[name_col].tolist():
            try:
                cons = ak.stock_board_industry_cons_em(symbol=bname)
                ccol = "代码" if "代码" in cons.columns else cons.columns[0]
                for code in cons[ccol].astype(str).str.zfill(6):
                    mapping[code] = bname
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"行业映射获取失败，将留空 gsector: {e}")
    _INDUSTRY_CACHE.update(mapping)
    return {t: mapping.get(t) for t in want}


# =====================================================================
# 4) 顶层组装
# =====================================================================
def build_dataset(tics, start_year=2016, start=None, end=None,
                  with_industry=True, fill_feature_na="median", tdx_dir=None, cache_path=None):
    """
    返回 {'fundamentals': df, 'prices': df}，可直接喂 FinRL 策略 generate_weights。
    fill_feature_na: 'median' | 'zero' | None  （ML 会对 X 的 NaN 行 dropna，故默认填充）
    """
    tics = [str(t).zfill(6) for t in tics]
    logger.info(f"构建数据集: {len(tics)} 只股票, start_year={start_year}")

    if cache_path and os.path.exists(cache_path + ".fund.parquet") and os.path.exists(cache_path + ".px.parquet"):
        try:
            f = pd.read_parquet(cache_path + ".fund.parquet")
            p = pd.read_parquet(cache_path + ".px.parquet")
            logger.info(f"命中缓存 {cache_path}: fundamentals {f.shape}, prices {p.shape}")
            return {"fundamentals": f, "prices": p}
        except Exception as e:
            logger.warning(f"缓存读取失败，重建: {e}")

    tdx_dir = validate_tdx_dir(tdx_dir)
    prices = get_price_panel(tics, start=start, end=end, tdx_dir=tdx_dir)
    logger.info(f"价格面板: {len(prices)} 行, {prices['tic'].nunique() if not prices.empty else 0} 只")

    fund_frames = []
    ind_map = get_industry_map(tics) if with_industry else {}
    for i, t in enumerate(tics, 1):
        try:
            f = get_fundamentals_raw(t, start_year=start_year)
        except Exception as e:
            logger.warning(f"基本面失败 {t}: {e}")
            continue
        if f is None or f.empty:
            continue
        try:
            dps_map = _fetch_dps(t)
        except Exception:
            dps_map = {}
        f = add_quarter_price_and_valuations(f, prices, dps_map=dps_map)
        if with_industry:
            f["gsector"] = ind_map.get(t)
        fund_frames.append(f)
        _time.sleep(0.4)
        if i % 25 == 0:
            logger.info(f"  基本面进度 {i}/{len(tics)}")

    if not fund_frames:
        return {"fundamentals": pd.DataFrame(), "prices": prices}

    fundamentals = pd.concat(fund_frames, ignore_index=True)
    fundamentals = add_y_return(fundamentals)

    # 清理：去掉辅助列
    for c in ["_src"]:
        if c in fundamentals.columns:
            fundamentals = fundamentals.drop(columns=[c])

    # 特征 NaN 处理（与 ML 的 dropna(X) 兼容）
    # 兜底：把任何残留的 object 列（除 id 外）尽量转数值，保证可缓存且可作特征
    for c in fundamentals.columns:
        if c in {"gvkey", "tic", "gsector", "datadate"}:
            continue
        if not pd.api.types.is_numeric_dtype(fundamentals[c]):
            fundamentals[c] = _to_num(fundamentals[c])
    id_like = {"gvkey", "tic", "gsector", "datadate", "y_return", "adj_close_q"}
    feat_cols = [c for c in fundamentals.columns
                 if c not in id_like and pd.api.types.is_numeric_dtype(fundamentals[c])]
    if fill_feature_na == "median":
        for c in feat_cols:
            med = fundamentals[c].median()
            fundamentals[c] = fundamentals[c].fillna(med if pd.notna(med) else 0.0)
    elif fill_feature_na == "zero":
        fundamentals[feat_cols] = fundamentals[feat_cols].fillna(0.0)

    fundamentals = fundamentals.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    logger.info(f"基本面表: {len(fundamentals)} 行, {fundamentals['gvkey'].nunique()} 只, "
                f"{len(feat_cols)} 个特征列, 日期 {fundamentals['datadate'].min()} ~ {fundamentals['datadate'].max()}")
    if cache_path:
        try:
            fundamentals.to_parquet(cache_path + ".fund.parquet")
            prices.to_parquet(cache_path + ".px.parquet")
            logger.info(f"已写缓存 {cache_path}")
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")
    return {"fundamentals": fundamentals, "prices": prices}



def attach_market_cap(ds, tdx_dir=None):
    """Attach total_mv/circ_mv from TDX gbbq onto dataset."""
    from src.data.tdx_gbbq import enrich_dataset_market_cap
    return enrich_dataset_market_cap(ds, tdx_dir=tdx_dir, attach_to_prices=True)


if __name__ == "__main__":
    import sys
    codes = sys.argv[1:] or ["600519", "000001", "600036"]
    ds = build_dataset(codes, start_year=2018)
    f = ds["fundamentals"]
    print("fundamentals shape:", f.shape)
    print("cols:", list(f.columns))
    if not f.empty:
        print(f[["gvkey", "datadate", "adj_close_q", "EPS", "pe", "pb", "roe", "y_return"]].tail(12).to_string())
