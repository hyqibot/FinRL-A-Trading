# -*- coding: utf-8 -*-
"""
通达信 gbbq（股本变迁）读取与市值计算。

gbbq 路径：<tdx_dir>/T0002/hq_cache/gbbq（tdx_dir 经 src.config.ashare_tdx 解析，可配置）。

单位约定（与本机通达信目录实测一致）：
  - SHARE_CATEGORIES 记录中四个浮点字段按「前流通 / 前总股本 / 后流通 / 后总股本」解读，单位为「万股」。
  - 股数 = 万股 * 10000；市值(元) = 收盘价 * 万股 * 10000。
  - category==1 为除权除息，字段含义不同，不参与股本/市值。

依赖：pytdx（GbbqReader）。若未安装请：
  python -m pip install "pytdx>=1.72" -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("tdx_gbbq")

# 股本相关类别（非除权除息）。pytdx 列名沿用分红口径，对本类事件按股本解读。
SHARE_CATEGORIES = {2, 3, 5, 7, 8, 9, 10}
WAN_TO_SHARES = 10000.0

# 内存缓存：path -> (mtime, events_df)
_EVENTS_CACHE: dict = {}


def resolve_gbbq_path(tdx_dir: Optional[str] = None) -> Path:
    """Resolve gbbq file under configured Tongdaxin root. Raises FileNotFoundError if missing."""
    from src.config.ashare_tdx import validate_tdx_dir

    root = Path(validate_tdx_dir(tdx_dir, require_vipdoc=False))
    path = root / "T0002" / "hq_cache" / "gbbq"
    if not path.is_file():
        raise FileNotFoundError(
            "未找到通达信股本文件 gbbq: {p}\n"
            "请确认「通达信目录」指向安装根目录（其下应有 T0002\\hq_cache\\gbbq），"
            "当前解析根目录为: {r}".format(p=str(path), r=str(root))
        )
    return path


def _require_gbbq_reader():
    try:
        from pytdx.reader.gbbq_reader import GbbqReader
    except ImportError as e:
        raise ImportError(
            "读取 gbbq 需要 pytdx。请在 FinRL-Trading 虚拟环境中执行:\n"
            '  python -m pip install "pytdx>=1.72" -i https://pypi.tuna.tsinghua.edu.cn/simple\n'
            "原始错误: %s" % e
        ) from e
    return GbbqReader


def load_share_events(
    tdx_dir: Optional[str] = None,
    *,
    gbbq_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    读取 gbbq 中股本相关事件，返回标准化表：
      code, date, circ_before, total_before, circ_after, total_after  （万股）
    缓存键：文件路径 + mtime。
    """
    path = Path(gbbq_path) if gbbq_path else resolve_gbbq_path(tdx_dir)
    if not path.is_file():
        raise FileNotFoundError("gbbq 不存在: %s" % path)
    mtime = path.stat().st_mtime
    key = (str(path.resolve()), mtime)
    if use_cache and key in _EVENTS_CACHE:
        return _EVENTS_CACHE[key].copy()

    cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache_gbbq"
    cache_dir.mkdir(parents=True, exist_ok=True)
    disk_name = "events_m{:.0f}.parquet".format(mtime)
    disk_path = cache_dir / disk_name
    if use_cache and disk_path.is_file():
        try:
            ev = pd.read_parquet(disk_path)
            _EVENTS_CACHE[key] = ev
            logger.info("命中 gbbq 事件缓存 %s (%d 行)", disk_path, len(ev))
            return ev.copy()
        except Exception as e:
            logger.warning("gbbq 磁盘缓存读取失败，将重解析: %s", e)

    GbbqReader = _require_gbbq_reader()
    logger.info("解析 gbbq: %s", path)
    raw = GbbqReader().get_df(str(path))
    if raw is None or raw.empty:
        raise RuntimeError("gbbq 为空或解析失败: %s" % path)

    df = raw[raw["category"].isin(SHARE_CATEGORIES)].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    # pytdx 列名按分红命名；股本类事件按前/后流通与总股本解读（万股）
    df["circ_before"] = pd.to_numeric(df["hongli_panqianliutong"], errors="coerce")
    df["total_before"] = pd.to_numeric(df["peigujia_qianzongguben"], errors="coerce")
    df["circ_after"] = pd.to_numeric(df["songgu_qianzongguben"], errors="coerce")
    df["total_after"] = pd.to_numeric(df["peigu_houzongguben"], errors="coerce")
    ev = (
        df[["code", "date", "circ_before", "total_before", "circ_after", "total_after"]]
        .sort_values(["code", "date"])
        .drop_duplicates(["code", "date"], keep="last")
        .reset_index(drop=True)
    )
    if use_cache:
        try:
            for old in cache_dir.glob("events_m*.parquet"):
                if old.name != disk_name:
                    try:
                        old.unlink()
                    except Exception:
                        pass
            ev.to_parquet(disk_path)
        except Exception as e:
            logger.warning("gbbq 磁盘缓存写入失败: %s", e)
        _EVENTS_CACHE[key] = ev
    logger.info("gbbq 股本事件 %d 行, %d 只代码", len(ev), ev["code"].nunique())
    return ev.copy()


def shares_asof_table(
    events: pd.DataFrame,
    codes: Sequence[str],
    as_of: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    对给定代码，取 as_of（默认=该代码事件最大日）时点的流通/总股本（万股）。
    规则：取 date<=as_of 的最近一条事件的「后」股本；
         若无此前事件，则用该代码最早事件的「前」股本。
    """
    codes = [str(c).zfill(6) for c in codes]
    if not codes:
        return pd.DataFrame(columns=["tic", "as_of", "circ_shares_wan", "total_shares_wan"])
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else None
    rows = []
    grouped = {k: g for k, g in events[events["code"].isin(codes)].groupby("code", sort=False)}
    for code in codes:
        sub = grouped.get(code)
        if sub is None or sub.empty:
            continue
        sub = sub.sort_values("date")
        if as_of_ts is None:
            cut = sub
            use_asof = sub["date"].iloc[-1]
        else:
            cut = sub[sub["date"] <= as_of_ts]
            use_asof = as_of_ts
        if not cut.empty:
            last = cut.iloc[-1]
            circ = float(last["circ_after"]) if pd.notna(last["circ_after"]) else np.nan
            total = float(last["total_after"]) if pd.notna(last["total_after"]) else np.nan
        else:
            first = sub.iloc[0]
            circ = float(first["circ_before"]) if pd.notna(first["circ_before"]) else np.nan
            total = float(first["total_before"]) if pd.notna(first["total_before"]) else np.nan
            use_asof = as_of_ts if as_of_ts is not None else first["date"]
        rows.append(
            {
                "tic": code,
                "as_of": pd.Timestamp(use_asof),
                "circ_shares_wan": circ,
                "total_shares_wan": total,
            }
        )
    return pd.DataFrame(rows)


def attach_shares_to_prices(
    prices: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    将 as-of 股本（万股）按 merge_asof 并入行情面板，并计算市值列（元）：
      total_mv = close * total_shares_wan * 10000
      circ_mv  = close * circ_shares_wan * 10000
    """
    if prices is None or prices.empty:
        out = prices.copy() if prices is not None else pd.DataFrame()
        for c in ("circ_shares_wan", "total_shares_wan", "circ_mv", "total_mv"):
            if c not in out.columns:
                out[c] = np.nan
        return out

    px = prices.copy()
    px["tic"] = px["tic"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    px = px.sort_values(["tic", "date"]).reset_index(drop=True)

    ev = events.copy()
    ev["tic"] = ev["code"].astype(str).str.zfill(6)
    ev["date"] = pd.to_datetime(ev["date"]).astype("datetime64[ns]")
    ev = ev.sort_values(["tic", "date"]).reset_index(drop=True)

    anchors = []
    for tic, sub in ev.groupby("tic", sort=False):
        first = sub.iloc[0]
        anchors.append(
            {
                "tic": tic,
                "date": first["date"] - pd.Timedelta(days=1),
                "circ_shares_wan": first["circ_before"],
                "total_shares_wan": first["total_before"],
            }
        )
    after = ev.rename(
        columns={
            "circ_after": "circ_shares_wan",
            "total_after": "total_shares_wan",
        }
    )[["tic", "date", "circ_shares_wan", "total_shares_wan"]]
    share_ts = pd.concat([pd.DataFrame(anchors), after], ignore_index=True)
    share_ts["date"] = pd.to_datetime(share_ts["date"]).astype("datetime64[ns]")
    share_ts = share_ts.sort_values(["tic", "date"]).drop_duplicates(["tic", "date"], keep="last")

    parts = []
    share_groups = {k: g for k, g in share_ts.groupby("tic", sort=False)}
    for tic, gpx in px.groupby("tic", sort=False):
        gpx = gpx.sort_values("date")
        gev = share_groups.get(tic)
        if gev is None or gev.empty:
            g = gpx.copy()
            g["circ_shares_wan"] = np.nan
            g["total_shares_wan"] = np.nan
        else:
            gev = gev.sort_values("date")
            g = pd.merge_asof(
                gpx,
                gev[["date", "circ_shares_wan", "total_shares_wan"]],
                on="date",
                direction="backward",
            )
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["total_mv"] = out["close"] * out["total_shares_wan"] * WAN_TO_SHARES
    out["circ_mv"] = out["close"] * out["circ_shares_wan"] * WAN_TO_SHARES
    for col in ("total_mv", "circ_mv"):
        out.loc[out[col].isna() | (out[col] <= 0), col] = np.nan
    return out.sort_values(["tic", "date"]).reset_index(drop=True)


def latest_market_caps(
    codes: Sequence[str],
    prices: pd.DataFrame,
    *,
    tdx_dir: Optional[str] = None,
    as_of: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """返回每只股票在 as_of（默认=行情末日）的 close / 股本(万股) / 总市值 / 流通市值。"""
    codes = [str(c).zfill(6) for c in codes]
    empty_cols = ["tic", "date", "close", "circ_shares_wan", "total_shares_wan", "circ_mv", "total_mv"]
    if not codes:
        return pd.DataFrame(columns=empty_cols)
    px = prices.copy()
    px["tic"] = px["tic"].astype(str).str.zfill(6)
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["tic"].isin(codes)]
    if as_of is not None:
        px = px[px["date"] <= pd.Timestamp(as_of)]
    if px.empty:
        return pd.DataFrame(columns=empty_cols)

    if {"total_mv", "circ_mv"}.issubset(px.columns) and px["total_mv"].notna().any():
        idx = px.sort_values("date").groupby("tic", sort=False).tail(1)
        cols = [c for c in empty_cols if c in idx.columns]
        return idx[cols].reset_index(drop=True)

    events = load_share_events(tdx_dir=tdx_dir)
    last_px = px.sort_values("date").groupby("tic", sort=False).tail(1).copy()
    asof_per = last_px.set_index("tic")["date"].to_dict()
    rows = []
    for _, r in last_px.iterrows():
        tic = r["tic"]
        close = float(r["close"]) if pd.notna(r["close"]) else np.nan
        one = shares_asof_table(events, [tic], as_of=asof_per[tic])
        if one.empty:
            circ_w = total_w = np.nan
        else:
            circ_w = float(one.iloc[0]["circ_shares_wan"]) if pd.notna(one.iloc[0]["circ_shares_wan"]) else np.nan
            total_w = float(one.iloc[0]["total_shares_wan"]) if pd.notna(one.iloc[0]["total_shares_wan"]) else np.nan
        total_mv = (
            close * total_w * WAN_TO_SHARES
            if pd.notna(close) and pd.notna(total_w) and total_w > 0
            else np.nan
        )
        circ_mv = (
            close * circ_w * WAN_TO_SHARES
            if pd.notna(close) and pd.notna(circ_w) and circ_w > 0
            else np.nan
        )
        if pd.notna(total_mv) and total_mv <= 0:
            total_mv = np.nan
        if pd.notna(circ_mv) and circ_mv <= 0:
            circ_mv = np.nan
        rows.append(
            {
                "tic": tic,
                "date": r["date"],
                "close": close,
                "circ_shares_wan": circ_w,
                "total_shares_wan": total_w,
                "circ_mv": circ_mv,
                "total_mv": total_mv,
            }
        )
    return pd.DataFrame(rows)


def enrich_dataset_market_cap(
    ds: dict,
    *,
    tdx_dir: Optional[str] = None,
    attach_to_prices: bool = True,
) -> dict:
    """
    为 dataset 附加市值：prices 增加 circ_mv/total_mv；并生成 market_caps 摘要表。
    fundamentals 若存在，按 datadate as-of 附加 total_mv/circ_mv。
    """
    out = dict(ds)
    prices = out.get("prices")
    if prices is None or prices.empty:
        return out
    events = load_share_events(tdx_dir=tdx_dir)
    if attach_to_prices:
        out["prices"] = attach_shares_to_prices(prices, events)
    codes = out["prices"]["tic"].astype(str).str.zfill(6).unique().tolist()
    out["market_caps"] = latest_market_caps(codes, out["prices"], tdx_dir=tdx_dir)

    fund = out.get("fundamentals")
    if fund is not None and not fund.empty and "datadate" in fund.columns:
        f = fund.copy()
        if "tic" in f.columns:
            f["tic"] = f["tic"].astype(str).str.zfill(6)
        else:
            f["tic"] = f["gvkey"].astype(str).str.zfill(6)
        f["datadate"] = pd.to_datetime(f["datadate"]).astype("datetime64[ns]")
        px = out["prices"]
        if "total_mv" in px.columns:
            parts = []
            px2 = px[["tic", "date", "total_mv", "circ_mv"]].dropna(subset=["total_mv"]).sort_values(["tic", "date"])
            for tic, gf in f.groupby("tic", sort=False):
                gp = px2[px2["tic"] == tic]
                if gp.empty:
                    g = gf.copy()
                    g["total_mv"] = np.nan
                    g["circ_mv"] = np.nan
                else:
                    left = gf.sort_values("datadate").rename(columns={"datadate": "date"})
                    merged = pd.merge_asof(
                        left,
                        gp[["date", "total_mv", "circ_mv"]],
                        on="date",
                        direction="backward",
                    )
                    merged = merged.rename(columns={"date": "datadate"})
                    g = merged
                parts.append(g)
            out["fundamentals"] = pd.concat(parts, ignore_index=True)
    return out
