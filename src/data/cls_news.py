# -*- coding: utf-8 -*-
"""财联社电报新闻拉取 + 情绪标注（A股专用，不走 FMP）。

接口与签名逻辑对齐 NEWStodo.md / daily_stock_analysis news_flow_sources：
  - 主：https://www.cls.cn/nodeapi/telegraphList  (NEWStodo)
  - 备：https://www.cls.cn/v1/roll/get_roll_list (daily_stock_analysis)

配置（与 TDX 相同风格，写入项目根 user_config.json）：
  CLS_CONFIG:
    enabled: true
    page_size: 50
    max_pages: 3
    openai_model: null
  环境变量：OPENAI_API_KEY / OPENAI_MODEL（可选；无 Key 时用词典规则兜底）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger("cls_news")

CN_TZ = timezone(timedelta(hours=8))
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
CLS_NODEAPI = "https://www.cls.cn/nodeapi/telegraphList"
CLS_ROLL = "https://www.cls.cn/v1/roll/get_roll_list"
CLS_CONFIG_KEY = "CLS_CONFIG"
POS_WORDS = ("利好", "增长", "上涨", "突破", "创新高", "加仓", "增持", "盈利", "超预期", "回暖", "提振")
NEG_WORDS = ("利空", "下跌", "暴跌", "亏损", "减持", "处罚", "立案", "风险", "爆雷", "下滑", "不及预期", "退市")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _user_config_path() -> Path:
    return _project_root() / "user_config.json"


def _read_user_config() -> Dict[str, Any]:
    path = _user_config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("read user_config failed: %s", e)
        return {}


def _resolve_cache_dir(raw: Optional[str] = None) -> str:
    """Always return an absolute cache dir under the project when relative."""
    root = _project_root()
    default = root / "data" / "cls_news"
    if not raw:
        return str(default)
    p = Path(str(raw))
    if not p.is_absolute():
        p = root / p
    return str(p.resolve())


def get_cls_config() -> Dict[str, Any]:
    data = _read_user_config()
    block = data.get(CLS_CONFIG_KEY)
    if not isinstance(block, dict):
        block = {}
    return {
        "enabled": bool(block.get("enabled", True)),
        "page_size": int(block.get("page_size", 50) or 50),
        "max_pages": int(block.get("max_pages", 3) or 3),
        "openai_model": block.get("openai_model") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
        "cache_dir": _resolve_cache_dir(block.get("cache_dir")),
    }


def save_cls_config(**kwargs) -> Path:
    path = _user_config_path()
    data = _read_user_config()
    block = data.get(CLS_CONFIG_KEY)
    if not isinstance(block, dict):
        block = {}
    for k, v in kwargs.items():
        if v is not None:
            block[k] = v
    # Keep cache_dir absolute so Streamlit cwd changes do not lose results
    block["cache_dir"] = _resolve_cache_dir(block.get("cache_dir"))
    data[CLS_CONFIG_KEY] = block
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _sign(params: Dict[str, str]) -> str:
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return hashlib.md5(hashlib.sha1(qs.encode()).hexdigest().encode()).hexdigest()


def _as_text(value: object, limit: int = 4000) -> str:
    text = "" if value is None else str(value)
    return text.replace("\r", " ").replace("\n", " ").strip()[:limit]


def _cn_time(stamp: object) -> str:
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        return ""
    return datetime.fromtimestamp(float(stamp), CN_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_rows(rows: List[dict], source: str) -> List[dict]:
    items = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = _as_text(item.get("title") or item.get("brief") or "")
        content = _as_text(item.get("content") or item.get("brief") or title)
        if not content and not title:
            continue
        items.append({
            "source": source,
            "title": title,
            "content": content,
            "time": _cn_time(item.get("ctime")),
            "ctime": item.get("ctime"),
            "url": _as_text(item.get("shareurl") or item.get("url") or "", 500),
            "importance": item.get("level") or item.get("importance"),
        })
    return items


def fetch_cls_telegraph_nodeapi(rn: int = 50, last_time: Optional[int] = None, timeout: float = 10) -> List[dict]:
    if last_time is None:
        last_time = int(time.time())
    params = {
        "app": "CailianpressWeb",
        "os": "web",
        "rn": str(max(1, min(int(rn), 100))),
        "sv": "7.7.5",
        "last_time": str(int(last_time)),
    }
    sign = _sign(params)
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    url = "%s?%s&sign=%s" % (CLS_NODEAPI, qs, sign)
    r = requests.get(
        url,
        headers={"User-Agent": UA, "Referer": "https://www.cls.cn/telegraph"},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    rows = (payload.get("data") or {}).get("roll_data") or []
    if not isinstance(rows, list):
        raise RuntimeError("cls nodeapi roll_data is not a list")
    return _normalize_rows(rows, "cls_nodeapi")


def fetch_cls_telegraph_roll(page_size: int = 50, timeout: float = 10) -> List[dict]:
    size = max(1, min(int(page_size), 100))
    params = {
        "appName": "CailianpressWeb",
        "os": "web",
        "sv": "7.7.5",
        "last_time": "",
        "refresh_type": "1",
        "rn": str(size),
    }
    qs = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    sign = hashlib.md5(hashlib.sha1(qs.encode()).hexdigest().encode()).hexdigest()
    url = "%s?%s&sign=%s" % (CLS_ROLL, qs, sign)
    r = requests.get(
        url,
        headers={"User-Agent": UA, "Referer": "https://www.cls.cn/"},
        timeout=timeout,
    )
    r.raise_for_status()
    payload = r.json()
    errno = payload.get("errno")
    if errno not in (0, "0", None):
        raise RuntimeError("cls roll errno=%s" % errno)
    rows = (payload.get("data") or {}).get("roll_data") or []
    if not isinstance(rows, list):
        raise RuntimeError("cls roll_data is not a list")
    return _normalize_rows(rows, "cls_roll")


def _dedupe_items(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("time"), it.get("title"), (it.get("content") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def fetch_cls_telegraph(page_size: int = 50, max_pages: int = 3, timeout: float = 10) -> Tuple[List[dict], str]:
    """优先 roll（实测更稳），nodeapi 作备用（NEWStodo 文档接口，部分环境 404）。"""
    errors = []
    # 1) roll
    try:
        batch = fetch_cls_telegraph_roll(page_size=page_size, timeout=timeout)
        if batch:
            return _dedupe_items(batch), "cls_roll"
        errors.append("roll: empty")
    except Exception as e:
        errors.append("roll: %s" % e)
        logger.warning("cls roll failed: %s", e)

    # 2) nodeapi multi-page
    collected: List[dict] = []
    try:
        last_time = int(time.time())
        for _ in range(max(1, int(max_pages))):
            batch = fetch_cls_telegraph_nodeapi(rn=page_size, last_time=last_time, timeout=timeout)
            if not batch:
                break
            collected.extend(batch)
            ctimes = [x.get("ctime") for x in batch if isinstance(x.get("ctime"), (int, float))]
            if not ctimes:
                break
            last_time = int(min(ctimes)) - 1
            if len(batch) < max(1, min(int(page_size), 100)):
                break
            time.sleep(0.35)
        if collected:
            return _dedupe_items(collected), "cls_nodeapi"
    except Exception as e:
        errors.append("nodeapi: %s" % e)
        logger.warning("cls nodeapi failed: %s", e)

    raise RuntimeError("财联社电报拉取失败: " + " | ".join(errors))


def _rule_sentiment(text: str) -> Tuple[str, float]:
    t = text or ""
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return "neutral", 0.4
    if pos > neg:
        return "positive", min(0.9, 0.5 + 0.1 * (pos - neg))
    if neg > pos:
        return "negative", min(0.9, 0.5 + 0.1 * (neg - pos))
    return "neutral", 0.5


def _openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        try:
            from src.config.settings import get_config
            cfg = get_config()
            openai_cfg = getattr(cfg, "openai", None)
            if openai_cfg and openai_cfg.api_key:
                key = openai_cfg.api_key.get_secret_value()
        except Exception:
            key = None
    if not key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except Exception as e:
        logger.warning("OpenAI client unavailable: %s", e)
        return None


def _parse_sentiment_response(content: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(content)
        sentiment = str(parsed.get("sentiment") or parsed.get("label") or "").lower()
        if sentiment in {"positive", "neutral", "negative"}:
            conf = parsed.get("confidence")
            try:
                conf_val = float(conf) if conf is not None else None
            except Exception:
                conf_val = None
            return {"sentiment": sentiment, "confidence": conf_val}
    except Exception:
        pass
    lowered = (content or "").lower()
    for sentiment in ("positive", "neutral", "negative"):
        if sentiment in lowered:
            return {"sentiment": sentiment, "confidence": None}
    return {}


def annotate_sentiment(items: List[dict], model: Optional[str] = None, use_openai: bool = True) -> List[dict]:
    client = _openai_client() if use_openai else None
    model_name = model or get_cls_config().get("openai_model") or "gpt-4o-mini"
    out = []
    for article in items:
        row = dict(article)
        text = ((row.get("title") or "") + " " + (row.get("content") or "")).strip()
        if client is not None:
            try:
                prompt = (
                    "你是财经新闻情绪分析助手。根据下面的财联社电报，判断情绪："
                    "positive / neutral / negative。"
                    "只返回 JSON，如 {\"sentiment\": \"neutral\", \"confidence\": 0.65}。\n"
                    "正文：%s" % text[:1500]
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                message = resp.choices[0].message.content or ""
                parsed = _parse_sentiment_response(message)
                if parsed.get("sentiment"):
                    row["sentiment"] = parsed["sentiment"]
                    row["sentiment_confidence"] = parsed.get("confidence")
                    row["sentiment_model"] = model_name
                    out.append(row)
                    continue
            except Exception as e:
                logger.debug("openai sentiment failed: %s", e)
        label, conf = _rule_sentiment(text)
        row["sentiment"] = label
        row["sentiment_confidence"] = conf
        row["sentiment_model"] = "rule_lexicon"
        out.append(row)
    return out


def sentiment_score(items: List[dict]) -> Dict[str, Any]:
    if not items:
        return {"score": 0.0, "n": 0, "positive": 0, "neutral": 0, "negative": 0, "label": "neutral"}
    score = 0.0
    wsum = 0.0
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for it in items:
        s = str(it.get("sentiment") or "neutral").lower()
        if s not in counts:
            s = "neutral"
        counts[s] += 1
        conf = it.get("sentiment_confidence")
        try:
            w = float(conf) if conf is not None else 0.5
        except Exception:
            w = 0.5
        val = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}[s]
        score += val * w
        wsum += w
    avg = score / wsum if wsum > 0 else 0.0
    if avg > 0.15:
        label = "positive"
    elif avg < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {
        "score": float(avg),
        "n": len(items),
        "positive": counts["positive"],
        "neutral": counts["neutral"],
        "negative": counts["negative"],
        "label": label,
    }





_CODE_RE = re.compile(r"(?<!\d)([0-9]{6})(?!\d)")
_NAME_CACHE_FILE = "a_share_name_lookup.json"
_NAME_CACHE_MAX_AGE_SEC = 24 * 3600
_name_lookup_cache = None  # type: Optional[Dict[str, str]]


def _normalize_code(ticker: str) -> str:
    digits = re.sub(r"\D", "", str(ticker or ""))
    if len(digits) < 6:
        raise ValueError("ticker needs 6-digit A-share code, got: %r" % ticker)
    return digits[-6:].zfill(6)


def _fetch_akshare_name_df() -> "pd.DataFrame":
    import akshare as ak
    df = ak.stock_info_a_code_name()
    if df is None or getattr(df, "empty", True):
        raise RuntimeError("akshare stock_info_a_code_name returned empty")
    return df


def load_code_name_lookup(force_refresh: bool = False) -> Dict[str, str]:
    """code(6) -> official name. For resolving a requested ticker only."""
    global _name_lookup_cache
    if _name_lookup_cache is not None and not force_refresh:
        return _name_lookup_cache
    cache_path = Path(get_cls_config()["cache_dir"]) / _NAME_CACHE_FILE
    if not force_refresh and cache_path.is_file():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age <= _NAME_CACHE_MAX_AGE_SEC:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("codes"):
                    _name_lookup_cache = {str(k).zfill(6): str(v) for k, v in raw["codes"].items()}
                    return _name_lookup_cache
        except Exception as e:
            logger.warning("load name lookup cache failed: %s", e)
    df = _fetch_akshare_name_df()
    cols = {str(c).lower(): c for c in df.columns}
    code_col = cols.get("code") or cols.get("symbol") or list(df.columns)[0]
    name_col = cols.get("name") or list(df.columns)[1]
    lookup = {}
    for _, row in df.iterrows():
        digits = re.sub(r"\D", "", str(row[code_col]).strip())
        if len(digits) < 6:
            continue
        code = digits[-6:].zfill(6)
        name = str(row[name_col]).strip()
        if not name or name.lower() == "nan":
            continue
        lookup[code] = name
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"codes": lookup}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("save name lookup cache failed: %s", e)
    _name_lookup_cache = lookup
    return lookup


def resolve_ticker_name(ticker: str) -> str:
    return load_code_name_lookup().get(_normalize_code(ticker)) or ""


def article_mentions_ticker(text: str, ticker: str, official_name: str = "") -> bool:
    """Retrieval filter for a requested ticker: body must contain code or full official name (len>=4)."""
    raw = text or ""
    code = _normalize_code(ticker)
    if code in _CODE_RE.findall(raw):
        return True
    name = (official_name or "").strip()
    if len(name) >= 4 and name in raw:
        return True
    return False


def _parse_item_date(it: dict):
    t = it.get("time")
    if t:
        try:
            return pd.to_datetime(t)
        except Exception:
            pass
    ctime = it.get("ctime")
    if isinstance(ctime, (int, float)):
        try:
            return pd.to_datetime(datetime.fromtimestamp(float(ctime), CN_TZ))
        except Exception:
            pass
    return None


def _filter_by_date(items: List[dict], start_date: str, end_date: str) -> List[dict]:
    start = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    out = []
    for it in items:
        dt = _parse_item_date(it)
        if dt is None:
            continue
        try:
            if getattr(dt, "tz", None) is not None:
                dt = dt.tz_convert("Asia/Shanghai").tz_localize(None)
        except Exception:
            try:
                dt = pd.Timestamp(dt.to_pydatetime().replace(tzinfo=None))
            except Exception:
                continue
        if start <= dt <= end:
            out.append(it)
    return out


def fetch_cls_news(
    ticker: str,
    start_date: str,
    end_date: str,
    analyze_sentiment: bool = False,
    sentiment_model: Optional[str] = None,
    use_openai: bool = True,
    page_size: Optional[int] = None,
    max_pages: Optional[int] = None,
    save: bool = False,
) -> pd.DataFrame:
    """Per-ticker CLS news (mirrors original fetch_news). Association = request ticker."""
    code = _normalize_code(ticker)
    try:
        start_s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_s = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError("Invalid from/to date: %s" % exc) from exc
    if pd.to_datetime(start_s) > pd.to_datetime(end_s):
        raise ValueError("from_date must be earlier than to_date")
    cfg = get_cls_config()
    ps = int(page_size if page_size is not None else cfg["page_size"])
    mp = int(max_pages if max_pages is not None else cfg["max_pages"])
    official_name = ""
    try:
        official_name = resolve_ticker_name(code)
    except Exception as e:
        logger.warning("resolve_ticker_name(%s) failed: %s", code, e)
    raw_items, source = fetch_cls_telegraph(page_size=ps, max_pages=mp)
    dated = _filter_by_date(raw_items, start_s, end_s)
    matched = []
    for it in dated:
        blob = "%s %s" % (it.get("title") or "", it.get("content") or "")
        if article_mentions_ticker(blob, code, official_name):
            row = dict(it)
            row["ticker"] = code
            row["symbol"] = code
            row["ticker_name"] = official_name
            matched.append(row)
    if analyze_sentiment and matched:
        matched = annotate_sentiment(
            matched,
            model=sentiment_model or cfg.get("openai_model"),
            use_openai=use_openai,
        )
    elif matched:
        for row in matched:
            row.setdefault("sentiment", None)
            row.setdefault("sentiment_confidence", None)
            row.setdefault("sentiment_model", None)
    df = pd.DataFrame(matched)
    if df.empty:
        df = pd.DataFrame(columns=[
            "ticker", "symbol", "ticker_name", "time", "title", "content",
            "sentiment", "sentiment_confidence", "sentiment_model",
            "source", "url", "ctime", "importance",
        ])
    meta = {
        "ticker": code,
        "ticker_name": official_name,
        "start_date": start_s,
        "end_date": end_s,
        "source": source,
        "n_raw": len(raw_items),
        "n_dated": len(dated),
        "n": int(len(df)),
        "fetched_at": datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
    }
    if analyze_sentiment and matched:
        meta.update(sentiment_score(matched))
    if save:
        cache_dir = Path(cfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(CN_TZ).strftime("%Y%m%d_%H%M%S")
        per_csv = cache_dir / ("cls_news_%s_%s.csv" % (code, stamp))
        latest_ticker = cache_dir / ("cls_news_%s_latest.csv" % code)
        df.to_csv(per_csv, index=False, encoding="utf-8-sig")
        df.to_csv(latest_ticker, index=False, encoding="utf-8-sig")
        meta["csv_path"] = str(per_csv)
        meta["latest_csv"] = str(latest_ticker)
        (cache_dir / ("cls_sentiment_%s_latest.json" % code)).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    df.attrs["meta"] = meta
    return df


def fetch_cls_news_many(
    tickers,
    start_date: str,
    end_date: str,
    analyze_sentiment: bool = True,
    sentiment_model: Optional[str] = None,
    use_openai: bool = True,
    page_size: Optional[int] = None,
    max_pages: Optional[int] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Batch per-ticker CLS news; one telegraph pull, then filter per requested ticker."""
    codes = [_normalize_code(t) for t in tickers]
    if not codes:
        raise ValueError("tickers must not be empty")
    try:
        start_s = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_s = pd.to_datetime(end_date).strftime("%Y-%m-%d")
    except Exception as exc:
        raise ValueError("Invalid from/to date: %s" % exc) from exc
    if pd.to_datetime(start_s) > pd.to_datetime(end_s):
        raise ValueError("from_date must be earlier than to_date")
    cfg = get_cls_config()
    ps = int(page_size if page_size is not None else cfg["page_size"])
    mp = int(max_pages if max_pages is not None else cfg["max_pages"])
    lookup = {}
    try:
        lookup = load_code_name_lookup()
    except Exception as e:
        logger.warning("name lookup unavailable: %s", e)
    raw_items, source = fetch_cls_telegraph(page_size=ps, max_pages=mp)
    dated = _filter_by_date(raw_items, start_s, end_s)
    all_rows = []
    per_ticker_counts = {}
    for code in codes:
        name = lookup.get(code) or ""
        hits = []
        for it in dated:
            blob = "%s %s" % (it.get("title") or "", it.get("content") or "")
            if article_mentions_ticker(blob, code, name):
                row = dict(it)
                row["ticker"] = code
                row["symbol"] = code
                row["ticker_name"] = name
                hits.append(row)
        per_ticker_counts[code] = len(hits)
        all_rows.extend(hits)
    if analyze_sentiment and all_rows:
        all_rows = annotate_sentiment(
            all_rows,
            model=sentiment_model or cfg.get("openai_model"),
            use_openai=use_openai,
        )
    df = pd.DataFrame(all_rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "ticker", "symbol", "ticker_name", "time", "title", "content",
            "sentiment", "sentiment_confidence", "sentiment_model",
            "source", "url", "ctime", "importance",
        ])
    summary = sentiment_score(all_rows) if all_rows else {
        "score": 0.0, "n": 0, "positive": 0, "neutral": 0, "negative": 0, "label": "neutral",
    }
    backend = "none"
    if analyze_sentiment:
        backend = "openai" if (use_openai and _openai_client() is not None) else "rule_lexicon"
    summary.update({
        "source": source,
        "start_date": start_s,
        "end_date": end_s,
        "tickers": codes,
        "per_ticker_counts": per_ticker_counts,
        "n_raw": len(raw_items),
        "n_dated": len(dated),
        "fetched_at": datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        "analyze_sentiment": bool(analyze_sentiment),
        "sentiment_backend": backend,
    })
    cache_dir = Path(cfg["cache_dir"])
    latest = cache_dir / "cls_news_latest.csv"
    summary_path = cache_dir / "cls_sentiment_latest.json"
    summary["latest_csv"] = str(latest)
    summary["summary_path"] = str(summary_path)
    if save:
        cache_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(CN_TZ).strftime("%Y%m%d_%H%M%S")
        stamped = cache_dir / ("cls_news_%s.csv" % stamp)
        df.to_csv(stamped, index=False, encoding="utf-8-sig")
        df.to_csv(latest, index=False, encoding="utf-8-sig")
        if not df.empty and "ticker" in df.columns:
            for code, sub in df.groupby("ticker"):
                sub.to_csv(
                    cache_dir / ("cls_news_%s_latest.csv" % code),
                    index=False,
                    encoding="utf-8-sig",
                )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["csv_path"] = str(stamped)
        logger.info(
            "cls per-ticker news saved n=%d tickers=%d source=%s -> %s",
            len(df), len(codes), source, latest,
        )
    return {
        "dataframe": df,
        "items": all_rows,
        "summary": summary,
        "latest_csv": str(latest) if save else None,
        "csv_path": summary.get("csv_path"),
    }


def load_latest_summary() -> Optional[Dict[str, Any]]:
    path = Path(get_cls_config()["cache_dir"]) / "cls_sentiment_latest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_latest_news_df() -> pd.DataFrame:
    path = Path(get_cls_config()["cache_dir"]) / "cls_news_latest.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception as e:
            logger.warning("load_latest_news_df failed: %s", e)
            return pd.DataFrame()
