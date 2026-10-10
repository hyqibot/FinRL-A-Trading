"""
Unified Data Source Module
==========================

Supports multiple financial data sources:
- Financial Modeling Prep (FMP, API required)

Provides unified data format for all sources.
"""

import os
import logging
import json
from typing import List, Optional, Dict, Any, Protocol, Tuple
from datetime import datetime
from abc import ABC
import pandas as pd
from pathlib import Path
import yfinance as yf
import pandas_market_calendars as mcal
import requests
import numpy as np
import concurrent.futures
from tqdm import tqdm
import pandas_market_calendars as mcal

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# Configure logging
logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Protocol for data source implementations."""

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components."""
        ...

    def get_fundamental_data(self, tickers: List[str],
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data for tickers."""
        ...

    def get_price_data(self, tickers: pd.DataFrame,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data for tickers."""
        ...

    def is_available(self) -> bool:
        """Check if data source is available."""
        ...

    def get_news(self, ticker: str, from_date: str, to_date: str,
                 analyze_sentiment: bool = False,
                 sentiment_model: Optional[str] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Get news articles for a ticker."""
        ...


class BaseDataFetcher(ABC):
    """Base class for data fetchers with common functionality."""

    def __init__(self, cache_dir: str = None):
        """
        Initialize base data fetcher.
        
        Args:
            cache_dir: Deprecated, kept for backward compatibility. Uses DATA_BASE_DIR env var instead.
        """
        # Import here to avoid circular dependency
        from src.data.data_store import get_data_store
        self.data_store = get_data_store(base_dir=cache_dir)

    def _standardize_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize fundamental data format."""
        required_columns = ['gvkey', 'datadate', 'tic', 'prccd', 'ajexdi']
        df = df.copy()

        # Ensure required columns exist
        for col in required_columns:
            if col not in df.columns:
                if col == 'gvkey':
                    df['gvkey'] = df.get('tic', df.index)  # Use ticker as gvkey if not available
                elif col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.get('date', df.index))
                elif col == 'tic':
                    df['tic'] = df.get('gvkey', df.index)  # Use gvkey as tic if not available
                elif col == 'prccd':
                    df['prccd'] = df.get('close', df.get('adj_close', 100))
                elif col == 'ajexdi':
                    df['ajexdi'] = df.get('adj_factor', 1.0)

        # Add adjusted price
        if 'adj_close' not in df.columns and 'prccd' in df.columns and 'ajexdi' in df.columns:
            df['adj_close'] = df['prccd'] / df['ajexdi']

        return df[required_columns + ['adj_close']]

    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize price data format."""
        df = df.copy()

        # Rename columns to match expected format
        column_mapping = {
            'Open': 'prcod',
            'High': 'prchd',
            'Low': 'prcld',
            'Close': 'prccd',
            'Adj Close': 'adj_close',
            'Volume': 'cshtrd'
        }

        df = df.rename(columns=column_mapping)

        # Ensure required columns exist
        required_columns = ['datadate', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']
        for col in required_columns:
            if col not in df.columns:
                if col == 'datadate':
                    df['datadate'] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index)
                elif col == 'prccd':
                    df['prccd'] = df.get('Close', df.get('close', 100))
                elif col == 'prcod':
                    df['prcod'] = df.get('Open', df.get('open', df['prccd']))
                elif col == 'prchd':
                    df['prchd'] = df.get('High', df.get('high', df['prccd']))
                elif col == 'prcld':
                    df['prcld'] = df.get('Low', df.get('low', df['prccd']))
                elif col == 'cshtrd':
                    df['cshtrd'] = df.get('Volume', df.get('volume', 1000000))
                elif col == 'adj_close':
                    df['adj_close'] = df.get('Adj Close', df.get('adj_close', df['prccd']))

        # Add gvkey column if missing
        if 'gvkey' not in df.columns:
            if 'tic' in df.columns:
                df['gvkey'] = df['tic']
            else:
                df['gvkey'] = 'UNKNOWN'

        # Add tic column if missing
        if 'tic' not in df.columns:
            if 'gvkey' in df.columns:
                df['tic'] = df['gvkey']
            else:
                df['tic'] = 'UNKNOWN'

        return df[['gvkey', 'datadate', 'tic', 'prccd', 'prcod', 'prchd', 'prcld', 'cshtrd', 'adj_close']]


class FMPFetcher(BaseDataFetcher, DataSource):
    """Financial Modeling Prep data fetcher."""

    def __init__(self, cache_dir: str = "./data/cache"):
        super().__init__(cache_dir)
        # self.base_url = "https://financialmodelingprep.com/api/v3"
        self.base_url = "https://financialmodelingprep.com/stable/"
        # Offline mode when API key is not provided; computed lazily but default here
        self.api_key = self._get_api_key()
        self.offline_mode = not bool(self.api_key)
        self._openai_client = None
        self._openai_api_key: Optional[str] = None
        self._sentiment_model: Optional[str] = None
        self._sentiment_request_timeout: int = 30
        self._init_sentiment_settings()

    def is_available(self) -> bool:
        """Check if data source is usable (online or offline)."""
        # FMPFetcher supports offline mode using local database even without API key
        return True

    def _get_api_key(self) -> Optional[str]:
        """Get FMP API key from config."""
        try:
            from src.config.settings import get_config
            config = get_config()
            if config.fmp.api_key:
                return config.fmp.api_key.get_secret_value()
            return None
        except Exception as e:
            logger.error(f"Failed to get FMP API key: {e}")
            return None

    def _init_sentiment_settings(self) -> None:
        """Load GPT sentiment analysis configuration."""
        try:
            from src.config.settings import get_config
            config = get_config()
            openai_cfg = getattr(config, 'openai', None)
            if openai_cfg and openai_cfg.api_key:
                self._openai_api_key = openai_cfg.api_key.get_secret_value()
                self._sentiment_model = openai_cfg.model
                self._sentiment_request_timeout = getattr(openai_cfg, 'request_timeout', 30) or 30
            else:
                self._openai_api_key = None
                self._sentiment_model = None
        except Exception as exc:
            logger.debug(f"Failed to initialize OpenAI settings: {exc}")
            self._openai_api_key = None
            self._sentiment_model = None

    def _get_openai_client(self):
        """Lazily initialize OpenAI client."""
        if not self._openai_api_key:
            return None
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; sentiment analysis unavailable")
            return None
        try:
            self._openai_client = OpenAI(api_key=self._openai_api_key)
        except Exception as exc:
            logger.warning(f"Failed to initialize OpenAI client: {exc}")
            self._openai_client = None
        return self._openai_client

    def _fetch_fmp_data(self, ticker: str, endpoint: str, period: str,
                        start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Helper to fetch data from FMP API with local-first strategy.

        - If start/end provided and raw payload in DB is sufficiently fresh, return it.
        - Else call API, then save raw payload and cache.
        """

        # Map endpoint -> payload key for raw save/load
        payload_key = None
        if endpoint in ('income-statement', 'balance-sheet-statement', 'cash-flow-statement', 'ratios', 'profile'):
            payload_key = {
                'income-statement': 'income',
                'balance-sheet-statement': 'balance',
                'cash-flow-statement': 'cashflow',
                'ratios': 'ratios',
                'profile': 'profile'
            }[endpoint]

            # 1) Local-first: if offline or data is fresh, use local store
            if start_date and end_date:
                try:
                    stored = self.data_store.get_raw_payload(ticker, payload_key, start_date, end_date, source='FMP')
                    if stored:
                        return stored
                    if not self.offline_mode:
                        latest_str = self.data_store.get_raw_payload_latest_date(ticker, payload_key, source='FMP')
                        if latest_str:
                            latest_dt = pd.to_datetime(latest_str)
                            today = pd.Timestamp.today().normalize()
                            req_end_dt = pd.to_datetime(end_date)
                            threshold_dt = min(today - pd.DateOffset(months=3), req_end_dt - pd.DateOffset(months=3))
                            if latest_dt >= threshold_dt:
                                stored2 = self.data_store.get_raw_payload(ticker, payload_key, start_date, end_date, source='FMP')
                                if stored2 is not None:
                                    return stored2
                except Exception:
                    pass

        # Offline mode: skip any network requests
        if self.offline_mode:
            logger.info(f"Offline mode: skip remote fetch for {endpoint} {ticker}; using local DB if available")
            return []

        # Build URL (profile endpoint doesn't use period).
        if endpoint == 'profile':
            url = f"{self.base_url}/{endpoint}?symbol={ticker}&apikey={self.api_key}"
        else:
            url = f"{self.base_url}/{endpoint}?symbol={ticker}&period={period}&limit=40&apikey={self.api_key}"
            # Append date range so FMP returns full history instead of recent quarters only
            if start_date:
                url += f"&from={start_date}"
            if end_date:
                url += f"&to={end_date}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Save raw payload for fundamentals endpoints
            if payload_key and start_date and end_date:
                try:
                    self.data_store._save_raw_payload('FMP', ticker, payload_key, start_date, end_date, data)
                except Exception as se:
                    logger.debug(f"Failed to save raw FMP payload {payload_key} for {ticker}: {se}")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch {endpoint} data for {ticker}: {e}")
            return []

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components from FMP."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Check database first
        cached_tickers, cached_sectors, cached_dateFirstAdded = self.data_store.get_sp500_components(date)
        if cached_tickers:
            logger.info(f"Loading S&P 500 components from database for {date}")
            return pd.DataFrame({'tickers': cached_tickers.split(","), 'sectors': cached_sectors.split(","), 'dateFirstAdded': cached_dateFirstAdded.split(",")})

        # If offline, return latest available from DB and skip network
        if self.offline_mode:
            latest, latest_sectors, latest_dateFirstAdded = self.data_store.get_sp500_components()
            if latest:
                logger.info("Offline mode: returning latest S&P 500 components from database")
                return pd.DataFrame({'tickers': latest.split(","), 'sectors': latest_sectors.split(","), 'dateFirstAdded': latest_dateFirstAdded.split(",")})
            logger.warning("Offline mode: no S&P 500 components available in database")
            return pd.DataFrame({'tickers': [], 'sectors': [], 'dateFirstAdded': []}).set_index(pd.Index([], name='date'))

        try:
            if not self.api_key:
                raise ValueError("FMP API key not found")

            url = f"{self.base_url}/sp500-constituent?apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()

            # Extract tickers
            tickers = [item['symbol'] for item in data]
            tickers_str = ','.join(tickers)
            sectors = [item['sector'] for item in data]
            sectors_str = ','.join(sectors)
            dateFirstAdded = [item['dateFirstAdded'] for item in data]
            dateFirstAdded_str = ','.join(dateFirstAdded)

            # Create DataFrame
            df = pd.DataFrame({
                'tickers': tickers,
                'sectors': sectors,
                'dateFirstAdded': dateFirstAdded
            })

            # Save to database
            self.data_store.save_sp500_components(date, tickers_str, sectors_str, dateFirstAdded_str)
            logger.info(f"Saved S&P 500 components to database for {date}")

            return df

        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 components from FMP: {e}")
            # Fallback to whatever in DB
            latest, latest_sectors, latest_dateFirstAdded = self.data_store.get_sp500_components()
            if latest:
                return pd.DataFrame({'tickers': latest.split(","), 'sectors': latest_sectors.split(","), 'dateFirstAdded': latest_dateFirstAdded.split(",")})
            return pd.DataFrame({'tickers': [], 'sectors': [], 'dateFirstAdded': []})

    def _parse_sentiment_response(self, content: str) -> Dict[str, Any]:
        """Parse GPT response into sentiment payload."""
        if not content:
            return {}
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                sentiment = str(parsed.get('sentiment') or parsed.get('label') or '').lower()
                if sentiment in {'positive', 'neutral', 'negative'}:
                    confidence = parsed.get('confidence') or parsed.get('score')
                    try:
                        confidence_val = float(confidence) if confidence is not None else None
                    except (TypeError, ValueError):
                        confidence_val = None
                    return {'sentiment': sentiment, 'confidence': confidence_val}
        except json.JSONDecodeError:
            pass

        lowered = content.strip().lower()
        for sentiment in ('positive', 'neutral', 'negative'):
            if sentiment in lowered:
                return {'sentiment': sentiment, 'confidence': None}
        return {}

    def _annotate_sentiment(self, articles: List[Dict[str, Any]], sentiment_model: Optional[str] = None) -> None:
        """Call GPT API to annotate sentiment for a batch of news articles."""
        if not articles:
            return

        client = self._get_openai_client()
        if not client:
            logger.info("OpenAI client unavailable, skip sentiment analysis")
            return

        model = sentiment_model or self._sentiment_model
        if not model:
            logger.info("Sentiment model not configured, skip sentiment analysis")
            return

        for article in articles:
            title = (article.get('title') or '').strip()
            body = (article.get('text') or article.get('body') or '').strip()
            if not title and not body:
                continue
            if len(body) > 1500:
                body = body[:1500]

            prompt = (
                "请阅读以下新闻并判断整体情绪是 positive、neutral 还是 negative。"
                " 仅返回 JSON，如 {\"sentiment\": \"neutral\", \"confidence\": 0.65}。"
                f"\n标题: {title}\n内容: {body}"
            )

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "你是一名金融新闻情绪分析助手，只输出JSON。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=60
                )
                message = response.choices[0].message.content.strip()
                parsed = self._parse_sentiment_response(message)
                if parsed.get('sentiment'):
                    article['sentiment'] = parsed['sentiment']
                    article['sentiment_confidence'] = parsed.get('confidence')
                    article['sentiment_model'] = model
            except Exception as exc:
                logger.debug(f"Sentiment analysis failed for news '{title}': {exc}")

    def _ensure_news_sentiment(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        df: pd.DataFrame,
        sentiment_model: Optional[str] = None
    ) -> pd.DataFrame:
        """Ensure cached rows also have sentiment when requested."""
        if df.empty or 'sentiment' not in df.columns:
            return df

        mask_missing = df['sentiment'].isna() | (df['sentiment'].astype(str).str.strip() == '')
        if not mask_missing.any():
            return df

        articles = df[mask_missing].to_dict('records')
        self._annotate_sentiment(articles, sentiment_model)

        updated = False
        for article in articles:
            sentiment = article.get('sentiment')
            if sentiment:
                self.data_store.update_news_sentiment(
                    ticker,
                    article.get('published_datetime'),
                    sentiment,
                    article.get('sentiment_confidence'),
                    article.get('sentiment_model') or sentiment_model or self._sentiment_model
                )
                updated = True

        if updated:
            return self.data_store.get_news_articles(ticker, start_date, end_date)
        return df

    def get_news(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        analyze_sentiment: bool = False,
        sentiment_model: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Get news from FMP with local caching and optional GPT sentiment analysis."""
        if not ticker:
            raise ValueError("ticker is required for get_news")

        try:
            start_date = pd.to_datetime(from_date).strftime('%Y-%m-%d')
            end_date = pd.to_datetime(to_date).strftime('%Y-%m-%d')
        except Exception as exc:
            raise ValueError(f"Invalid from/to date: {exc}") from exc

        if pd.to_datetime(start_date) > pd.to_datetime(end_date):
            raise ValueError("from_date must be earlier than to_date")

        cached = self.data_store.get_news_articles(ticker, start_date, end_date)
        logger.info(f"Loaded {len(cached)} cached news rows for {ticker} ({start_date} -> {end_date})")

        if self.offline_mode or not self.api_key:
            logger.info("Offline or missing API key: returning cached news only")
            if analyze_sentiment:
                cached = self._ensure_news_sentiment(ticker, start_date, end_date, cached, sentiment_model)
            return cached

        missing_ranges = (
            [(start_date, end_date)]
            if force_refresh
            else self.data_store.get_missing_news_ranges(ticker, start_date, end_date)
        )

        new_articles: List[Dict[str, Any]] = []
        completed_ranges: List[Tuple[str, str, int]] = []
        if missing_ranges:
            for range_start, range_end in missing_ranges:
                url = (
                    f"{self.base_url}/news/stock?"
                    f"symbols={ticker}&from={range_start}&to={range_end}&apikey={self.api_key}"
                )
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict) and 'news' in payload:
                        news_items = payload['news']
                    elif isinstance(payload, list):
                        news_items = payload
                    else:
                        news_items = []

                    logger.info(f"Fetched {len(news_items)} news entries for {ticker} ({range_start}->{range_end})")

                    for item in news_items:
                        item.setdefault('symbol', ticker)
                        if not item.get('publishedDate'):
                            item['publishedDate'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

                    new_articles.extend(news_items)
                    completed_ranges.append((range_start, range_end, len(news_items)))
                except requests.RequestException as exc:
                    logger.warning(f"Failed to fetch news for {ticker} ({range_start}->{range_end}): {exc}")
        else:
            logger.info(f"News cache already covers requested range for {ticker}")

        if new_articles:
            if analyze_sentiment:
                self._annotate_sentiment(new_articles, sentiment_model)
            self.data_store.save_news_articles(ticker, new_articles)

        if completed_ranges:
            for range_start, range_end, count in completed_ranges:
                self.data_store.save_news_fetch_range(ticker, range_start, range_end, count)

        result = self.data_store.get_news_articles(ticker, start_date, end_date)
        if analyze_sentiment:
            result = self._ensure_news_sentiment(ticker, start_date, end_date, result, sentiment_model)

        return result

    def get_fundamental_data(self, tickers: pd.DataFrame, start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data from FMP with extended fields and forward y_return and incremental updates.
        Adds one next quarter to compute the last forward return, then drops that extra row, and drops rows with missing y_return.
        If align_quarter_dates is True, align each quarter to Mar/Jun/Sep/Dec 1st and compute prices and y_return based on these aligned dates."""

        # # Step 1: Check database for existing data
        # existing_data = self.data_store.get_fundamental_data(tickers, start_date, end_date)
        # logger.info(f"Found {len(existing_data)} existing fundamental records in database")

        # Step 2: Decide tickers to handle (local-first inside _fetch_fmp_data avoids extra API calls)
        tickers_to_fetch = list(tickers['tickers'])

        # Step 3: Fetch missing data from API
        all_records: List[Dict[str, Any]] = []
        if tickers_to_fetch:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            # extend both ends to ensure previous/next quarter prices available
            extended_start_dt = (start_dt - pd.DateOffset(months=4))
            # Ensure extended_end_dt does not exceed today
            today = pd.Timestamp.today().normalize()
            candidate_end_dt = (end_dt + pd.DateOffset(months=3))
            extended_end_dt = min(candidate_end_dt, today)
            extended_start_date = extended_start_dt.strftime('%Y-%m-%d')
            extended_end_date = extended_end_dt.strftime('%Y-%m-%d')
            
            def align_to_mjsd_first(d: pd.Timestamp) -> pd.Timestamp:
                # Map quarter end month to the 1st day of the following 2 months later: target months 3,6,9,12 => day 1
                # We approximate by finding the quarter index and composing a date with month in {3,6,9,12} and day=1
                month = int(d.month)
                year = int(d.year)
                # Determine quarter start month
                # Q1->3, Q2->6, Q3->9, Q4->12
                if month in (12, 1, 2):
                    if month == 12:
                        year = year + 1
                    return pd.Timestamp(year=year, month=3, day=1)
                if month in (3, 4, 5):
                    return pd.Timestamp(year=year, month=6, day=1)
                if month in (6, 7, 8):
                    return pd.Timestamp(year=year, month=9, day=1)
                # 9,10,11
                return pd.Timestamp(year=year, month=12, day=1)

        prices = self.get_price_data(tickers, extended_start_date, extended_end_date)

        for ticker in tickers_to_fetch if tickers_to_fetch else []:
            try:
                # Local-first: helper will use DB in offline mode
                income_data = self._fetch_fmp_data(ticker, 'income-statement', 'quarter', start_date, end_date)
                balance_data = self._fetch_fmp_data(ticker, 'balance-sheet-statement', 'quarter', start_date, end_date)
                cashflow_data = self._fetch_fmp_data(ticker, 'cash-flow-statement', 'quarter', start_date, end_date)

                # ratios
                # Ratios via helper
                ratios_data = self._fetch_fmp_data(ticker, 'ratios', 'quarter', start_date, end_date) or []

                # profile for sector
                # Profile via helper
                gsector = None
                try:
                    gsector = tickers[tickers['tickers'] == ticker]['sectors'].values[0]
                    # prof_json = self._fetch_fmp_data(ticker, 'profile', 'na', start_date, end_date)
                    # if isinstance(prof_json, list) and prof_json:
                    #     gsector = prof_json[0].get('sector')
                except Exception as e:
                    logger.warning(f"Failed to fetch profile for {ticker}: {e}")

                # price data for quarter adj_close (need next quarter too)
                
                prices_t = prices[prices['tic'] == ticker].copy() if not prices.empty else pd.DataFrame()
                if not prices_t.empty and 'datadate' in prices_t.columns:
                    prices_t['datadate'] = pd.to_datetime(prices_t['datadate'], errors='coerce')
                    prices_t = prices_t.dropna(subset=['datadate']).sort_values('datadate', ascending=True)

                # Index data by date for quick lookup
                def index_by_date(items: List[Dict[str, Any]]) -> Dict[pd.Timestamp, Dict[str, Any]]:
                    out = {}
                    for it in items or []:
                        key_ts = None
                        try:
                            year = it.get('calendarYear')
                            period_raw = it.get('period')
                            period = str(period_raw).upper() if period_raw is not None else None
                            if year is not None and period:
                                # 统一使用 calendarYear + period(Q1/Q2/Q3/Q4) 的季度末日期作为键
                                q_map = {
                                    'Q1': (3, 31), 
                                    'Q2': (6, 30), 
                                    'Q3': (9, 30), 
                                    'Q4': (12, 31),
                                    'FY': (12, 31),  # 若出现 FY，按 Q4 处理
                                }
                                md = q_map.get(period)
                                if md:
                                    key_ts = pd.Timestamp(int(year), md[0], md[1])
                            # 若缺失 calendarYear/period，则回退到原始 date，再映射到所在公历季度末
                            if key_ts is None and 'date' in it and it.get('date'):
                                d = pd.to_datetime(it['date'], errors='coerce')
                                if pd.notna(d):
                                    q = ((int(d.month) - 1) // 3) + 1
                                    end_month = q * 3
                                    end_day = 31 if end_month in (3, 12) else 30
                                    key_ts = pd.Timestamp(int(d.year), end_month, end_day)
                        except Exception:
                            key_ts = None
                        if key_ts is not None:
                            out[key_ts] = it
                    return out

                income_by_date = index_by_date(income_data)
                balance_by_date = index_by_date(balance_data)
                cashflow_by_date = index_by_date(cashflow_data)
                ratios_by_date = index_by_date(ratios_data)

                # in-range quarter dates
                # TODO: ratios and cashflow dates are different in NVDA, need to adapt to this case
                all_quarter_dates = sorted(set(income_by_date.keys()) | set(balance_by_date.keys()) | set(cashflow_by_date.keys()))
                inrange_quarters = [d for d in all_quarter_dates if start_dt <= d <= end_dt]

                # If aligning, create a mapping from original qd -> aligned date (仅用于计算 y_return 的价格)
                if align_quarter_dates:
                    aligned_dates = {qd: align_to_mjsd_first(qd) for qd in inrange_quarters}
                else:
                    aligned_dates = {qd: qd for qd in inrange_quarters}

                for qd, aligned_date in aligned_dates.items():
                    # if aligned_date > today: 
                    #     continue
                    income_q = income_by_date.get(qd, {})
                    balance_q = balance_by_date.get(qd, {})
                    cash_q = cashflow_by_date.get(qd, {})
                    ratio_q = ratios_by_date.get(qd, {})

                    # shares outstanding
                    shares_out = balance_q.get('commonStockSharesOutstanding') or income_q.get('weightedAverageShsOutDil') or income_q.get('weightedAverageShsOut')
                    try:
                        shares_out = float(shares_out) if shares_out is not None else None
                    except Exception:
                        shares_out = None

                    # equity and income
                    equity = balance_q.get('totalStockholdersEquity', balance_q.get('totalStockholdersEquity', 0)) or 0
                    try:
                        equity = float(equity)
                    except Exception:
                        equity = 0.0

                    # net_income_ratio: prefer ratio endpoint's netProfitMargin
                    net_income_ratio = ratio_q.get('netProfitMargin')
                    if net_income_ratio is None:
                        net_income_ratio = income_q.get('netIncomeRatio')
                    try:
                        net_income_ratio = float(net_income_ratio) if net_income_ratio is not None else np.nan
                    except (TypeError, ValueError):
                        net_income_ratio = np.nan

                    revenue = income_q.get('revenue', income_q.get('sales', 0))
                    try:
                        revenue = float(revenue)
                    except Exception:
                        revenue = 0.0

                    # 价格：区分原始季度日与对齐日
                    prccd_orig = np.nan
                    adj_close_orig = np.nan
                    prccd_aligned = np.nan
                    adj_close_aligned = np.nan
                    if not prices_t.empty and 'datadate' in prices_t.columns:
                        # 原始季度日价格：优先取季度日(含)之后最近交易日，若无则回退到之前最近交易日
                        price_row_orig = prices_t[prices_t['datadate'] >= qd].head(1)
                        if price_row_orig.empty:
                            price_row_orig = prices_t[prices_t['datadate'] < qd].tail(1)
                        if not price_row_orig.empty:
                            prccd_orig = float(price_row_orig.iloc[0].get('prccd', np.nan))
                            ac_o = price_row_orig.iloc[0].get('adj_close', np.nan)
                            if pd.notna(ac_o):
                                adj_close_orig = float(ac_o)

                        # 对齐日价格：仅当开启对齐时计算，用于 y_return
                        if align_quarter_dates:
                            max_days_forward = 10
                            price_row_aln = pd.DataFrame()
                            for days_offset in range(max_days_forward + 1):
                                search_date = aligned_date + pd.Timedelta(days=days_offset)
                                price_row_aln = prices_t[prices_t['datadate'] == search_date]
                                if not price_row_aln.empty:
                                    break
                            if price_row_aln.empty:
                                price_row_aln = prices_t[prices_t['datadate'] <= aligned_date].tail(1)
                            if not price_row_aln.empty:
                                prccd_aligned = float(price_row_aln.iloc[0].get('prccd', np.nan))
                                ac_a = price_row_aln.iloc[0].get('adj_close', np.nan)
                                if pd.notna(ac_a):
                                    adj_close_aligned = float(ac_a)

                    # EPS, BPS, DPS
                    eps = income_q.get('eps')
                    net_income_raw = income_q.get('netIncome')
                    try:
                        net_income = float(net_income_raw) if net_income_raw is not None else np.nan
                    except Exception:
                        net_income = np.nan
                    if eps is None and shares_out:
                        eps = (net_income / shares_out) if (pd.notna(net_income) and shares_out) else np.nan

                    try:
                        eps = float(eps) if eps is not None else np.nan
                    except Exception:
                        eps = np.nan

                    bps = (equity / shares_out) if shares_out else np.nan

                    # DPS: prefer ratio endpoint, fallback to cashflow
                    dps_raw = ratio_q.get('dividendPerShare')
                    try:
                        dps = float(dps_raw) if dps_raw is not None else np.nan
                    except (TypeError, ValueError):
                        dps = np.nan
                    if pd.isna(dps):
                        dividends_paid = cash_q.get('commonDividendsPaid') or cash_q.get('dividendsPaid')
                        try:
                            dps = (abs(float(dividends_paid)) / shares_out) if (dividends_paid is not None and shares_out) else np.nan
                        except Exception:
                            dps = np.nan

                    # ratios (prefer API fields, else compute)
                    cur_ratio = ratio_q.get('currentRatio')
                    if cur_ratio is None:
                        ca = balance_q.get('totalCurrentAssets')
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            cur_ratio = (float(ca) / float(cl)) if (ca and cl and float(cl) != 0) else np.nan
                        except Exception:
                            cur_ratio = np.nan

                    quick_ratio = ratio_q.get('quickRatio')
                    if quick_ratio is None:
                        ca = balance_q.get('totalCurrentAssets')
                        inv = balance_q.get('inventory') or balance_q.get('inventoryAndOtherCurrentAssets')
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            quick_ratio = ((float(ca) - float(inv)) / float(cl)) if (ca and cl and float(cl) != 0 and inv is not None) else np.nan
                        except Exception:
                            quick_ratio = np.nan

                    cash_ratio = ratio_q.get('cashRatio')
                    if cash_ratio is None:
                        cash_st = balance_q.get('cashAndShortTermInvestments') or balance_q.get('cashAndShortTermInvestments', None)
                        cl = balance_q.get('totalCurrentLiabilities')
                        try:
                            cash_ratio = (float(cash_st) / float(cl)) if (cash_st and cl and float(cl) != 0) else np.nan
                        except Exception:
                            cash_ratio = np.nan

                    acc_rec_turnover = ratio_q.get('receivablesTurnover') or ratio_q.get('accountsReceivableTurnover')

                    debt_ratio = ratio_q.get('debtRatio')
                    if debt_ratio is None:
                        liabilities = balance_q.get('totalLiabilities')
                        assets = balance_q.get('totalAssets')
                        try:
                            debt_ratio = (float(liabilities) / float(assets)) if (liabilities and assets and float(assets) != 0) else np.nan
                        except Exception:
                            debt_ratio = np.nan

                    debt_to_equity = ratio_q.get('debtEquityRatio') or ratio_q.get('debtToEquity')
                    if debt_to_equity is None:
                        liabilities = balance_q.get('totalLiabilities')
                        try:
                            debt_to_equity = (float(liabilities) / float(equity)) if (liabilities and equity and float(equity) != 0) else np.nan
                        except Exception:
                            debt_to_equity = np.nan

                    # price multiples
                    pe = ratio_q.get('priceEarningsRatio')
                    if pe is None:
                        try:
                            pe = (prccd_orig / eps) if (pd.notna(prccd_orig) and eps and eps != 0) else np.nan
                        except Exception:
                            pe = np.nan

                    ps = ratio_q.get('priceToSalesRatio')
                    pb = ratio_q.get('priceToBookRatio')
                    if pb is None:
                        try:
                            pb = (prccd_orig / bps) if (pd.notna(prccd_orig) and bps and bps != 0) else np.nan
                        except Exception:
                            pb = np.nan

                    roe = (net_income / equity) if (pd.notna(net_income) and equity and float(equity) != 0.0) else np.nan

                    # --- Extract all additional ratio fields ---
                    RATIO_FIELD_MAP = {
                        # Profitability
                        'grossProfitMargin': 'gross_margin',
                        'operatingProfitMargin': 'operating_margin',
                        'ebitdaMargin': 'ebitda_margin',
                        'pretaxProfitMargin': 'pretax_margin',
                        'effectiveTaxRate': 'effective_tax_rate',
                        'ebtPerEbit': 'ebt_per_ebit',
                        # netIncomePerEBT removed (= 1 - effective_tax_rate)
                        # Efficiency
                        'assetTurnover': 'asset_turnover',
                        'fixedAssetTurnover': 'fixed_asset_turnover',
                        'inventoryTurnover': 'inventory_turnover',
                        'payablesTurnover': 'payables_turnover',
                        'workingCapitalTurnoverRatio': 'wc_turnover',
                        # Leverage
                        'debtToAssetsRatio': 'debt_to_assets',
                        'debtToCapitalRatio': 'debt_to_capital',
                        'longTermDebtToCapitalRatio': 'lt_debt_to_capital',
                        # financialLeverageRatio removed (= debt_to_equity + 1)
                        'interestCoverageRatio': 'interest_coverage',
                        'debtServiceCoverageRatio': 'debt_service_coverage',
                        'debtToMarketCap': 'debt_to_mktcap',
                        # Cash Flow
                        'freeCashFlowPerShare': 'fcf_per_share',
                        'operatingCashFlowPerShare': 'ocf_per_share',
                        'cashPerShare': 'cash_per_share',
                        'capexPerShare': 'capex_per_share',
                        'freeCashFlowOperatingCashFlowRatio': 'fcf_to_ocf',
                        'operatingCashFlowRatio': 'ocf_ratio',
                        'operatingCashFlowSalesRatio': 'ocf_to_sales',
                        'operatingCashFlowCoverageRatio': 'ocf_coverage',
                        'shortTermOperatingCashFlowCoverageRatio': 'st_ocf_coverage',
                        'capitalExpenditureCoverageRatio': 'capex_coverage',
                        # Per-Share
                        'revenuePerShare': 'revenue_per_share',
                        'tangibleBookValuePerShare': 'tangible_bvps',
                        'interestDebtPerShare': 'interest_debt_per_share',
                        # Valuation
                        'priceToEarningsGrowthRatio': 'peg',
                        'priceToFreeCashFlowRatio': 'price_to_fcf',
                        'priceToOperatingCashFlowRatio': 'price_to_ocf',
                        # priceToFairValue removed (= pb)
                        'enterpriseValueMultiple': 'ev_multiple',
                        # Dividend
                        'dividendPayoutRatio': 'dividend_payout',
                        'dividendYield': 'dividend_yield',
                        'dividendPaidAndCapexCoverageRatio': 'div_capex_coverage',
                        # Solvency
                        'solvencyRatio': 'solvency_ratio',
                    }

                    extra_ratios = {}
                    for fmp_key, db_col in RATIO_FIELD_MAP.items():
                        v = ratio_q.get(fmp_key)
                        try:
                            extra_ratios[db_col] = float(v) if v is not None else np.nan
                        except (TypeError, ValueError):
                            extra_ratios[db_col] = np.nan

                    # Earnings release dates from income statement
                    filing_date = income_q.get('filingDate') or ''
                    accepted_date = (income_q.get('acceptedDate') or '')[:10]  # trim time part

                    record = {
                        'gvkey': ticker,
                        'datadate': aligned_date.strftime('%Y-%m-%d') if align_quarter_dates else qd.strftime('%Y-%m-%d'),
                        'tic': ticker,
                        'gsector': gsector,
                        'filing_date': filing_date,
                        'accepted_date': accepted_date,
                        'adj_close_q': (adj_close_aligned if align_quarter_dates and pd.notna(adj_close_aligned) else adj_close_orig) if pd.notna(adj_close_orig) or pd.notna(adj_close_aligned) else np.nan,
                        'EPS': eps if eps is not None else np.nan,
                        'BPS': bps if bps is not None else np.nan,
                        'DPS': dps if pd.notna(dps) else np.nan,
                        'cur_ratio': cur_ratio if cur_ratio is not None else np.nan,
                        'quick_ratio': quick_ratio if quick_ratio is not None else np.nan,
                        'cash_ratio': cash_ratio if cash_ratio is not None else np.nan,
                        'acc_rec_turnover': acc_rec_turnover if acc_rec_turnover is not None else np.nan,
                        'debt_ratio': debt_ratio if debt_ratio is not None else np.nan,
                        'debt_to_equity': debt_to_equity if debt_to_equity is not None else np.nan,
                        'pe': pe if pe is not None else np.nan,
                        'ps': ps if ps is not None else np.nan,
                        'pb': pb if pb is not None else np.nan,
                        'roe': roe if pd.notna(roe) else np.nan,
                        'net_income_ratio': net_income_ratio,
                        **extra_ratios,
                    }
                    all_records.append(record)
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {ticker}: {e}")
                continue
        
        # Step 4: Process and combine data
        df = pd.DataFrame()
        if all_records:
            df = pd.DataFrame(all_records)
            try:
                df = df.sort_values(['tic', 'datadate'])
                # forward return: current quarter vs last quarter
                # df['y_return'] = np.log(df['adj_close_q'].shift(-1) / df['adj_close_q'])
                df['y_return'] = df.groupby('tic')['adj_close_q'].transform(lambda s: np.log(s.shift( -1) / s))

                # Fill missing y_return for the last quarter of each ticker
                # using price data (e.g., Q4 2025 y_return needs Q1 2026 price which is available even without Q1 2026 fundamentals)
                if not prices.empty and 'datadate' in prices.columns:
                    prices_dt = prices.copy()
                    prices_dt['datadate'] = pd.to_datetime(prices_dt['datadate'], errors='coerce')
                    for tic_name, grp in df.groupby('tic'):
                        last_idx = grp.index[-1]
                        if pd.isna(df.loc[last_idx, 'y_return']):
                            last_qd = pd.to_datetime(df.loc[last_idx, 'datadate'])
                            cur_price = df.loc[last_idx, 'adj_close_q']
                            if pd.isna(cur_price) or cur_price <= 0:
                                continue
                            # Determine next quarter-end date
                            next_q_end = last_qd + pd.offsets.QuarterEnd(1)
                            # Look up price near next quarter-end from price data
                            tic_prices = prices_dt[prices_dt['tic'] == tic_name].sort_values('datadate')
                            if tic_prices.empty:
                                continue
                            # Find closest trading day to next_q_end (within 7 days before)
                            mask_near = (tic_prices['datadate'] >= next_q_end - pd.Timedelta(days=7)) & (tic_prices['datadate'] <= next_q_end + pd.Timedelta(days=3))
                            near_prices = tic_prices[mask_near]
                            if near_prices.empty:
                                # Fallback: closest price before next_q_end
                                near_prices = tic_prices[tic_prices['datadate'] <= next_q_end + pd.Timedelta(days=3)].tail(1)
                            if not near_prices.empty:
                                # Pick the one closest to next_q_end
                                near_prices = near_prices.iloc[(near_prices['datadate'] - next_q_end).abs().argsort()[:1]]
                                next_price = near_prices.iloc[0].get('adj_close', np.nan)
                                if pd.notna(next_price) and float(next_price) > 0:
                                    df.loc[last_idx, 'y_return'] = np.log(float(next_price) / cur_price)

                # keep only original in-range quarters
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                mask_in_range = pd.to_datetime(df['datadate']).between(start_dt, end_dt, inclusive='both')
                df = df[mask_in_range].reset_index(drop=True)
                # don't need to drop rows without y_return
                # df = df[df['y_return'].notna()].reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Failed to compute forward y_return: {e}")

        # Step 5: Return combined data
        mode = 'offline' if self.offline_mode else 'online'
        logger.info(f"Returning {len(df)} total fundamental records ({mode} mode)")
        
        return df

    def get_active_trading_list(self) -> pd.DataFrame:
        """Get active trading list from FMP."""
        if not self.api_key:
            raise ValueError("FMP API key not found")
        url = f"{self.base_url}/actively-trading-list?apikey={self.api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def get_realtime_single_price_data(self, ticker: str) -> pd.DataFrame:
        """Get realtime price data from FMP."""
        if not self.api_key:
            raise ValueError("FMP API key not found")
        url = f"{self.base_url}/quote?symbol={ticker}&apikey={self.api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def get_realtime_batch_price_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get realtime price data from FMP."""
        if not self.api_key:
            raise ValueError("FMP API key not found")
        url = f"{self.base_url}/batch-quote?symbols={','.join(tickers)}&apikey={self.api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    def get_price_data(self, tickers: pd.DataFrame,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data from FMP with incremental updates."""

        # If end_date is today in exchange timezone and current time is before market close,
        # move end_date back by one day to avoid fetching incomplete day
        now_local = pd.Timestamp.now(tz='America/New_York')
        if now_local.date() <= pd.to_datetime(end_date).date():
            schedule = mcal.get_calendar(name='NYSE').schedule(start_date=end_date, end_date=end_date, tz='America/New_York')

            if not schedule.empty:
                try:
                    close_time = schedule['market_close'].iloc[-1]
                    # If user asked up to today and before close, shift end_date back one day
                    if now_local.time() < close_time.time():
                        end_date = (now_local - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                except Exception:
                    pass

        # Step 1: Check database for existing data
        existing_data = self.data_store.get_price_data(tickers['tickers'], start_date, end_date)
        logger.info(f"Found {len(existing_data)} existing price records in database")

        # Offline: return existing data only
        if self.offline_mode:
            logger.info("Offline mode: returning existing price data from database and skipping remote fetch")
            return existing_data

        # Step 2: Identify tickers that need to be fetched
        tickers_to_fetch = {}

        # Bulk query for missing ranges to reduce overhead
        try:
            bulk_missing = self.data_store.get_missing_price_dates_bulk(tickers, start_date, end_date, exchange='NYSE')
        except Exception as e:
            logger.debug(f"bulk missing ranges failed, fallback to per-ticker: {e}")
            bulk_missing = None

        if bulk_missing is not None:
            for t, ranges in (bulk_missing or {}).items():
                if ranges:
                    tickers_to_fetch[t] = ranges
        else:
            # Prepare per-ticker dateFirstAdded lookup
            tickers_list = tickers['tickers'].astype(str).tolist() if isinstance(tickers, pd.DataFrame) else list(tickers)
            if isinstance(tickers, pd.DataFrame) and 'dateFirstAdded' in tickers.columns:
                dfa_map = {row['tickers']: row['dateFirstAdded'] for _, row in tickers[['tickers', 'dateFirstAdded']].iterrows()}
            else:
                dfa_map = {t: None for t in tickers_list}

            def check_missing_ranges(ticker: str):
                dfa_raw = dfa_map.get(ticker)
                try:
                    dfa = pd.to_datetime(dfa_raw, errors='coerce') if dfa_raw is not None else None
                except Exception:
                    dfa = None
                eff_start_dt = max(pd.to_datetime(start_date), dfa) if dfa is not None else pd.to_datetime(start_date)
                eff_start_str = eff_start_dt.strftime('%Y-%m-%d')

                missing_ranges = self.data_store.get_missing_price_dates(ticker, eff_start_str, end_date, exchange='NYSE')
                if missing_ranges:
                    logger.info(f"Ticker {ticker}: Need to fetch price data")
                    return (ticker, missing_ranges)
                return None

            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(check_missing_ranges, tickers_list))

            for res in results:
                if res:
                    ticker, missing_ranges = res
                    tickers_to_fetch[ticker] = missing_ranges

        # Step 3: Fetch missing data from API
        all_data = []
        if tickers_to_fetch:
            logger.info(f"Fetching price data for {len(tickers_to_fetch)} tickers from FMP")
            
            for ticker, date_ranges in tqdm(tickers_to_fetch.items()):
                try:
                    # 取最小和最大日期
                    min_date = min(start for start, _ in date_ranges)
                    max_date = max(end for _, end in date_ranges)

                    url = f"{self.base_url}/historical-price-eod/full?symbol={ticker}&from={min_date}&to={max_date}&apikey={self.api_key}"
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    data = response.json()

                    if isinstance(data, dict) and 'historical' in data:
                        historical_rows = data.get('historical') or []
                    elif isinstance(data, list):
                        historical_rows = data
                    else:
                        logger.warning(f"No historical data key in response for {ticker} ({min_date} to {max_date})")
                        historical_rows = []

                    if historical_rows:
                        ticker_data = []
                        for item in historical_rows:
                            record = {
                                'gvkey': ticker,
                                'datadate': item['date'],
                                'tic': ticker,
                                'prccd': item['close'],
                                'prcod': item['open'],
                                'prchd': item['high'],
                                'prcld': item['low'],
                                'cshtrd': item['volume'],
                                'adj_close': item.get('adjClose', item['close'])
                            }
                            ticker_data.append(record)
                        
                        if ticker_data:
                            all_data.extend(ticker_data)
                            logger.debug(f"Fetched {len(ticker_data)} records for {ticker} ({min_date} to {max_date})")
                        else:
                            logger.warning(f"No historical data for {ticker} ({min_date} to {max_date})")

                except Exception as e:
                    logger.warning(f"Failed to fetch price data for {ticker} ({min_date} to {max_date}): {e}")
        else:
            logger.warning(f"No price data to fetch")
            return existing_data

        # Step 4: Save new data to database
        if all_data:
            df = pd.DataFrame(all_data)
            df = self._standardize_price_data(df)
            rows_saved = self.data_store.save_price_data(df)
            logger.info(f"Saved {rows_saved} new price records to database")

        # Step 5: Return combined data from database
        final_data = self.data_store.get_price_data(tickers['tickers'].astype(str).tolist() if isinstance(tickers, pd.DataFrame) else list(tickers), start_date, end_date)
        logger.info(f"Returning {len(final_data)} total price records")
        
        return final_data


class DataSourceManager:
    """Manager for multiple data sources with automatic fallback."""

    def __init__(self, cache_dir: str = "./data/cache", preferred_source: Optional[str] = None):
        """
        Initialize DataSourceManager.
        
        Args:
            cache_dir: Directory for caching data
            preferred_source: Preferred data source name ('FMP'), 
                            None for automatic selection
        """
        self.cache_dir = cache_dir
        self.preferred_source = preferred_source

        # Initialize data sources in priority order
        self.data_sources = [
            ('FMP', FMPFetcher(cache_dir))
        ]

        # Determine best available source
        self._select_best_source()

    def _select_best_source(self):
        """Select the best available data source."""
        # If a preferred source is specified, try to use it first
        if self.preferred_source:
            preferred_source_upper = self.preferred_source.upper()
            for name, source in self.data_sources:
                if name.upper() == preferred_source_upper:
                    if source.is_available():
                        self.current_source = source
                        self.current_source_name = name
                        logger.info(f"Using preferred data source: {name}")
                        return
                    else:
                        logger.warning(f"Preferred data source '{name}' is not available, falling back to automatic selection")
                        break
        
        # Automatic selection (priority order)
        for name, source in self.data_sources:
            if source.is_available():
                self.current_source = source
                self.current_source_name = name
                logger.info(f"Selected data source: {name}")
                break

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        """Get S&P 500 components using best available source."""
        return self.current_source.get_sp500_components(date)

    def get_fundamental_data(self, tickers: pd.DataFrame,
                           start_date: str, end_date: str, align_quarter_dates: bool = False) -> pd.DataFrame:
        """Get fundamental data using best available source."""
        return self.current_source.get_fundamental_data(tickers, start_date, end_date, align_quarter_dates)

    def get_price_data(self, tickers: pd.DataFrame,
                      start_date: str, end_date: str) -> pd.DataFrame:
        """Get price data using best available source."""
        return self.current_source.get_price_data(tickers, start_date, end_date)

    def get_news(self, ticker: str, from_date: str, to_date: str,
                 analyze_sentiment: bool = False,
                 sentiment_model: Optional[str] = None,
                 force_refresh: bool = False) -> pd.DataFrame:
        """Get news data using best available source."""
        fetcher = getattr(self.current_source, 'get_news', None)
        if not callable(fetcher):
            raise NotImplementedError(f"{self.current_source_name} does not support news retrieval")
        return fetcher(
            ticker,
            from_date,
            to_date,
            analyze_sentiment=analyze_sentiment,
            sentiment_model=sentiment_model,
            force_refresh=force_refresh
        )

    def get_source_info(self) -> Dict[str, Any]:
        """Get information about current data source."""
        return {
            'current_source': self.current_source_name,
            'available_sources': [name for name, source in self.data_sources if source.is_available()],
            'cache_dir': self.cache_dir
        }


# Global data source manager instance
_data_manager = None
_data_manager_config = {}

def get_data_manager(cache_dir: str = "./data/cache", preferred_source: Optional[str] = None) -> DataSourceManager:
    """
    Get global data source manager instance.
    
    Args:
        cache_dir: Directory for caching data
        preferred_source: Preferred data source name ('FMP'), 
                        None for automatic selection
    
    Returns:
        DataSourceManager instance
        
    Examples:
        # Automatic selection (default)
        manager = get_data_manager()
        
        # Force use FMP (if API key is configured)
        manager = get_data_manager(preferred_source='FMP')
    """
    global _data_manager, _data_manager_config
    
    # Check if we need to recreate the manager
    current_config = {'cache_dir': cache_dir, 'preferred_source': preferred_source}
    
    if _data_manager is None or _data_manager_config != current_config:
        _data_manager = DataSourceManager(cache_dir, preferred_source)
        _data_manager_config = current_config
        
    return _data_manager


# Convenience functions for backward compatibility
def fetch_sp500_tickers(output_path: str = "./data/sp500_tickers.csv", preferred_source='FMP') -> pd.DataFrame:
    """Fetch S&P 500 tickers and save to file."""
    manager = get_data_manager(preferred_source=preferred_source)
    components = manager.get_sp500_components()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_path, index=False)

    logger.info(f"Saved {len(components)} tickers to {output_path}")
    return components


def fetch_nasdaq100_tickers(preferred_source='FMP') -> pd.DataFrame:
    """Fetch NASDAQ 100 tickers from FMP."""
    manager = get_data_manager(preferred_source=preferred_source)
    fetcher = manager.current_source

    url = f"{fetcher.base_url}/nasdaq-constituent?apikey={fetcher.api_key}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame({
        'tickers': [item['symbol'] for item in data],
        'sectors': [item.get('sector', '') for item in data],
        'dateFirstAdded': [item.get('dateFirstAdded') or '' for item in data],
    })
    logger.info(f"Fetched {len(df)} NASDAQ 100 tickers")
    return df


def get_sp500_members_at_date(target_date: str,
                              csv_path: str = None) -> set:
    """Return the set of SP500 tickers as of *target_date*.

    Uses the historical constituents CSV (one row per snapshot date).
    Picks the latest snapshot whose date <= target_date.
    """
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "sp500_historical_constituents.csv")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    target = pd.to_datetime(target_date)
    valid = df[df['date'] <= target]
    if valid.empty:
        return set()
    row = valid.iloc[-1]
    return set(row['tickers'].split(','))


def get_all_historical_sp500_tickers(csv_path: str = None,
                                     start_date: str = "2015-01-01") -> set:
    """Return every ticker that has ever been in SP500 since *start_date*."""
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "sp500_historical_constituents.csv")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= pd.to_datetime(start_date)]
    all_tickers: set = set()
    for tickers_str in df['tickers']:
        all_tickers.update(tickers_str.split(','))
    return all_tickers


def fetch_fundamental_data(tickers: List[str] | pd.DataFrame, start_date: str, end_date: str,
                          align_quarter_dates: bool = False, preferred_source='FMP') -> pd.DataFrame:
    """
    Fetch fundamental data for tickers.
    
    All data is automatically stored in and retrieved from the database.
    No CSV files are created.
    
    Args:
        tickers: List of ticker symbols or DataFrame with tickers and sectors
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        align_quarter_dates: Whether to align quarter dates to Mar/Jun/Sep/Dec 1st
        preferred_source: Preferred data source ('FMP')
        
    Returns:
        DataFrame with fundamental data from database
    """
    manager = get_data_manager(preferred_source=preferred_source)
    if isinstance(tickers, list):
        tickers = pd.DataFrame({'tickers': tickers, 'sectors': [None] * len(tickers), 'dateFirstAdded': [None] * len(tickers)})
    df = manager.get_fundamental_data(tickers, start_date, end_date, align_quarter_dates)
    
    logger.info(f"Retrieved {len(df)} fundamental records from database")
    return df


def fetch_price_data(tickers: List[str] | pd.DataFrame, start_date: str, end_date: str,
                    preferred_source='FMP') -> pd.DataFrame:
    """
    Fetch price data for tickers.
    
    All data is automatically stored in and retrieved from the database.
    No CSV files are created.
    
    Args:
        tickers: List of ticker symbols or DataFrame with tickers and sectors
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        preferred_source: Preferred data source ('FMP')
        
    Returns:
        DataFrame with price data from database
    """
    manager = get_data_manager(preferred_source=preferred_source)
    if isinstance(tickers, list):
        tickers = pd.DataFrame({'tickers': tickers, 'sectors': [None] * len(tickers), 'dateFirstAdded': [None] * len(tickers)})
    df = manager.get_price_data(tickers, start_date, end_date)
    
    logger.info(f"Retrieved {len(df)} price records from database")
    return df


def fetch_news(ticker: str, start_date: str, end_date: str,
               analyze_sentiment: bool = False,
               sentiment_model: Optional[str] = None,
               force_refresh: bool = False,
               preferred_source='FMP') -> pd.DataFrame:
    """
    Fetch news for a ticker with optional GPT情绪分析.

    Args:
        ticker: 股票代码
        start_date: 起始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        analyze_sentiment: 是否调用 GPT 进行情绪分析
        sentiment_model: 覆盖默认 GPT 模型
        force_refresh: 是否忽略缓存强制重新抓取
        preferred_source: 指定数据源

    Returns:
        DataFrame of news articles with sentiment metadata.
    """
    manager = get_data_manager(preferred_source=preferred_source)
    return manager.get_news(
        ticker,
        start_date,
        end_date,
        analyze_sentiment=analyze_sentiment,
        sentiment_model=sentiment_model,
        force_refresh=force_refresh
    )


if __name__ == "__main__":
    # Test the data source manager
    logging.basicConfig(level=logging.INFO)

    manager = get_data_manager()
    info = manager.get_source_info()
    print(f"Data Source Info: {info}")

    # Test fetching data
    components = fetch_sp500_tickers()
    print(f"Fetched {len(components)} tickers")

    # # 按字母顺序对tickers排序
    # tickers = sorted(tickers)
    # tickers = ["AEP", "ADM", "INCY", "LIN", "URI", "NVDA"]
    tickers = ["NVDA"]

    # Fetch sample data
    # end_date = datetime.now().strftime('%Y-%m-%d')
    # fundamentals = fetch_fundamental_data(
    #     components, "2015-10-15", "2025-10-15", align_quarter_dates=True
    # )
    # print(f"Fetched {len(fundamentals)} fundamental records")
    # fundamentals.to_csv("./data/fundamentals.csv", index=False)

    # # Fetch sample data
    # fundamentals = fetch_fundamental_data(
    #     tickers[:5], "2025-06-01", "2025-09-15", align_quarter_dates=True
    # )
    # print(f"Fetched {len(fundamentals)} fundamental records")

    # # Fetch sample data
    # fundamentals = fetch_fundamental_data(
    #     tickers[:5], "2025-09-10", "2025-10-20", align_quarter_dates=True
    # )
    # print(f"Fetched {len(fundamentals)} fundamental records")

    # prices = fetch_price_data(
    #     tickers[:5], "2022-01-01", "2025-12-31"
    # )
    # print(f"Fetched {len(prices)} price records")

    # Fetch news sample with情绪分析（需配置 FMP & OpenAI API Key）
    news_df = fetch_news(
        ticker="NVDA",
        start_date="2025-01-01",
        end_date="2025-01-03",
        analyze_sentiment=True
    )
    print(f"Fetched {len(news_df)} cached+fresh news rows")
    news_df.to_csv("./data/news_nvda.csv", index=False)
    if not news_df.empty:
        preview_cols = ['published_date', 'publisher', 'title', 'sentiment']
        available_cols = [col for col in preview_cols if col in news_df.columns]
        print(news_df[available_cols].head())
