# -*- coding: utf-8 -*-
"""
Data Store Module
================

Manages data persistence and caching:
- Local database storage
- Incremental data updates
- Cache management
- Data versioning
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from datetime import datetime
import json
import pickle
import sqlite3
import os

import pandas as pd
import numpy as np

from src.data.trading_calendar import (
    get_missing_trading_days,
    consolidate_date_ranges,
    get_trading_days_set,
)

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)


class DataStore:
    """Data store for managing persistent data storage."""

    def __init__(self, base_dir: str = None):
        """
        Initialize data store.

        Args:
            base_dir: Base directory for data storage (uses config.data.base_dir if None)
        """
        # Use config settings if base_dir not provided
        if base_dir is None:
            try:
                from src.config.settings import get_config
                config = get_config()
                base_dir = config.data.base_dir
            except Exception as e:
                logger.warning(f"Failed to load config, using default 'data': {e}")
                base_dir = 'data'
        
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "processed"
        self.db_path = self.base_dir / "finrl_trading.db"

        # Create directories
        for dir_path in [self.base_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, date)
                )
            ''')


            # Create S&P 500 components table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sp500_components_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    tickers TEXT NOT NULL,
                    sectors TEXT NOT NULL,
                    dateFirstAdded TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, tickers, sectors)
                )
            ''')


            # Raw fundamentals (per-row) storage keyed by source/payload/ticker/date
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS raw_payloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    row_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, payload, ticker, date)
                )
            ''')

            # Create news articles table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    published_datetime TEXT NOT NULL,
                    published_date TEXT NOT NULL,
                    publisher TEXT,
                    title TEXT,
                    site TEXT,
                    image TEXT,
                    body TEXT,
                    url TEXT,
                    raw_json TEXT,
                    sentiment TEXT,
                    sentiment_confidence REAL,
                    sentiment_model TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, published_datetime, title)
                )
            ''')

            # Track requested news ranges to avoid repeated API calls
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    record_count INTEGER DEFAULT 0,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, start_date, end_date)
                )
            ''')


            # Fundamental data table (processed quarterly factors)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fundamental_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    datadate TEXT NOT NULL,
                    gsector TEXT,
                    adj_close_q REAL,
                    EPS REAL,
                    BPS REAL,
                    DPS REAL,
                    cur_ratio REAL,
                    quick_ratio REAL,
                    cash_ratio REAL,
                    acc_rec_turnover REAL,
                    debt_ratio REAL,
                    debt_to_equity REAL,
                    pe REAL,
                    ps REAL,
                    pb REAL,
                    roe REAL,
                    net_income_ratio REAL,
                    y_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, datadate)
                )
            ''')

            # Migrate: add metadata columns (idempotent)
            for col in ['filing_date TEXT', 'accepted_date TEXT']:
                try:
                    cursor.execute(f'ALTER TABLE fundamental_data ADD COLUMN {col}')
                except sqlite3.OperationalError:
                    pass

            # Migrate: add new fundamental columns (idempotent)
            _new_fundamental_cols = [
                'gross_margin', 'operating_margin', 'ebitda_margin', 'pretax_margin',
                'effective_tax_rate', 'ebt_per_ebit', 'net_income_per_ebt',
                'asset_turnover', 'fixed_asset_turnover', 'inventory_turnover',
                'payables_turnover', 'wc_turnover',
                'debt_to_assets', 'debt_to_capital', 'lt_debt_to_capital',
                'financial_leverage', 'interest_coverage', 'debt_service_coverage',
                'debt_to_mktcap',
                'fcf_per_share', 'ocf_per_share', 'cash_per_share', 'capex_per_share',
                'fcf_to_ocf', 'ocf_ratio', 'ocf_to_sales', 'ocf_coverage',
                'st_ocf_coverage', 'capex_coverage',
                'revenue_per_share', 'tangible_bvps', 'interest_debt_per_share',
                'peg', 'price_to_fcf', 'price_to_ocf', 'price_to_fair_value',
                'ev_multiple',
                'dividend_payout', 'dividend_yield', 'div_capex_coverage',
                'solvency_ratio',
            ]
            for col in _new_fundamental_cols:
                try:
                    cursor.execute(f'ALTER TABLE fundamental_data ADD COLUMN {col} REAL')
                except sqlite3.OperationalError:
                    pass  # column already exists

            conn.commit()
            logger.info(f"Initialized database at {self.db_path}")


    def save_price_data(self, df: pd.DataFrame) -> int:
        """
        Save price data to database (upsert).
        
        Args:
            df: DataFrame with columns: ticker, datadate, prcod, prchd, prcld, prccd, adj_close, cshtrd
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0
        
        # Standardize column names
        df = df.copy()
        column_mapping = {
            'tic': 'ticker',
            'datadate': 'date',
            'prcod': 'open',
            'prchd': 'high',
            'prcld': 'low',
            'prccd': 'close',
            'cshtrd': 'volume'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'ticker':
                    df['ticker'] = df.get('gvkey', 'UNKNOWN')
                elif col == 'date':
                    df['date'] = df.index if isinstance(df.index, pd.DatetimeIndex) else df.get('datadate')
                else:
                    df[col] = np.nan
        
        # Convert date to string format
        if not isinstance(df['date'].iloc[0], str):
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        rows_affected = 0
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO price_data 
                        (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['ticker'],
                        row['date'],
                        float(row['open']) if pd.notna(row['open']) else None,
                        float(row['high']) if pd.notna(row['high']) else None,
                        float(row['low']) if pd.notna(row['low']) else None,
                        float(row['close']) if pd.notna(row['close']) else None,
                        float(row['adj_close']) if pd.notna(row['adj_close']) else None,
                        float(row['volume']) if pd.notna(row['volume']) else None
                    ))
                    rows_affected += 1
                except Exception as e:
                    logger.warning(f"Failed to save price data for {row.get('ticker')} on {row.get('date')}: {e}")
                    continue
            
            conn.commit()
        
        logger.info(f"Saved {rows_affected} price data records to database")
        return rows_affected

    def get_price_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get price data from database.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with price data
        """
        # Normalize tickers to a plain List[str] to avoid Series + list arithmetic
        if isinstance(tickers, pd.Series):
            tickers_list = tickers.astype(str).tolist()
        elif isinstance(tickers, pd.DataFrame):
            tickers_list = tickers['tickers'].astype(str).tolist() if 'tickers' in tickers.columns else []
        else:
            tickers_list = list(tickers) if tickers is not None else []

        if not tickers_list:
            return pd.DataFrame()

        placeholders = ','.join(['?' for _ in tickers_list])
        query = f'''
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM price_data
            WHERE ticker IN ({placeholders})
            AND date >= ? AND date <= ?
            ORDER BY ticker, date
        '''
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tickers_list + [start_date, end_date])
        
        if not df.empty:
            # Rename to match expected format
            df = df.rename(columns={
                'ticker': 'tic',
                'date': 'datadate',
                'open': 'prcod',
                'high': 'prchd',
                'low': 'prcld',
                'close': 'prccd',
                'volume': 'cshtrd'
            })
            df['gvkey'] = df['tic']
            
        return df

    # =========================
    # News helpers
    # =========================

    def save_news_articles(self, ticker: str, articles: List[Dict[str, Any]]) -> int:
        """Persist news articles."""
        if not ticker or not articles:
            return 0

        rows = []
        for article in articles:
            symbol = article.get('symbol') or ticker
            published_raw = article.get('publishedDate') or article.get('published_datetime') or article.get('date')
            published_ts = pd.to_datetime(published_raw, errors='coerce')
            if pd.isna(published_ts):
                published_ts = pd.Timestamp.utcnow()
            published_datetime = published_ts.strftime('%Y-%m-%d %H:%M:%S')
            published_date = published_ts.strftime('%Y-%m-%d')
            body = article.get('text') or article.get('body') or ''

            rows.append((
                symbol,
                published_datetime,
                published_date,
                article.get('publisher'),
                article.get('title'),
                article.get('site'),
                article.get('image'),
                body,
                article.get('url'),
                json.dumps(article, ensure_ascii=False),
                article.get('sentiment'),
                article.get('sentiment_confidence'),
                article.get('sentiment_model')
            ))

        if not rows:
            return 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO news_articles
                (ticker, published_datetime, published_date, publisher, title, site, image, body, url, raw_json,
                 sentiment, sentiment_confidence, sentiment_model, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', rows)
            conn.commit()

        logger.info(f"Saved {len(rows)} news articles for {ticker}")
        return len(rows)

    def get_news_articles(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load cached news articles."""
        if not ticker:
            return pd.DataFrame()

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT ticker, published_datetime, published_date, publisher, title, site, image, body, url,
                       sentiment, sentiment_confidence, sentiment_model
                FROM news_articles
                WHERE ticker = ?
                  AND published_date >= ?
                  AND published_date <= ?
                ORDER BY published_datetime DESC
            ''', conn, params=(ticker, start_date, end_date))

        return df

    def save_news_fetch_range(self, ticker: str, start_date: str, end_date: str, record_count: int) -> None:
        """Store fetched date ranges for news."""
        if not ticker or not start_date or not end_date:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO news_fetch_log (ticker, start_date, end_date, record_count, fetched_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (ticker, start_date, end_date, int(record_count)))
            conn.commit()

    def get_missing_news_ranges(self, ticker: str, start_date: str, end_date: str) -> List[Tuple[str, str]]:
        """Identify which sub ranges of [start_date, end_date] have not been fetched yet."""
        if not ticker:
            return [(start_date, end_date)]

        req_start = pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date)
        if req_start > req_end:
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT start_date, end_date
                FROM news_fetch_log
                WHERE ticker = ?
                  AND NOT (end_date < ? OR start_date > ?)
                ORDER BY start_date
            ''', (ticker, start_date, end_date))
            rows = cursor.fetchall()

        existing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for row in rows:
            start_ts = pd.to_datetime(row[0])
            end_ts = pd.to_datetime(row[1])
            if pd.isna(start_ts) or pd.isna(end_ts):
                continue
            existing_ranges.append((start_ts, end_ts))

        merged_ranges = self._merge_date_ranges(existing_ranges)

        missing: List[Tuple[str, str]] = []
        pointer = req_start
        for start_ts, end_ts in merged_ranges:
            if end_ts < req_start:
                continue
            if start_ts > req_end:
                break

            cover_start = max(start_ts, req_start)
            cover_end = min(end_ts, req_end)

            if cover_start > pointer:
                gap_end = cover_start - pd.Timedelta(days=1)
                if gap_end >= pointer:
                    missing.append((pointer.strftime('%Y-%m-%d'), gap_end.strftime('%Y-%m-%d')))

            pointer = max(pointer, cover_end + pd.Timedelta(days=1))

        if pointer <= req_end:
            missing.append((pointer.strftime('%Y-%m-%d'), req_end.strftime('%Y-%m-%d')))

        return [rng for rng in missing if pd.to_datetime(rng[0]) <= pd.to_datetime(rng[1])]

    def update_news_sentiment(self, ticker: str, published_datetime: str,
                              sentiment: Optional[str], sentiment_confidence: Optional[float],
                              sentiment_model: Optional[str]) -> None:
        """Update sentiment fields for an existing news record."""
        if not ticker or not published_datetime:
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE news_articles
                   SET sentiment = ?,
                       sentiment_confidence = ?,
                       sentiment_model = ?,
                       updated_at = CURRENT_TIMESTAMP
                 WHERE ticker = ? AND published_datetime = ?
            ''', (sentiment, sentiment_confidence, sentiment_model, ticker, published_datetime))
            conn.commit()

    @staticmethod
    def _merge_date_ranges(ranges: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merge overlapping or adjacent date ranges."""
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda item: item[0])
        merged: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

        for start_ts, end_ts in sorted_ranges:
            if not merged:
                merged.append((start_ts, end_ts))
                continue

            last_start, last_end = merged[-1]
            if start_ts <= last_end + pd.Timedelta(days=1):
                merged[-1] = (last_start, max(last_end, end_ts))
            else:
                merged.append((start_ts, end_ts))

        return merged

    def get_missing_price_dates(self, ticker: str, start_date: str, end_date: str, exchange: str = 'NYSE') -> List[Tuple[str, str]]:
        """
        Identify missing date ranges for price data using real trading calendar.
        
        Args:
            ticker: Ticker symbol
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            exchange: Exchange name (default: NYSE)
        Returns:
            List of (start_date, end_date) tuples for missing ranges
            
        Note:
            - Uses real NYSE trading calendar to determine trading days
            - Only reports missing trading days, excludes weekends and holidays
            - If trading calendar library is not available, falls back to business days
        """
        
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get all existing dates for this ticker in the range
            cursor.execute('''
                SELECT date
                FROM price_data
                WHERE ticker = ? AND date >= ? AND date <= ?
                ORDER BY date
            ''', (ticker, start_date, end_date))
            
            existing_dates = [row[0] for row in cursor.fetchall()]
            
            if not existing_dates:
                # No data exists for this ticker in the range
                return [(start_date, end_date)]
            
            # Use trading calendar to find missing trading days
            missing_trading_days = get_missing_trading_days(
                existing_dates, 
                start_date, 
                end_date,
                exchange=exchange
            )
            
            if not missing_trading_days:
                # No missing trading days
                return []
            
            # Consolidate consecutive dates into ranges
            missing_ranges = consolidate_date_ranges(missing_trading_days)
        
        return missing_ranges

    def get_missing_price_dates_bulk(self, tickers: List[str] | pd.DataFrame, start_date: str, end_date: str, exchange: str = 'NYSE') -> Dict[str, List[Tuple[str, str]]]:
        """
        Bulk version of missing trading date range detection.

        Args:
            tickers: List of symbols or DataFrame with columns 'tickers' and optionally 'dateFirstAdded'
            start_date: YYYY-MM-DD
            end_date: YYYY-MM-DD
            exchange: Trading calendar name

        Returns:
            Dict[ticker, List[(start, end)]]
        """
        results: Dict[str, List[Tuple[str, str]]] = {}
        if tickers is None:
            return results

        # Normalize inputs
        if isinstance(tickers, pd.DataFrame):
            if 'tickers' not in tickers.columns:
                return results
            tickers_list: List[str] = tickers['tickers'].astype(str).tolist()
            # map for per-ticker first-added date (may contain NaN/None)
            try:
                date_first_added_map: Dict[str, Optional[str]] = {
                    r['tickers']: r.get('dateFirstAdded') for _, r in tickers[['tickers', 'dateFirstAdded']].to_dict('index').items()
                }
            except Exception:
                # If dateFirstAdded missing, default to None
                date_first_added_map = {t: None for t in tickers_list}
        else:
            tickers_list = list(tickers)
            date_first_added_map = {t: None for t in tickers_list}

        if not tickers_list:
            return results

        # Pre-compute trading days set ONCE for the whole [start_date, end_date]
        trading_days_all: Set[str] = get_trading_days_set(start_date, end_date, exchange)

        # Fetch existing dates for all tickers in one query to reduce round-trips
        placeholders = ','.join(['?' for _ in tickers_list])
        query = f'''
            SELECT ticker, date
            FROM price_data
            WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?
            ORDER BY ticker, date
        '''

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=tickers_list + [start_date, end_date])

        # Build per-ticker existing date sets
        existing_dates_by_ticker: Dict[str, Set[str]] = {t: set() for t in tickers_list}
        if not df.empty:
            for t, grp in df.groupby('ticker'):
                existing_dates_by_ticker[str(t)] = set(grp['date'].astype(str).tolist())

        # For each ticker, adjust start by dateFirstAdded and compute missing ranges
        for t in tickers_list:
            dfa_raw = date_first_added_map.get(t)
            try:
                dfa = pd.to_datetime(dfa_raw, errors='coerce') if dfa_raw is not None else None
            except Exception:
                dfa = None

            # Effective start = max(global start, dateFirstAdded) if available
            eff_start_dt = max(pd.to_datetime(start_date), dfa) if dfa is not None else pd.to_datetime(start_date)
            eff_end_dt = pd.to_datetime(end_date)

            if eff_start_dt > eff_end_dt:
                results[t] = []
                continue

            eff_start_str = eff_start_dt.strftime('%Y-%m-%d')
            eff_end_str = eff_end_dt.strftime('%Y-%m-%d')

            # Subset precomputed trading days for the effective window
            trading_subset = [d for d in trading_days_all if eff_start_str <= d <= eff_end_str]
            trading_subset_sorted = sorted(trading_subset)

            if not trading_subset_sorted:
                results[t] = []
                continue

            existing_set = existing_dates_by_ticker.get(t, set())
            missing_days_sorted = [d for d in trading_subset_sorted if d not in existing_set]

            if not missing_days_sorted:
                results[t] = []
            else:
                results[t] = consolidate_date_ranges(missing_days_sorted)

        return results


    def save_sp500_components(self, date: str, tickers: str, sectors: str, dateFirstAdded: str) -> bool:
        """
        Save S&P 500 components to database.
        
        Args:
            date: Date for the components (YYYY-MM-DD)
            tickers: Comma-separated ticker string
            sectors: Comma-separated sector string
            dateFirstAdded: Date the component was added to the S&P 500 (YYYY-MM-DD)
        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO sp500_components_details (date, tickers, sectors, dateFirstAdded)
                    VALUES (?, ?, ?, ?)
                ''', (date, tickers, sectors, dateFirstAdded))
                conn.commit()
            
            logger.info(f"Saved S&P 500 components for {date}")
            return True
        except Exception as e:
            logger.error(f"Failed to save S&P 500 components: {e}")
            return False

    def get_sp500_components(self, date: str = None) -> Optional[str]:
        """
        Get S&P 500 components from database.
        
        Args:
            date: Date for the components (latest if None)
            
        Returns:
            Comma-separated ticker string, Comma-separated sector string, Date the component was added to the S&P 500 (YYYY-MM-DD) or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if date:
                cursor.execute('''
                    SELECT tickers, sectors, dateFirstAdded FROM sp500_components_details
                    WHERE date = ?
                ''', (date,))
            else:
                cursor.execute('''
                    SELECT tickers, sectors, dateFirstAdded FROM sp500_components_details
                    ORDER BY date DESC LIMIT 1
                ''')
            
            result = cursor.fetchone()
            if result:
                return result[0], result[1], result[2]
            else:
                return None, None, None

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        total_size = 0
        file_count = 0

        for dir_path in [self.processed_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

        # Add database size
        if self.db_path.exists():
            total_size += self.db_path.stat().st_size

        # Database stats
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM price_data")
            price_count = cursor.fetchone()[0]

        return {
            'total_files': file_count,
            'total_size_mb': total_size / (1024 * 1024),
            'price_records': price_count,
            # 'data_objects': objects_count,
            'database_path': str(self.db_path)
        }

    # =========================
    # Raw fundamentals helpers
    # =========================
    def _find_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify likely date column within a DataFrame."""
        if df is None or df.empty:
            return None

        preferred = [
            'date', 'datadate', 'reportdate', 'reporteddate',
            'filingdate', 'fillingdate', 'calendardate', 'timestamp', 'datetime'
        ]
        lower_map = {col.lower(): col for col in df.columns}

        for candidate in preferred:
            if candidate in lower_map:
                return lower_map[candidate]

        for col in df.columns:
            if col.lower().endswith('date'):
                return col
        return None

    def _save_raw_payload(self, source: str, ticker: Optional[str], payload: str, start_date: Optional[str], end_date: Optional[str], data: Any) -> Optional[str]:
        """
        Save raw fundamentals payload into raw_payloads table.

        Args:
            source: 'FMP' | 'Yahoo' | 'WRDS' etc.
            ticker: Ticker symbol or None for bulk
            payload: e.g., 'income', 'balance', 'cashflow', 'ratios', 'profile', 'yahoo_quarterly_financials'
            start_date: requested start (inclusive, YYYY-MM-DD) or None
            end_date: requested end (inclusive, YYYY-MM-DD) or None
            data: list/dict/DataFrame

        Returns:
            version string if saved, else None
        """
        if data is None:
            return None

        # Normalize to DataFrame where possible; otherwise skip
        obj_to_store: Optional[pd.DataFrame]
        if isinstance(data, pd.DataFrame):
            obj_to_store = data.copy()
        elif isinstance(data, list):
            try:
                obj_to_store = pd.DataFrame(data)
            except Exception:
                obj_to_store = None
        elif isinstance(data, dict):
            try:
                obj_to_store = pd.json_normalize(data)
            except Exception:
                obj_to_store = None
        else:
            obj_to_store = None

        if obj_to_store is None or obj_to_store.empty:
            logger.warning(f"No tabular data to store for raw payload {payload} ({source})")
            return None

        date_column = self._find_date_column(obj_to_store)
        if not date_column:
            logger.warning(f"Skipping raw payload {payload} from {source} for {ticker}: no date column found")
            return None

        try:
            obj_to_store[date_column] = pd.to_datetime(obj_to_store[date_column], errors='coerce')
        except Exception:
            logger.warning(f"Failed to parse date column '{date_column}' for raw payload {payload} ({source})")
            return None

        obj_to_store = obj_to_store.dropna(subset=[date_column])
        if obj_to_store.empty:
            logger.warning(f"All rows invalid after date parsing for raw payload {payload} ({source})")
            return None

        # Filter by requested date range if provided
        try:
            if start_date:
                start_dt = pd.to_datetime(start_date)
                obj_to_store = obj_to_store[obj_to_store[date_column] >= start_dt]
            if end_date:
                end_dt = pd.to_datetime(end_date)
                obj_to_store = obj_to_store[obj_to_store[date_column] <= end_dt]
        except Exception:
            pass

        if obj_to_store.empty:
            logger.warning(f"No rows within requested range for raw payload {payload} of {ticker} ({source})")
            return None

        obj_to_store[date_column] = obj_to_store[date_column].dt.strftime('%Y-%m-%d')
        obj_to_store = obj_to_store.sort_values(date_column).drop_duplicates(subset=[date_column], keep='last')

        # Insert per-row into raw_payloads keyed by source/payload/ticker/date
        params: List[Tuple[str, str, str, str, str]] = []
        for _, row in obj_to_store.iterrows():
            try:
                date_str = str(row[date_column])
                row_dict = row.to_dict()
                params.append((source, payload, ticker or 'bulk', date_str, json.dumps(row_dict)))
            except Exception:
                continue

        if not params:
            return None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany('''
                    INSERT OR REPLACE INTO raw_payloads (source, payload, ticker, date, row_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', params)
                conn.commit()
            return f"raw_{payload}_{ticker or 'bulk'}"
        except Exception as e:
            logger.warning(f"Failed to save raw payload rows for {payload}/{ticker}: {e}")
            return None


    def get_raw_payload(self, ticker: str, payload: str,
                             start_date: str, end_date: str,
                             source: str = 'FMP') -> Optional[List[Dict[str, Any]]]:
        """
        Load saved raw payload rows from raw_payloads in the requested range.

        Returns list[dict] or None when not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT row_json FROM raw_payloads
                   WHERE source = ? AND payload = ? AND ticker = ?
                     AND date >= ? AND date <= ?
                   ORDER BY date''', (source, payload, ticker, start_date, end_date)
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        out: List[Dict[str, Any]] = []
        for (row_json,) in rows:
            try:
                out.append(json.loads(row_json))
            except Exception:
                continue
        return out if out else None

    # =========================
    # Fundamental data helpers
    # =========================
    FUNDAMENTAL_COLS = [
        # Original 14 factors
        'EPS', 'BPS', 'DPS', 'cur_ratio', 'quick_ratio', 'cash_ratio',
        'acc_rec_turnover', 'debt_ratio', 'debt_to_equity', 'pe', 'ps', 'pb',
        'roe', 'net_income_ratio',
        # Profitability (6) — net_income_per_ebt removed (= 1 - effective_tax_rate)
        'gross_margin', 'operating_margin', 'ebitda_margin', 'pretax_margin',
        'effective_tax_rate', 'ebt_per_ebit',
        # Efficiency (5)
        'asset_turnover', 'fixed_asset_turnover', 'inventory_turnover',
        'payables_turnover', 'wc_turnover',
        # Leverage (6) — financial_leverage removed (= debt_to_equity + 1)
        'debt_to_assets', 'debt_to_capital', 'lt_debt_to_capital',
        'interest_coverage', 'debt_service_coverage', 'debt_to_mktcap',
        # Cash Flow (10)
        'fcf_per_share', 'ocf_per_share', 'cash_per_share', 'capex_per_share',
        'fcf_to_ocf', 'ocf_ratio', 'ocf_to_sales', 'ocf_coverage',
        'st_ocf_coverage', 'capex_coverage',
        # Per-Share (3)
        'revenue_per_share', 'tangible_bvps', 'interest_debt_per_share',
        # Valuation (4) — price_to_fair_value removed (= pb)
        'peg', 'price_to_fcf', 'price_to_ocf', 'ev_multiple',
        # Dividend (3)
        'dividend_payout', 'dividend_yield', 'div_capex_coverage',
        # Solvency (1)
        'solvency_ratio',
        # Target
        'y_return',
    ]

    def save_fundamental_data(self, df: pd.DataFrame) -> int:
        """
        Save processed fundamental data to database (upsert).

        Args:
            df: DataFrame from data_fetcher.get_fundamental_data().
                Expected columns: gvkey/tic, datadate, gsector, adj_close_q,
                EPS, BPS, DPS, cur_ratio, … , y_return

        Returns:
            Number of rows inserted/updated
        """
        if df is None or df.empty:
            return 0

        df = df.copy()
        # Normalize ticker column
        if 'ticker' not in df.columns:
            df['ticker'] = df.get('tic', df.get('gvkey', 'UNKNOWN'))
        if 'datadate' not in df.columns and 'date' in df.columns:
            df['datadate'] = df['date']

        if 'datadate' not in df.columns or 'ticker' not in df.columns:
            logger.warning("fundamental_data missing required columns (ticker, datadate)")
            return 0

        # Ensure date string
        df['datadate'] = pd.to_datetime(df['datadate'], errors='coerce').dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['datadate'])

        factor_cols = self.FUNDAMENTAL_COLS
        rows_affected = 0

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                vals = []
                for c in factor_cols:
                    v = row.get(c)
                    vals.append(float(v) if pd.notna(v) else None)

                try:
                    cursor.execute(f'''
                        INSERT OR REPLACE INTO fundamental_data
                        (ticker, datadate, gsector, adj_close_q,
                         filing_date, accepted_date,
                         {", ".join(factor_cols)})
                        VALUES (?, ?, ?, ?, ?, ?, {", ".join(["?"] * len(factor_cols))})
                    ''', (
                        str(row['ticker']),
                        str(row['datadate']),
                        str(row.get('gsector', '')) or None,
                        float(row['adj_close_q']) if pd.notna(row.get('adj_close_q')) else None,
                        str(row.get('filing_date', '')) or None,
                        str(row.get('accepted_date', '')) or None,
                        *vals,
                    ))
                    rows_affected += 1
                except Exception as e:
                    logger.warning(f"Failed to save fundamental row {row.get('ticker')} {row.get('datadate')}: {e}")
                    continue
            conn.commit()

        logger.info(f"Saved {rows_affected} fundamental records to database")
        return rows_affected

    def get_fundamental_data(self, tickers: List[str] = None,
                              start_date: str = None,
                              end_date: str = None) -> pd.DataFrame:
        """
        Load fundamental data from database.

        Args:
            tickers: Optional list of ticker symbols (all if None)
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataFrame with fundamental columns
        """
        conditions = []
        params: list = []

        if tickers:
            if isinstance(tickers, (pd.Series, pd.DataFrame)):
                tickers = list(tickers) if isinstance(tickers, pd.Series) else tickers['tickers'].tolist()
            placeholders = ','.join(['?'] * len(tickers))
            conditions.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start_date:
            conditions.append("datadate >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("datadate <= ?")
            params.append(end_date)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM fundamental_data {where} ORDER BY ticker, datadate"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            # Rename to match fetcher schema
            df = df.rename(columns={'ticker': 'tic'})
            df['gvkey'] = df['tic']

        return df

    def get_raw_payload_latest_date(self, ticker: str, payload: str, source: str = 'FMP') -> Optional[str]:
        """Return the latest available date for a given (source, payload, ticker)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT date FROM raw_payloads
                   WHERE source = ? AND payload = ? AND ticker = ?
                   ORDER BY date DESC LIMIT 1''', (source, payload, ticker)
            )
            row = cursor.fetchone()
            return row[0] if row else None


# Global data store instance
_data_store = None
_data_store_config = {}

def get_data_store(base_dir: str = None) -> DataStore:
    """
    Get global data store instance.
    
    Args:
        base_dir: Base directory for data storage. If None, uses config.data.base_dir
        
    Returns:
        DataStore instance
        
    Note:
        The base_dir is read from config/settings.py (DATA_BASE_DIR environment variable).
        To change the database location, set the DATA_BASE_DIR environment variable
        in your .env file or system environment.
    """
    global _data_store, _data_store_config

    from src.config.settings import get_config
    config = get_config()
    if base_dir is None:
        base_dir = config.data.base_dir

    resolved_base_dir = Path(base_dir)
    if not resolved_base_dir.is_absolute():
        resolved_base_dir = Path(project_root) / resolved_base_dir

    current_config = {'base_dir': str(resolved_base_dir)}
    if _data_store is None or _data_store_config != current_config:
        _data_store = DataStore(str(resolved_base_dir))
        _data_store_config = current_config
        
    return _data_store


if __name__ == "__main__":
    # Intentionally left minimal: avoid fake I/O examples.
    logging.basicConfig(level=logging.INFO)
    store = get_data_store()
    stats = store.get_storage_stats()
    print(f"Database path: {stats['database_path']}")
    print(f"Records - price: {stats['price_records']}")
