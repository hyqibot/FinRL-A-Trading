# 1. updated to set dynamic date range : 
# the start date is the first order date, and the end date is the current date
# the start date is the first order date, and the end date is the current date
# 2. updated to add run_multi_account_performance : 
# loop accounts and print metrics/plot per account using existing functions.
# Returns a dict of {account_name: portfolio_df} for further analysis.
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import logging
from typing import Dict, Optional ,List
# from alpaca_trade_api import REST as AlpacaAPI

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.config.settings import get_config
from src.data.data_fetcher import fetch_price_data  # Assuming this is the fetcher class
from src.trading.alpaca_manager import AlpacaManager, create_alpaca_account_from_env, create_multiple_accounts_from_config
import numpy as np

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252

def get_first_order_date(manager: AlpacaManager) -> Optional[datetime]:
    """Get the date of the first order in the account using AlpacaManager."""
    try:
        orders = manager.get_orders(status='closed', limit=1, direction='asc')
        if orders:
            first_order = orders[0]
            return datetime.fromisoformat(first_order['submitted_at'].replace('Z', '+00:00'))
        else:
            logger.warning("No closed orders found. Using account creation date.")
            account_info = manager.get_account_info()
            creation_date = datetime.fromisoformat(account_info['created_at'].replace('Z', '+00:00'))
            return creation_date
    except Exception as e:
        logger.error(f"Error getting first order date: {e}")
        return None

def get_portfolio_history(manager: AlpacaManager, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get portfolio history from start_date to end_date using AlpacaManager."""
    history = manager.get_portfolio_history(
        timeframe='1D',
        date_start=start_date.date().isoformat(),
        date_end=end_date.date().isoformat(),  
        extended_hours=False
    )
    df = pd.DataFrame({
        'date': pd.to_datetime(history['timestamp'], unit='s'),
        'equity': history['equity'],
        'profit_loss': history['profit_loss'],
        'profit_loss_pct': history['profit_loss_pct']
    })
    print(f"[DEBUG] Portfolio history from {df['date'].min().date()} to {df['date'].max().date()}")

    return df

def get_benchmark_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Get historical price data for SPY and QQQ."""
    #print(f"[DEBUG] Benchmark data from {start_date} to {end_date}")
    print(f"***************start_date is {start_date}******************")
    print(f"***************end_date is {end_date}******************")
    df = fetch_price_data(['SPY', 'QQQ'], start_date, end_date)
    df['date'] = pd.to_datetime(df['datadate'])
    df = df.pivot(index='date', columns='tic', values='adj_close')
    # 保留原有列名并明确列顺序，避免因列顺序变化导致 SPY/QQQ 对调
    df = df[[c for c in ['SPY', 'QQQ'] if c in df.columns]]
    print(df.tail(10))   

    return df

def calculate_returns(df: pd.DataFrame, column: str) -> float:
    """Calculate total return percentage."""
    if df.empty:
        return 0.0
    start_value = df[column].iloc[0]
    end_value = df[column].iloc[-1]
    return (end_value - start_value) / start_value * 100

def _compute_daily_returns(series: pd.Series) -> pd.Series:
    """Compute daily percentage returns from a price/equity series."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(series).astype(float)
    return s.pct_change().dropna()

def compute_performance_metrics(series: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """Compute key performance metrics for a price/equity series.

    Returns metrics as percentages for return/volatility/drawdown and raw value for Sharpe.
    - total_return: %
    - annual_return: % (geometric, annualized from daily returns)
    - annual_volatility: % (std of daily returns * sqrt(252))
    - sharpe_ratio: (annual_return - rf) / annual_volatility
    - max_drawdown: % (min drawdown)
    """
    result = {
        'total_return': 0.0,
        'annual_return': 0.0,
        'annual_volatility': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
    }
    if series is None or len(series) < 2:
        return result

    s = pd.Series(series).astype(float)
    s = s.dropna()
    if len(s) < 2:
        return result

    # Total return
    total_return = (s.iloc[-1] / s.iloc[0] - 1.0) * 100.0

    # Daily returns
    daily_returns = s.pct_change().dropna()
    n = len(daily_returns)
    if n == 0:
        result['total_return'] = float(total_return)
        return result

    # Annualized return (geometric)
    compounded = (1.0 + daily_returns).prod()
    annual_return = (compounded ** (TRADING_DAYS_PER_YEAR / n) - 1.0) * 100.0

    # Annualized volatility
    annual_volatility = daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0

    # Sharpe ratio (use annualized values; rf is annual decimal)
    eps = 1e-12
    sharpe_ratio = 0.0
    if annual_volatility > eps:
        sharpe_ratio = ((annual_return / 100.0) - float(risk_free_rate)) / (annual_volatility / 100.0)

    # Max drawdown
    rolling_max = s.cummax()
    drawdown = s / rolling_max - 1.0
    max_drawdown = float(drawdown.min()) * 100.0

    result.update({
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'annual_volatility': float(annual_volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
    })
    return result

def display_metrics_table(portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame, risk_free_rate: Optional[float] = None):
    """Display performance metrics for Portfolio/Benchmarks using aligned common non-null dates."""
    if portfolio_df.empty or benchmark_df.empty:
        print("No data available for metrics table.")
        return

    try:
        cfg = get_config()
        rfr = risk_free_rate if risk_free_rate is not None else float(cfg.strategy.risk_free_rate)
    except Exception:
        rfr = risk_free_rate if risk_free_rate is not None else 0.02

    # Prepare aligned DataFrame on common dates with no NaNs across selected columns
    ps = portfolio_df[['date', 'equity']].copy()
    ps['date'] = pd.to_datetime(ps['date'])
    ps = ps.set_index('date').sort_index().rename(columns={'equity': 'Portfolio'})

    bs = benchmark_df.copy()
    bs.index = pd.to_datetime(bs.index)
    present_benchmarks = [c for c in ['SPY', 'QQQ'] if c in bs.columns]
    if not present_benchmarks:
        print("No benchmark columns (SPY/QQQ) available for metrics table.")
        return
    bs = bs[present_benchmarks].sort_index()

    combined = pd.concat([ps, bs], axis=1)
    required_cols = ['Portfolio'] + present_benchmarks
    combined = combined.dropna(subset=required_cols)

    if len(combined) < 2:
        print("Not enough aligned data to compute metrics.")
        return

    # Compute metrics using the exact same date set for all series
    columns = ['Portfolio'] + present_benchmarks
    data = {}
    for name in columns:
        data[name] = compute_performance_metrics(combined[name], rfr)

    # Build display DataFrame
    metrics_order = ['total_return', 'annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
    metric_names = {
        'total_return': 'Total Return (%)',
        'annual_return': 'Annual Return (%)',
        'annual_volatility': 'Annual Volatility (%)',
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown (%)',
    }

    rows = []

    # Add aligned period start/end rows
    start_dt = combined.index.min()
    end_dt = combined.index.max()
    start_val = start_dt.date() if hasattr(start_dt, 'date') else start_dt
    end_val = end_dt.date() if hasattr(end_dt, 'date') else end_dt

    start_row = {'Metric': 'Start Date'}
    end_row = {'Metric': 'End Date'}
    for col in columns:
        start_row[col] = f"{start_val}"
        end_row[col] = f"{end_val}"
    rows.append(start_row)
    rows.append(end_row)
    for m in metrics_order:
        row = {'Metric': metric_names[m]}
        for col in columns:
            val = data[col][m]
            if m == 'sharpe_ratio':
                row[col] = f"{val:.2f}"
            else:
                row[col] = f"{val:.2f}%"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    print("\nPerformance Metrics (Aligned):")
    print(summary_df.to_string(index=False))

def display_table(portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame):
    """Display summary table."""
    if portfolio_df.empty or benchmark_df.empty:
        print("No data available for table.")
        return

    summary = {
        'Metric': ['Start Date', 'End Date', 'Starting Value', 'Ending Value', 'Total Return (%)'],
        'Portfolio': [
            portfolio_df['date'].iloc[0].date(),
            portfolio_df['date'].iloc[-1].date(),
            f"${portfolio_df['equity'].iloc[0]:.2f}",
            f"${portfolio_df['equity'].iloc[-1]:.2f}",
            f"{calculate_returns(portfolio_df, 'equity'):.2f}%"
        ],
        'SPY': [
            benchmark_df.index[0].date(),
            benchmark_df.index[-1].date(),
            f"${benchmark_df['SPY'].iloc[0]:.2f}",
            f"${benchmark_df['SPY'].iloc[-1]:.2f}",
            f"{calculate_returns(benchmark_df, 'SPY'):.2f}%"
        ],
        'QQQ': [
            benchmark_df.index[0].date(),
            benchmark_df.index[-1].date(),
            f"${benchmark_df['QQQ'].iloc[0]:.2f}",
            f"${benchmark_df['QQQ'].iloc[-1]:.2f}",
            f"{calculate_returns(benchmark_df, 'QQQ'):.2f}%"
        ]
    }
    summary_df = pd.DataFrame(summary)
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))

def plot_performance(portfolio_df: pd.DataFrame, benchmark_df: pd.DataFrame):
    """Plot normalized performance using aligned dates across Portfolio and Benchmarks."""
    if portfolio_df.empty or benchmark_df.empty:
        print("No data available for plotting.")
        return

    # Align on common dates with no NaNs across present columns
    ps = portfolio_df[['date', 'equity']].copy()
    ps['date'] = pd.to_datetime(ps['date'])
    ps = ps.set_index('date').sort_index().rename(columns={'equity': 'Portfolio'})

    bs = benchmark_df.copy()
    bs.index = pd.to_datetime(bs.index)
    present_benchmarks = [c for c in ['SPY', 'QQQ'] if c in bs.columns]
    if not present_benchmarks:
        print("No benchmark columns (SPY/QQQ) available for plotting.")
        return
    bs = bs[present_benchmarks].sort_index()

    combined = pd.concat([ps, bs], axis=1)
    required_cols = ['Portfolio'] + present_benchmarks
    combined = combined.dropna(subset=required_cols)
    if len(combined) < 2:
        print("Not enough aligned data to plot.")
        return

    # Normalize to start at 1
    normalized = combined[required_cols].apply(lambda s: s / s.iloc[0])

    plt.figure(figsize=(12, 6))
    plt.plot(combined.index, normalized['Portfolio'], label='Portfolio', linewidth=2)
    for name in present_benchmarks:
        plt.plot(combined.index, normalized[name], label=name, linestyle='--')
    plt.title('Normalized Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def _build_manager_from_env() -> AlpacaManager:
    """
    get nicknames from .env , and create multiple accounts from the config.
    get API key/secret/base_url from the config.
    """
    names_raw = os.getenv("APCA_ACCOUNTS", "").strip()
    if not names_raw:
        # if no multi-account is configured, use the single account default
        account = create_alpaca_account_from_env(name="default")
        return AlpacaManager([account])

    default_base = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets")
    cfg: Dict[str, Dict[str, str]] = {}
    for raw in [n.strip() for n in names_raw.split(",") if n.strip()]:
        tag = raw.upper()
        key = os.getenv(f"APCA_{tag}_API_KEY") or os.getenv("APCA_API_KEY")
        sec = os.getenv(f"APCA_{tag}_API_SECRET") or os.getenv("APCA_API_SECRET")
        base = os.getenv(f"APCA_{tag}_BASE_URL") or default_base
        if not key or not sec:
            raise ValueError(f"Missing API key/secret for account '{raw}' in .env")
        cfg[raw] = {"api_key": key, "api_secret": sec, "base_url": base}

    accounts = create_multiple_accounts_from_config(cfg)
    return AlpacaManager(accounts)

def run_multi_account_performance(manager: AlpacaManager,
                                  account_names: Optional[List[str]] = None,
                                  plot: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Loop accounts and print metrics/plot per account using existing functions.
    Returns a dict of {account_name: portfolio_df} for further analysis.
    """
    if account_names is None:
        account_names = manager.get_available_accounts()

    results: Dict[str, pd.DataFrame] = {}
    for name in account_names:
        try:
            manager.set_account(name)

            # per-account date window: first order - 1 day -> now
            start = get_first_order_date(manager)
            if not start:
                print(f"[{name}] No order history. Skip.")
                continue
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            start = start - timedelta(days=1)
            end = datetime.now(timezone.utc)

            # fetch data
            pf = get_portfolio_history(manager, start, end)
            bench = get_benchmark_data(start.date().isoformat(),
                                       (end + timedelta(days=1)).date().isoformat())
            # align benchmark to portfolio dates (reuse existing alignment logic downstream)

            if not bench.empty and not pf.empty:
                mask = np.isin(bench.index.date, pf['date'].dt.date.unique())
                bench = bench[mask]

            print(f"\n===== {name} =====")
            if pf.empty or bench.empty:
                print("No data available for this account/benchmark window.")
                continue

            # reuse existing displays
            display_metrics_table(pf, bench)
            if plot:
                plot_performance(pf, bench)

            results[name] = pf

        except Exception as e:
            print(f"[{name}] performance fetch failed: {e}")

    return results
'''
signal accounts from .env, and create a manager from the config.
def main():
    logging.basicConfig(level=logging.INFO)
    
    # Create AlpacaManager instance
    account = create_alpaca_account_from_env()
    manager = AlpacaManager([account])
    
    # Use dynamic date range: from first order date (minus 1 day) to now
    end_date = datetime.now(timezone.utc)
    first_order_date = get_first_order_date(manager)
    
    if not first_order_date:
        logger.error("Could not determine start date. Exiting.")
        return
        
    # Adjust start date to be at least 1 day before for data fetching
    if first_order_date.tzinfo is None:
        first_order_date = first_order_date.replace(tzinfo=timezone.utc)
    start_date = first_order_date - timedelta(days=1)
    start_date_str = start_date.date().isoformat()
    fmp_end_date = end_date + timedelta(days=1)
    end_date_str = fmp_end_date.date().isoformat()
    
    logger.info(f"Fetching data from {start_date_str} to {end_date_str} for FMP (adjusted to include end_date)")
    
    portfolio_df = get_portfolio_history(manager, start_date, end_date)
    logger.info(f"Portfolio data dates: from {portfolio_df['date'].min().date()} to {portfolio_df['date'].max().date()}")
    benchmark_df = get_benchmark_data(start_date_str, end_date_str)
    if not benchmark_df.empty:
        logger.info(f"Benchmark data dates: from {benchmark_df.index.min().date()} to {benchmark_df.index.max().date()}")
    
    # Align dates if necessary
    common_dates = portfolio_df['date'].dt.date.unique()
    if not benchmark_df.empty:
        mask = np.in1d(benchmark_df.index.date, common_dates)
        benchmark_df = benchmark_df[mask]
    
    # display_table(portfolio_df, benchmark_df)
    display_metrics_table(portfolio_df, benchmark_df)
    plot_performance(portfolio_df, benchmark_df)
'''
#multi-account performance analyzer
def main():
    logging.basicConfig(level=logging.INFO)

    manager = _build_manager_from_env()
    names: List[str] = manager.get_available_accounts()
    print(f"Accounts loaded: {names}")

    for name in names:
        try:
            manager.set_account(name)

            # per-account date window: first order - 1 day -> now
            start = get_first_order_date(manager)
            if not start:
                print(f"[{name}] No order history. Skip.")
                continue
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            start = start - timedelta(days=1)
            end = datetime.now(timezone.utc)
            print(f"start is {start}")
            print(f"end is { end.date().isoformat()}")
            print(f'input end-data is {(end + timedelta(days=1)).date().isoformat()}')

            # fetch data
            pf = get_portfolio_history(manager, start, end)
            bench = get_benchmark_data(start.date().isoformat(), (end + timedelta(days=1)).date().isoformat())

           # print("\n[DEBUG] Raw Benchmark (FMP) Data:")
           # print(bench.tail(10))   
            if not bench.empty and not pf.empty:
                    if len(bench) > len(pf):
                        bench = bench.iloc[:-1]
            if not bench.empty and not pf.empty:
                mask = np.isin(bench.index.date, pf['date'].dt.date.unique())
                bench = bench[mask]

            #print("\n[DEBUG] Raw Benchmark (FMP) Data:")
            #print(bench.tail(10))   

            print(f"\n===== {name} =====")
            if pf.empty or bench.empty:
                print("No data available for this account/benchmark window.")
                continue

            display_metrics_table(pf, bench)
            plot_performance(pf, bench)

        except Exception as e:
            print(f"[{name}] performance fetch failed: {e}")

if __name__ == "__main__":
    main()
