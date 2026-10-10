"""
Trade Executor Module
====================

Executes trading strategies through Alpaca:
- Strategy execution orchestration
- Order management and risk controls
- Multi-account portfolio management
- Execution reporting and monitoring
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    from .alpaca_manager import AlpacaManager, OrderRequest, OrderResponse
    from ..strategies.base_strategy import BaseStrategy, StrategyResult
except ImportError:
    # Fallback for direct module testing
    from alpaca_manager import AlpacaManager, OrderRequest, OrderResponse
    from strategies.base_strategy import BaseStrategy, StrategyResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for trade execution."""
    max_order_value: float = 100000.0  # Maximum value per order
    max_portfolio_turnover: float = 0.5  # Maximum portfolio turnover per rebalance
    min_order_size: float = 100.0  # Minimum order value
    risk_checks_enabled: bool = True
    execution_timeout: int = 300  # Seconds to wait for order fills
    log_orders: bool = True
    order_log_path: str = "./logs/orders"


@dataclass
class ExecutionResult:
    """Result of strategy execution."""
    strategy_name: str
    account_name: str
    orders_placed: List[OrderResponse]
    orders_failed: List[Dict]
    portfolio_before: Dict[str, Any]
    portfolio_after: Dict[str, Any]
    execution_time: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TradeExecutor:
    """Executes trading strategies through Alpaca."""

    def __init__(self, alpaca_manager: AlpacaManager, config: ExecutionConfig = None):
        """
        Initialize trade executor.

        Args:
            alpaca_manager: Alpaca manager instance
            config: Execution configuration
        """
        self.alpaca = alpaca_manager
        self.config = config or ExecutionConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create log directory
        if self.config.log_orders:
            Path(self.config.order_log_path).mkdir(parents=True, exist_ok=True)

    def execute_strategy(self, strategy: BaseStrategy, data: Dict[str, pd.DataFrame],
                        account_name: Optional[str] = None,
                        target_date: Optional[str] = None) -> ExecutionResult:
        """
        Execute a trading strategy.

        Args:
            strategy: Strategy to execute
            data: Market data for strategy
            account_name: Account to execute on
            target_date: Target date for execution

        Returns:
            Execution result
        """
        self.logger.info(f"Executing strategy: {strategy.config.name}")

        start_time = datetime.now()

        # Set account
        if account_name:
            self.alpaca.set_account(account_name)
        else:
            account_name = list(self.alpaca.accounts.keys())[0]

        # Get portfolio state before execution
        portfolio_before = self._get_portfolio_state(account_name)

        # Generate strategy weights
        strategy_result = strategy.generate_weights(data, target_date)

        if len(strategy_result.weights) == 0:
            self.logger.warning("Strategy generated no weights, skipping execution")
            return ExecutionResult(
                strategy_name=strategy.config.name,
                account_name=account_name,
                orders_placed=[],
                orders_failed=[],
                portfolio_before=portfolio_before,
                portfolio_after=portfolio_before,
                execution_time=start_time,
                metadata={'error': 'No weights generated'}
            )

        # Convert weights to order requests
        orders = self._weights_to_orders(strategy_result.weights, account_name)

        # Apply risk checks
        if self.config.risk_checks_enabled:
            orders = self._apply_risk_checks(orders, account_name)

        # Execute orders
        orders_placed, orders_failed = self._execute_orders(orders, account_name)

        # Get portfolio state after execution
        portfolio_after = self._get_portfolio_state(account_name)

        # Log execution
        if self.config.log_orders:
            self._log_execution(strategy_result, orders_placed, orders_failed, account_name)

        execution_time = datetime.now() - start_time

        result = ExecutionResult(
            strategy_name=strategy.config.name,
            account_name=account_name,
            orders_placed=orders_placed,
            orders_failed=orders_failed,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
            execution_time=datetime.now(),
            metadata={
                'execution_duration': execution_time.total_seconds(),
                'orders_total': len(orders_placed) + len(orders_failed),
                'orders_successful': len(orders_placed),
                'orders_failed': len(orders_failed)
            }
        )

        self.logger.info(f"Strategy execution completed: {len(orders_placed)} orders placed, {len(orders_failed)} failed")
        return result

    def execute_portfolio_rebalance(self, target_weights: Dict[str, float],
                                   account_name: Optional[str] = None) -> ExecutionResult:
        """
        Execute portfolio rebalance to target weights.

        Args:
            target_weights: Target portfolio weights
            account_name: Account to execute on

        Returns:
            Execution result
        """
        self.logger.info("Executing portfolio rebalance")

        start_time = datetime.now()

        # Set account
        if account_name:
            self.alpaca.set_account(account_name)
        else:
            account_name = list(self.alpaca.accounts.keys())[0]

        # Get portfolio state before execution
        portfolio_before = self._get_portfolio_state(account_name)

        # Execute rebalance through Alpaca manager
        rebalance_result = self.alpaca.execute_portfolio_rebalance(target_weights, account_name)

        # Convert results to ExecutionResult format
        orders_placed = []
        orders_failed = []

        for order_dict in rebalance_result.get('orders', []):
            if order_dict.get('status') == 'failed':
                orders_failed.append(order_dict)
            else:
                # Create OrderResponse from dict
                orders_placed.append(OrderResponse(**order_dict))

        # Get portfolio state after execution
        portfolio_after = self._get_portfolio_state(account_name)

        # Log execution
        if self.config.log_orders:
            self._log_rebalance(target_weights, orders_placed, orders_failed, account_name)

        execution_time = datetime.now() - start_time

        result = ExecutionResult(
            strategy_name="Portfolio Rebalance",
            account_name=account_name,
            orders_placed=orders_placed,
            orders_failed=orders_failed,
            portfolio_before=portfolio_before,
            portfolio_after=portfolio_after,
            execution_time=datetime.now(),
            metadata={
                'execution_duration': execution_time.total_seconds(),
                'orders_total': len(orders_placed) + len(orders_failed),
                'orders_successful': len(orders_placed),
                'orders_failed': len(orders_failed),
                'target_weights': target_weights
            }
        )

        self.logger.info(f"Portfolio rebalance completed: {len(orders_placed)} orders placed, {len(orders_failed)} failed")
        return result

    def _weights_to_orders(self, weights_df: pd.DataFrame, account_name: str) -> List[OrderRequest]:
        """Convert strategy weights to order requests."""
        orders = []

        # Get current portfolio value
        portfolio_value = self.alpaca.get_portfolio_value(account_name)

        for _, row in weights_df.iterrows():
            gvkey = row['gvkey']
            weight = row['weight']

            # Convert gvkey to ticker (simplified - in practice you'd need a mapping)
            ticker = self._gvkey_to_ticker(gvkey)

            # Calculate target position value
            target_value = portfolio_value * weight

            # Get current position
            current_value = self._get_current_position_value(ticker, account_name)

            # Calculate order value
            order_value = target_value - current_value

            if abs(order_value) >= self.config.min_order_size:
                # Get current price
                try:
                    price = self._get_current_price(ticker, account_name)
                    shares = order_value / price

                    if abs(shares) >= 1:  # Minimum 1 share
                        order = OrderRequest(
                            symbol=ticker,
                            quantity=abs(shares),
                            side='buy' if shares > 0 else 'sell',
                            order_type='market'
                        )
                        orders.append(order)

                except Exception as e:
                    self.logger.warning(f"Could not get price for {ticker}: {e}")

        return orders

    def _apply_risk_checks(self, orders: List[OrderRequest], account_name: str) -> List[OrderRequest]:
        """Apply risk checks to orders."""
        filtered_orders = []

        for order in orders:
            # Check order value limit
            try:
                price = self._get_current_price(order.symbol, account_name)
                order_value = abs(order.quantity * price)

                if order_value > self.config.max_order_value:
                    self.logger.warning(f"Order value ${order_value:.2f} exceeds limit ${self.config.max_order_value:.2f} for {order.symbol}")
                    continue

                filtered_orders.append(order)

            except Exception as e:
                self.logger.warning(f"Could not check risk for {order.symbol}: {e}")
                continue

        # Check portfolio turnover
        total_turnover = sum(
            abs(order.quantity * self._get_current_price(order.symbol, account_name))
            for order in filtered_orders
        )

        portfolio_value = self.alpaca.get_portfolio_value(account_name)
        turnover_ratio = total_turnover / portfolio_value if portfolio_value > 0 else 0

        if turnover_ratio > self.config.max_portfolio_turnover:
            self.logger.warning(f"Portfolio turnover {turnover_ratio:.2%} exceeds limit {self.config.max_portfolio_turnover:.2%}")
            # Scale down orders to meet turnover limit
            scale_factor = self.config.max_portfolio_turnover / turnover_ratio
            for order in filtered_orders:
                order.quantity *= scale_factor

        return filtered_orders

    def _execute_orders(self, orders: List[OrderRequest], account_name: str) -> Tuple[List[OrderResponse], List[Dict]]:
        """Execute orders and handle results."""
        orders_placed = []
        orders_failed = []

        for order in orders:
            try:
                # Place order
                response = self.alpaca.place_order(order, account_name)
                orders_placed.append(response)

                # Wait for fill if required
                if self.config.execution_timeout > 0:
                    final_response = self.alpaca.wait_for_order_fill(
                        response.order_id, self.config.execution_timeout, account_name
                    )
                    if final_response:
                        # Update with final status
                        orders_placed[-1] = final_response

                self.logger.info(f"Executed order: {order.symbol} {order.side} {order.quantity}")

            except Exception as e:
                error_info = {
                    'symbol': order.symbol,
                    'quantity': order.quantity,
                    'side': order.side,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                orders_failed.append(error_info)
                self.logger.error(f"Failed to execute order for {order.symbol}: {e}")

        return orders_placed, orders_failed

    def _get_portfolio_state(self, account_name: str) -> Dict[str, Any]:
        """Get current portfolio state."""
        try:
            account_info = self.alpaca.get_account_info(account_name)
            positions = self.alpaca.get_positions(account_name)

            return {
                'equity': float(account_info.get('equity', 0)),
                'cash': float(account_info.get('cash', 0)),
                'portfolio_value': float(account_info.get('portfolio_value', 0)),
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Could not get portfolio state: {e}")
            return {}

    def _gvkey_to_ticker(self, gvkey: str) -> str:
        """Convert gvkey to ticker symbol."""
        # Simplified mapping - in practice you'd use a lookup table
        ticker_map = {
            '001': 'AAPL',
            '002': 'MSFT',
            '003': 'GOOGL',
            '004': 'AMZN',
            '005': 'META'
        }
        return ticker_map.get(gvkey, gvkey)  # Return gvkey if not found

    def _get_current_position_value(self, ticker: str, account_name: str) -> float:
        """Get current position value for a ticker."""
        try:
            positions = self.alpaca.get_positions(account_name)
            for position in positions:
                if position['symbol'] == ticker:
                    return float(position['market_value'])
        except Exception as e:
            self.logger.warning(f"Could not get position value for {ticker}: {e}")

        return 0.0

    def _get_current_price(self, ticker: str, account_name: str) -> float:
        """Get current price for a ticker."""
        try:
            # Get quote from Alpaca
            account = self.alpaca._get_account(account_name)
            quote = self.alpaca._api_request("GET", f"/v2/stocks/{ticker}/quotes", account=account)
            return float(quote['askprice'])  # Use ask price for buying
        except Exception as e:
            self.logger.warning(f"Could not get price for {ticker}: {e}")
            # Return a default price
            return 100.0

    def _log_execution(self, strategy_result: StrategyResult,
                      orders_placed: List[OrderResponse],
                      orders_failed: List[Dict], account_name: str):
        """Log strategy execution."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config.order_log_path) / f"execution_{timestamp}.json"

        log_data = {
            'timestamp': timestamp,
            'strategy_name': strategy_result.strategy_name,
            'account_name': account_name,
            'weights': strategy_result.weights.to_dict('records') if hasattr(strategy_result.weights, 'to_dict') else strategy_result.weights,
            'orders_placed': [order.__dict__ for order in orders_placed],
            'orders_failed': orders_failed,
            'metadata': strategy_result.metadata
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        self.logger.info(f"Execution logged to {log_file}")

    def _log_rebalance(self, target_weights: Dict[str, float],
                      orders_placed: List[OrderResponse],
                      orders_failed: List[Dict], account_name: str):
        """Log portfolio rebalance."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(self.config.order_log_path) / f"rebalance_{timestamp}.json"

        log_data = {
            'timestamp': timestamp,
            'action': 'portfolio_rebalance',
            'account_name': account_name,
            'target_weights': target_weights,
            'orders_placed': [order.__dict__ for order in orders_placed],
            'orders_failed': orders_failed
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        self.logger.info(f"Rebalance logged to {log_file}")


# Utility functions
def create_trade_executor_from_env() -> TradeExecutor:
    """Create trade executor from environment variables."""
    from src.trading.alpaca_manager import create_alpaca_account_from_env

    account = create_alpaca_account_from_env()
    alpaca_manager = AlpacaManager([account])

    return TradeExecutor(alpaca_manager)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        executor = create_trade_executor_from_env()
        print("Trade executor initialized successfully")
        print(f"Available accounts: {executor.alpaca.get_available_accounts()}")

        # Load latest weights from data/ml_weights.csv and execute rebalance
        weights_path = Path("data/ml_weights.csv")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        df = pd.read_csv(weights_path)
        if "date" not in df.columns or "gvkey" not in df.columns or "weight" not in df.columns:
            raise ValueError("ml_weights.csv should have columns: date, gvkey, weight")

        # Parse date and get latest
        df["date"] = pd.to_datetime(df["date"]).dt.date
        latest_date = df["date"].max()
        df_latest = df[df["date"] == latest_date].copy()

        # Aggregate by symbol (gvkey here is ticker), ensure non-negative, and normalize to 1
        df_latest = df_latest.groupby("gvkey", as_index=False)["weight"].sum()
        df_latest = df_latest[df_latest["weight"] > 0]
        weight_sum = df_latest["weight"].sum()
        if weight_sum <= 0:
            raise ValueError("Sum of weights is 0, cannot execute rebalance")
        df_latest["weight"] = df_latest["weight"] / weight_sum

        target_weights = {row["gvkey"]: float(row["weight"]) for _, row in df_latest.iterrows()}

        print(f"Latest weights date: {latest_date}")
        print(f"Symbols count: {len(target_weights)}")

        # 1) Dry-run: generate order plan without submitting
        plan = executor.alpaca.execute_portfolio_rebalance(
            target_weights,
            account_name="default",
            dry_run=True
        )
        plan_sell = plan.get("orders_plan", {}).get("sell", [])
        plan_buy = plan.get("orders_plan", {}).get("buy", [])
        print(f"Dry-run plan generated. Market open: {plan.get('market_open')} TIF: {plan.get('used_time_in_force')} ")
        print(f"Planned sells: {len(plan_sell)}, buys: {len(plan_buy)}")

        # 2) Submit based on market status and USE_OPG env
        use_opg = os.getenv("USE_OPG", "false").lower() == "true"
        if plan.get("market_open"):
            submit = executor.alpaca.execute_portfolio_rebalance(
                target_weights,
                account_name="default"
            )
            print(f"Submitted during market hours. Orders placed: {submit.get('orders_placed', 0)}")
        else:
            if use_opg:
                submit = executor.alpaca.execute_portfolio_rebalance(
                    target_weights,
                    account_name="default",
                    market_closed_action="opg"
                )
                print(f"Submitted as OPG. Orders placed: {submit.get('orders_placed', 0)}")
            else:
                print("Market is closed. Skipping submission. Set USE_OPG=true to submit OPG at open.")

    except Exception as e:
        print(f"Initialization failed: {e}")
        print("Please check your Alpaca API credentials")
