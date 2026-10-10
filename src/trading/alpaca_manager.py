"""
Alpaca Trading Manager Module
============================

Manages Alpaca API interactions:
- Multi-account support
- Order execution and management
- Position tracking
- Risk management
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import time
import pandas as pd

import requests
from dotenv import load_dotenv
import os
import math

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(f"project_root: {project_root}")
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)


@dataclass
class AlpacaAccount:
    """Alpaca account configuration."""
    name: str
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"

    @property
    def is_paper(self) -> bool:
        """Check if this is a paper trading account."""
        return "paper" in self.base_url


@dataclass
class OrderRequest:
    """Order request structure."""
    symbol: str
    quantity: float
    side: str  # 'buy' or 'sell'
    order_type: str = 'market'  # 'market', 'limit', 'stop', 'stop_limit'
    time_in_force: str = 'day'  # 'day', 'gtc', 'opg', 'cls', 'ioc', 'fok'
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    extended_hours: bool = False


@dataclass
class OrderResponse:
    """Order response structure."""
    order_id: str
    status: str
    symbol: str
    quantity: float
    filled_quantity: float
    side: str
    order_type: str
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    average_fill_price: Optional[float] = None


class AlpacaManager:
    """Manager for Alpaca trading operations."""

    def __init__(self, accounts: List[AlpacaAccount]):
        """
        Initialize Alpaca manager.

        Args:
            accounts: List of Alpaca account configurations
        """
        self.accounts = {acc.name: acc for acc in accounts}
        self.current_account = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Cache for asset metadata: symbol -> asset dict
        self._asset_cache: Dict[str, Dict[str, Any]] = {}
        self._assets_loaded: bool = False

        # Load environment variables
        load_dotenv()

        # Set default account if only one provided
        if len(self.accounts) == 1:
            self.current_account = list(self.accounts.values())[0]

    def set_account(self, account_name: str):
        """Set the current active account."""
        if account_name not in self.accounts:
            raise ValueError(f"Account '{account_name}' not found")
        self.current_account = self.accounts[account_name]
        self.logger.info(f"Switched to account: {account_name}")

    def get_account_info(self, account_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get account information.

        Args:
            account_name: Account name (uses current if None)

        Returns:
            Account information dictionary
        """
        account = self._get_account(account_name)
        return self._api_request("GET", "/v2/account", account=account)

    def get_positions(self, account_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get current positions.

        Args:
            account_name: Account name (uses current if None)

        Returns:
            List of position dictionaries
        """
        account = self._get_account(account_name)
        return self._api_request("GET", "/v2/positions", account=account)

    def get_portfolio_value(self, account_name: Optional[str] = None) -> float:
        """
        Get current portfolio value.

        Args:
            account_name: Account name (uses current if None)

        Returns:
            Portfolio value
        """
        account_info = self.get_account_info(account_name)
        return float(account_info.get('portfolio_value', 0))

    def get_orders(self, status: str = 'all', limit: int = 500, direction: str = 'desc',
                   account_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of orders.

        Args:
            status: Order status filter ('open', 'closed', 'all')
            limit: Maximum number of orders to return
            direction: Sort direction ('asc' or 'desc')
            account_name: Account name (uses current if None)

        Returns:
            List of order dictionaries
        """
        account = self._get_account(account_name)
        params = {
            'status': status,
            'limit': limit,
            'direction': direction
        }
        return self._api_request("GET", "/v2/orders", params=params, account=account)

    def get_portfolio_history(self, timeframe: str = '1D', period: Optional[str] = None,
                              date_start: Optional[str] = None, date_end: Optional[str] = None,
                              extended_hours: bool = False, account_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get portfolio history.

        Args:
            timeframe: Timeframe ('1Min', '5Min', '15Min', '1H', '1D')
            period: Period (e.g., '1D', '1W', '1M')
            date_start: Start date (YYYY-MM-DD)
            date_end: End date (YYYY-MM-DD)
            extended_hours: Include extended hours
            account_name: Account name (uses current if None)

        Returns:
            Portfolio history dictionary
        """
        account = self._get_account(account_name)
        params = {
            'timeframe': timeframe,
            'extended_hours': extended_hours
        }
        if period:
            params['period'] = period
        if date_start:
            params['date_start'] = date_start
        if date_end:
            params['date_end'] = date_end
        return self._api_request("GET", "/v2/account/portfolio/history", params=params, account=account)

    def place_order(self, order: OrderRequest,
                   account_name: Optional[str] = None) -> OrderResponse:
        """
        Place an order.

        Args:
            order: Order request
            account_name: Account name (uses current if None)

        Returns:
            Order response
        """
        account = self._get_account(account_name)

        # Build order payload
        # Fractional shares must be DAY orders on Alpaca. If quantity is fractional and TIF is not DAY,
        # force time_in_force to 'day'.
        try:
            is_fractional_qty = abs(order.quantity - round(order.quantity)) > 1e-6
        except Exception:
            is_fractional_qty = False

        tif = (order.time_in_force or 'day').lower()
        if is_fractional_qty and tif != 'day':
            self.logger.info(
                f"Fractional order detected for {order.symbol} qty={order.quantity:.6f}; overriding time_in_force '{tif}' -> 'day'"
            )
            tif = 'day'

        payload = {
            'symbol': order.symbol.upper(),
            'qty': str(order.quantity),
            'side': order.side.lower(),
            'type': order.order_type.lower(),
            'time_in_force': tif,
            'extended_hours': order.extended_hours
        }

        if order.limit_price is not None:
            payload['limit_price'] = str(order.limit_price)
        if order.stop_price is not None:
            payload['stop_price'] = str(order.stop_price)

        # Place order
        response = self._api_request("POST", "/v2/orders", json_body=payload, account=account)

        # Convert to OrderResponse
        return OrderResponse(
            order_id=response['id'],
            status=response['status'],
            symbol=response['symbol'],
            quantity=float(response['qty']),
            filled_quantity=float(response.get('filled_qty', 0)),
            side=response['side'],
            order_type=response['type'],
            submitted_at=pd.to_datetime(response['submitted_at']),
            filled_at=pd.to_datetime(response['filled_at']) if response.get('filled_at') else None,
            average_fill_price=float(response.get('filled_avg_price', 0)) if response.get('filled_avg_price') else None
        )

    def place_orders_batch(self, orders: List[OrderRequest],
                          account_name: Optional[str] = None) -> List[OrderResponse]:
        """
        Place multiple orders.

        Args:
            orders: List of order requests
            account_name: Account name (uses current if None)

        Returns:
            List of order responses
        """
        responses = []
        for order in orders:
            try:
                response = self.place_order(order, account_name)
                responses.append(response)
                self.logger.info(f"Placed order: {order.symbol} {order.side} {order.quantity}")

                # Small delay to avoid rate limits
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Failed to place order for {order.symbol}: {e}")
                # Create failed order response
                responses.append(OrderResponse(
                    order_id="",
                    status="failed",
                    symbol=order.symbol,
                    quantity=order.quantity,
                    filled_quantity=0,
                    side=order.side,
                    order_type=order.order_type,
                    submitted_at=datetime.now()
                ))

        return responses

    def cancel_order(self, order_id: str, account_name: Optional[str] = None) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            account_name: Account name (uses current if None)

        Returns:
            True if cancelled successfully
        """
        account = self._get_account(account_name)
        try:
            self._api_request("DELETE", f"/v2/orders/{order_id}", account=account)
            self.logger.info(f"Cancelled order: {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self, account_name: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            account_name: Account name (uses current if None)

        Returns:
            Number of orders cancelled
        """
        account = self._get_account(account_name)
        try:
            response = self._api_request("DELETE", "/v2/orders", account=account)
            cancelled_count = len(response) if isinstance(response, list) else 0
            self.logger.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            return 0

    def get_order_status(self, order_id: str, account_name: Optional[str] = None) -> Optional[OrderResponse]:
        """
        Get order status.

        Args:
            order_id: Order ID
            account_name: Account name (uses current if None)

        Returns:
            Order response if found, None otherwise
        """
        account = self._get_account(account_name)
        try:
            response = self._api_request("GET", f"/v2/orders/{order_id}", account=account)
            return OrderResponse(
                order_id=response['id'],
                status=response['status'],
                symbol=response['symbol'],
                quantity=float(response['qty']),
                filled_quantity=float(response.get('filled_qty', 0)),
                side=response['side'],
                order_type=response['type'],
                submitted_at=pd.to_datetime(response['submitted_at']),
                filled_at=pd.to_datetime(response['filled_at']) if response.get('filled_at') else None,
                average_fill_price=float(response.get('filled_avg_price', 0)) if response.get('filled_avg_price') else None
            )
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {e}")
            return None

    def wait_for_order_fill(self, order_id: str, timeout_seconds: int = 300,
                           account_name: Optional[str] = None) -> Optional[OrderResponse]:
        """
        Wait for an order to be filled.

        Args:
            order_id: Order ID
            timeout_seconds: Maximum time to wait
            account_name: Account name (uses current if None)

        Returns:
            Final order response
        """
        start_time = time.time()
        terminal_statuses = {'filled', 'canceled', 'rejected', 'expired'}

        while time.time() - start_time < timeout_seconds:
            order = self.get_order_status(order_id, account_name)
            if order and order.status in terminal_statuses:
                return order

            time.sleep(2)  # Check every 2 seconds

        self.logger.warning(f"Order {order_id} did not reach terminal status within {timeout_seconds} seconds")
        return None

    def execute_portfolio_rebalance(self, target_weights: Dict[str, float],
                                   account_name: Optional[str] = None,
                                   dry_run: bool = False,
                                   market_closed_action: str = 'skip') -> Dict[str, Any]:
        """
        Execute portfolio rebalance to target weights.

        This adjusts current holdings to match the provided target weights.
        Symbols not present in target_weights will be treated as target weight 0
        and thus will be fully sold if currently held.

        Args:
            target_weights: Dictionary of symbol -> target weight
            account_name: Account name (uses current if None)
            dry_run: If True, do not place orders; only return planned orders
            market_closed_action: Behavior when market is closed: 'skip' (default) or 'opg'

        Returns:
            Rebalance execution results
        """
        account = self._get_account(account_name)

        # Detect market status and decide order TIF
        is_open = self._is_market_open()
        use_opg = (not is_open) and (market_closed_action == 'opg')
        default_tif = 'opg' if use_opg else 'day'

        # Only cancel open orders when we intend to place new ones
        will_place_orders = (not dry_run) and (is_open or use_opg)
        if will_place_orders:
            try:
                self.cancel_all_orders(account_name)
            except Exception as e:
                self.logger.warning(f"Failed to cancel open orders before rebalance: {e}")

        # Ensure asset metadata is available
        self._ensure_assets_loaded()

        # Get current positions and portfolio value
        positions = self.get_positions(account_name)
        portfolio_value = self.get_portfolio_value(account_name)

        # Calculate current weights
        current_weights = {}
        for position in positions:
            symbol = position['symbol']
            market_value = float(position['market_value'])
            current_weights[symbol] = (market_value / portfolio_value) if portfolio_value > 0 else 0.0

        # Filter与规范化: 仅保留可交易标的；负权重视为0；必要时归一化到<=1
        filtered_targets: Dict[str, float] = {}
        raw_targets = target_weights or {}
        for s, w in raw_targets.items():
            try:
                w_float = float(w)
            except Exception:
                self.logger.warning(f"Invalid weight for {s}: {w}; treating as 0")
                w_float = 0.0
            if w_float < 0:
                self.logger.info(f"Negative weight for {s} -> clamped to 0")
                w_float = 0.0
            if self._is_symbol_tradable(s):
                filtered_targets[s] = w_float
            else:
                self.logger.info(f"Skipping non-tradable or inactive asset: {s}")

        sum_w = sum(filtered_targets.values())
        used_target_weights: Dict[str, float]
        if sum_w > 1.0001:
            # 若权重和>1，按总和进行缩放归一化
            used_target_weights = {s: (w / sum_w) for s, w in filtered_targets.items()}
            self.logger.info(f"Target weights sum {sum_w:.6f} > 1; normalized to 1.0 proportionally")
        else:
            # 和<=1则保留，允许剩余现金
            used_target_weights = dict(filtered_targets)

        all_symbols = set(current_weights.keys()) | set(used_target_weights.keys())
        full_target_weights: Dict[str, float] = {s: float(used_target_weights.get(s, 0.0)) for s in all_symbols}

        # Phase 1: build SELL orders first (reduce positions to free cash)
        sell_orders = []
        # Build quick map for position quantities
        pos_qty_map: Dict[str, float] = {}
        for p in positions:
            qty_avl = p.get('qty_available', p.get('qty', '0'))
            try:
                pos_qty_map[p['symbol']] = float(qty_avl)
            except Exception:
                try:
                    pos_qty_map[p['symbol']] = float(p.get('qty', '0'))
                except Exception:
                    pos_qty_map[p['symbol']] = 0.0

        for symbol, target_weight in full_target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            if weight_diff < -0.001:
                target_value = portfolio_value * target_weight
                current_value = portfolio_value * current_weight
                value_to_trade = target_value - current_value  # negative

                price = None
                try:
                    price = self._get_latest_price(symbol, account=account)
                except Exception:
                    print(f"Failed to get latest price for {symbol}: {e}")
                    price = None
                if price is None:
                    try:
                        # fallback to position avg price if available
                        pos_map = {p['symbol']: p for p in positions}
                        if symbol in pos_map and pos_map[symbol].get('avg_entry_price'):
                            price = float(pos_map[symbol]['avg_entry_price'])
                    except Exception:
                        price = None
                if price is None:
                    price = 100.0

                shares_to_trade = value_to_trade / price  # negative
                shares_abs = abs(shares_to_trade)
                # Cap to available quantity
                available_qty = max(0.0, pos_qty_map.get(symbol, 0.0))
                shares_abs = min(shares_abs, available_qty)

                # Rounding for non-fractionable assets
                if not self._is_symbol_fractionable(symbol):
                    shares_abs = math.floor(shares_abs)

                # Respect minimum share quantity
                if shares_abs >= (0.0001 if self._is_symbol_fractionable(symbol) else 1.0):
                    sell_orders.append(OrderRequest(
                        symbol=symbol,
                        quantity=shares_abs,
                        side='sell',
                        order_type='market'
                    ))

        results_sell = []
        if dry_run or (not will_place_orders):
            # no side effects; keep state as-is for planning
            pass
        else:
            if sell_orders:
                # Apply time_in_force
                for o in sell_orders:
                    o.time_in_force = default_tif
                results_sell = self.place_orders_batch(sell_orders, account_name)
                self.logger.info(f"Executed {len(results_sell)} sell orders in rebalance phase 1")

            # Refresh state after sells
            positions = self.get_positions(account_name)
            portfolio_value = self.get_portfolio_value(account_name)
            current_weights = {}
            for position in positions:
                symbol = position['symbol']
                market_value = float(position['market_value'])
                current_weights[symbol] = (market_value / portfolio_value) if portfolio_value > 0 else 0.0

        # Phase 2: build BUY orders with buying power scaling and rounding
        from src.config.settings import get_config
        cfg = get_config()
        min_notional = float(getattr(cfg.trading, 'min_order_size', 100.0))

        buy_candidates = []  # list of (symbol, price, desired_value)
        for symbol, target_weight in full_target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            if weight_diff > 0.001:
                target_value = portfolio_value * target_weight
                current_value = portfolio_value * current_weight
                value_to_trade = max(0.0, target_value - current_value)
                price = None
                try:
                    price = self._get_latest_price(symbol, account=account)
                except Exception:
                    price = None
                if price is None:
                    try:
                        pos_map = {p['symbol']: p for p in positions}
                        if symbol in pos_map and pos_map[symbol].get('avg_entry_price'):
                            price = float(pos_map[symbol]['avg_entry_price'])
                    except Exception:
                        price = None
                if price is None:
                    price = 100.0

                if value_to_trade > 0 and price > 0:
                    buy_candidates.append((symbol, price, value_to_trade))

        total_desired = sum(v for _, _, v in buy_candidates)
        # Get current buying power, apply safety buffer
        try:
            acct = self.get_account_info(account_name)
            buying_power = float(acct.get('buying_power', 0))
        except Exception:
            buying_power = 0.0
        budget = max(0.0, buying_power * 0.98)
        scale = min(1.0, (budget / total_desired)) if total_desired > 0 else 0.0

        buy_orders = []
        for symbol, price, desired_value in buy_candidates:
            scaled_value = desired_value * scale
            if scaled_value < min_notional:
                continue
            qty = scaled_value / price
            if not self._is_symbol_fractionable(symbol):
                qty = math.floor(qty)
            # Minimum quantity check
            if qty >= (0.0001 if self._is_symbol_fractionable(symbol) else 1.0):
                buy_orders.append(OrderRequest(
                    symbol=symbol,
                    quantity=qty,
                    side='buy',
                    order_type='market'
                ))

        results_buy = []
        if dry_run or (not will_place_orders):
            pass
        else:
            if buy_orders:
                # Apply time_in_force
                for o in buy_orders:
                    o.time_in_force = default_tif
                results_buy = self.place_orders_batch(buy_orders, account_name)
                self.logger.info(f"Executed {len(results_buy)} buy orders in rebalance phase 2")

        # If dry-run or market closed and skipping, return plan only
        if dry_run or ((not is_open) and (market_closed_action == 'skip')):
            def serialize_order_req(o: OrderRequest) -> Dict[str, Any]:
                return {
                    'symbol': o.symbol,
                    'quantity': float(o.quantity),
                    'side': o.side,
                    'order_type': o.order_type,
                    'time_in_force': default_tif
                }
            return {
                'orders_placed': 0,
                'orders': [],
                'orders_plan': {
                    'sell': [serialize_order_req(o) for o in sell_orders],
                    'buy': [serialize_order_req(o) for o in buy_orders]
                },
                'market_open': is_open,
                'used_time_in_force': default_tif,
                'target_weights': used_target_weights
            }

        all_results = results_sell + results_buy
        if all_results:
            return {
                'orders_placed': len(all_results),
                'orders': [r.__dict__ for r in all_results],
                'target_weights': used_target_weights,
                'market_open': is_open,
                'used_time_in_force': default_tif
            }
        else:
            self.logger.info("No rebalance orders needed")
            return {
                'orders_placed': 0,
                'orders': [],
                'target_weights': used_target_weights,
                'market_open': is_open,
                'used_time_in_force': default_tif
            }

    def _ensure_assets_loaded(self):
        """Load and cache active assets list once."""
        if self._assets_loaded:
            return
        try:
            assets = self._api_request("GET", "/v2/assets", params={"status": "active"}, account=self._get_account())
            if isinstance(assets, list):
                for a in assets:
                    sym = a.get('symbol')
                    if sym:
                        self._asset_cache[sym.upper()] = a
                self._assets_loaded = True
                self.logger.info(f"Cached {len(self._asset_cache)} active assets from Alpaca")
        except Exception as e:
            self.logger.warning(f"Failed to load assets list: {e}")

    def _get_asset_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get asset info for a symbol, using cache then API fallback."""
        sym = (symbol or "").upper()
        if sym in self._asset_cache:
            return self._asset_cache[sym]
        try:
            info = self._api_request("GET", f"/v2/assets/{sym}", account=self._get_account())
            if isinstance(info, dict):
                self._asset_cache[sym] = info
                return info
        except Exception:
            return None
        return None

    def _is_symbol_tradable(self, symbol: str) -> bool:
        info = self._get_asset_info(symbol)
        if not info:
            return False
        return bool(info.get('tradable', False)) and (info.get('status') == 'active')

    def _is_symbol_fractionable(self, symbol: str) -> bool:
        info = self._get_asset_info(symbol)
        if not info:
            return False
        return bool(info.get('fractionable', False))

    def _get_market_clock(self) -> Optional[Dict[str, Any]]:
        try:
            return self._api_request("GET", "/v2/clock", account=self._get_account())
        except Exception as e:
            self.logger.warning(f"Failed to get market clock: {e}")
            return None

    def _is_market_open(self) -> bool:
        clock = self._get_market_clock()
        if isinstance(clock, dict):
            val = clock.get('is_open')
            try:
                return bool(val)
            except Exception:
                return False
        return False

    def _get_account(self, account_name: Optional[str] = None) -> AlpacaAccount:
        """Get account for API calls."""
        if account_name:
            if account_name not in self.accounts:
                raise ValueError(f"Account '{account_name}' not found")
            return self.accounts[account_name]
        elif self.current_account:
            return self.current_account
        else:
            raise ValueError("No account specified and no current account set")

    def _api_request(self, method: str, path: str, account: AlpacaAccount = None,
                    json_body: Optional[Dict] = None, params: Optional[Dict] = None,
                    timeout: int = 30) -> Any:
        """
        Make API request to Alpaca.

        Args:
            method: HTTP method
            path: API path
            account: Account to use
            json_body: Request JSON body
            params: Query parameters
            timeout: Request timeout

        Returns:
            API response
        """
        if account is None:
            account = self._get_account()

        url = f"{account.base_url}{path}"
        headers = {
            'APCA-API-KEY-ID': account.api_key,
            'APCA-API-SECRET-KEY': account.api_secret,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        try:
            response = requests.request(
                method, url, headers=headers, json=json_body, params=params, timeout=timeout
            )

            if response.status_code >= 400:
                error_info = response.json() if response.headers.get('content-type', '').startswith('application/json') else {'message': response.text}
                raise RuntimeError(f"Alpaca API error {response.status_code}: {error_info}")

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")

    def _api_data_request(self, method: str, path: str, account: AlpacaAccount = None,
                          json_body: Optional[Dict] = None, params: Optional[Dict] = None,
                          timeout: int = 10) -> Any:
        """Make API request to Alpaca Market Data endpoint.

        Uses https://data.alpaca.markets/v2 regardless of trading base_url.
        """
        if account is None:
            account = self._get_account()

        data_base = "https://data.alpaca.markets"
        # Ensure we always call v2 endpoints
        if not path.startswith("/v2/"):
            if path.startswith("/"):
                path = "/v2" + path
            else:
                path = "/v2/" + path

        url = f"{data_base}{path}"
        headers = {
            'APCA-API-KEY-ID': account.api_key,
            'APCA-API-SECRET-KEY': account.api_secret,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        try:
            response = requests.request(
                method, url, headers=headers, json=json_body, params=params, timeout=timeout
            )

            if response.status_code >= 400:
                error_info = response.json() if response.headers.get('content-type', '').startswith('application/json') else {'message': response.text}
                raise RuntimeError(f"Alpaca DATA API error {response.status_code}: {error_info}")

            if response.status_code == 204:
                return {}

            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Data request failed: {e}")

    def _get_latest_price(self, symbol: str, account: AlpacaAccount = None) -> Optional[float]:
        """Get latest trade/quote-derived price for a symbol from Alpaca Market Data.

        Tries trades/latest, then quotes/latest (mid), then bars/latest (close).
        """
        sym = (symbol or "").upper()
        # 1) Latest trade price
        try:
            resp = self._api_data_request("GET", f"/stocks/{sym}/trades/latest", account=account)
            # v2 format: {"symbol":"AAPL","trade":{"p":123.45,...}}
            trade = resp.get('trade') if isinstance(resp, dict) else None
            if isinstance(trade, dict):
                p = trade.get('p') or trade.get('price')
                if p is not None:
                    return float(p)
            # Some responses may be flattened
            p = resp.get('p') if isinstance(resp, dict) else None
            if p is not None:
                return float(p)
        except Exception:
            pass

        # 2) Latest quote mid
        try:
            resp = self._api_data_request("GET", f"/stocks/{sym}/quotes/latest", account=account)
            quote = resp.get('quote') if isinstance(resp, dict) else None
            if isinstance(quote, dict):
                ap = quote.get('ap') or quote.get('ask_price')
                bp = quote.get('bp') or quote.get('bid_price')
                if ap is not None and bp is not None:
                    apf = float(ap)
                    bpf = float(bp)
                    if apf > 0 and bpf > 0:
                        return (apf + bpf) / 2.0
        except Exception:
            pass

        # 3) Latest bar close
        try:
            resp = self._api_data_request("GET", f"/stocks/{sym}/bars/latest", account=account)
            bar = resp.get('bar') if isinstance(resp, dict) else None
            if isinstance(bar, dict):
                c = bar.get('c') or bar.get('close')
                if c is not None:
                    return float(c)
        except Exception:
            pass

        return None

    def get_available_accounts(self) -> List[str]:
        """Get list of available account names."""
        return list(self.accounts.keys())


# Utility functions for creating account configurations
def create_alpaca_account_from_env(name: str = "default") -> AlpacaAccount:
    """
    Create Alpaca account from environment variables.

    Args:
        name: Account name

    Returns:
        AlpacaAccount instance
    """

    from src.config.settings import get_config
    config = get_config()
    api_key = config.alpaca.api_key
    api_secret = config.alpaca.api_secret
    base_url = config.alpaca.base_url

    if not api_key or not api_secret:
        raise ValueError("Alpaca API credentials not found in environment variables")

    return AlpacaAccount(
        name=name,
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url
    )


def create_multiple_accounts_from_config(config: Dict[str, Dict]) -> List[AlpacaAccount]:
    """
    Create multiple Alpaca accounts from configuration.

    Args:
        config: Configuration dictionary with account settings

    Returns:
        List of AlpacaAccount instances
    """
    accounts = []
    for name, settings in config.items():
        accounts.append(AlpacaAccount(
            name=name,
            api_key=settings['api_key'],
            api_secret=settings['api_secret'],
            base_url=settings.get('base_url', 'https://paper-api.alpaca.markets')
        ))

    return accounts


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create account from environment
    try:
        account = create_alpaca_account_from_env()
        manager = AlpacaManager([account])

        # Get account info
        account_info = manager.get_account_info()
        print(f"Account equity: ${account_info['equity']}")

        # Get positions
        positions = manager.get_positions()
        print(f"Current positions: {len(positions)}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set APCA_API_KEY and APCA_API_SECRET environment variables")
