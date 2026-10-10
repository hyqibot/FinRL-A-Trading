"""
Risk Manager
============

Manages stop-loss rules for positions in the portfolio.

Key Features:
- Absolute stop-loss (e.g., -5% from entry)
- Trailing stop-loss (e.g., -10% from peak)
- Cooldown period after stop-loss
- Position tracking (entry price, peak price)

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta

from .config_loader import AdaptiveRotationConfig


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StopLossSignal:
    """Stop-loss trigger signal"""
    symbol: str
    trigger_type: str          # "absolute" | "trailing"
    trigger_date: pd.Timestamp
    
    # Price levels
    entry_price: float
    current_price: float
    peak_price: float
    
    # Loss metrics
    loss_from_entry_pct: float     # (current - entry) / entry
    loss_from_peak_pct: float      # (current - peak) / peak
    
    # Threshold that was breached
    threshold_breached: float
    
    def __str__(self) -> str:
        return (
            f"StopLoss[{self.symbol}]: {self.trigger_type} "
            f"@ {self.current_price:.2f} "
            f"({self.loss_from_entry_pct:.2%} from entry, "
            f"{self.loss_from_peak_pct:.2%} from peak)"
        )


@dataclass
class PositionState:
    """State of a position for stop-loss tracking"""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    peak_price: float
    peak_date: pd.Timestamp
    
    # Cooldown tracking
    in_cooldown: bool = False
    cooldown_until: Optional[pd.Timestamp] = None
    
    def update_peak(self, current_price: float, current_date: pd.Timestamp) -> bool:
        """
        Update peak price if current price is higher
        
        Returns:
            True if peak was updated
        """
        if current_price > self.peak_price:
            self.peak_price = current_price
            self.peak_date = current_date
            return True
        return False
    
    def is_in_cooldown(self, as_of_date: pd.Timestamp) -> bool:
        """Check if position is in cooldown period"""
        if not self.in_cooldown:
            return False
        
        if self.cooldown_until is None:
            return False
        
        return as_of_date < self.cooldown_until


@dataclass
class RiskCheckResult:
    """Result of risk check"""
    as_of_date: pd.Timestamp
    triggered_stops: List[StopLossSignal]
    updated_positions: Dict[str, PositionState]
    cooldowns_active: Dict[str, pd.Timestamp]  # symbol → cooldown_until
    
    def has_stops(self) -> bool:
        """Check if any stops were triggered"""
        return len(self.triggered_stops) > 0
    
    def get_stopped_symbols(self) -> List[str]:
        """Get list of symbols that triggered stops"""
        return [sig.symbol for sig in self.triggered_stops]


# ============================================================================
# Stop-Loss Logic
# ============================================================================

def check_absolute_stop(
    entry_price: float,
    current_price: float,
    threshold: float,
) -> Tuple[bool, float]:
    """
    Check if absolute stop-loss is triggered
    
    Args:
        entry_price: Entry price of position
        current_price: Current market price
        threshold: Stop-loss threshold (e.g., -0.05 for -5%)
    
    Returns:
        Tuple of (is_triggered, loss_pct)
    
    Examples:
        >>> triggered, loss = check_absolute_stop(100, 94, -0.05)
        >>> print(f"Triggered: {triggered}, Loss: {loss:.2%}")
    """
    loss_pct = (current_price - entry_price) / entry_price
    
    is_triggered = loss_pct <= threshold
    
    return is_triggered, loss_pct


def check_trailing_stop(
    peak_price: float,
    current_price: float,
    threshold: float,
) -> Tuple[bool, float]:
    """
    Check if trailing stop-loss is triggered
    
    Args:
        peak_price: Peak price since entry
        current_price: Current market price
        threshold: Trailing stop threshold (e.g., -0.10 for -10%)
    
    Returns:
        Tuple of (is_triggered, loss_from_peak_pct)
    
    Examples:
        >>> # Peak was 110, current is 98, threshold -10%
        >>> triggered, loss = check_trailing_stop(110, 98, -0.10)
        >>> print(f"Loss from peak: {loss:.2%}")
    """
    loss_from_peak_pct = (current_price - peak_price) / peak_price
    
    is_triggered = loss_from_peak_pct <= threshold
    
    return is_triggered, loss_from_peak_pct


def check_position_stops(
    symbol: str,
    position: PositionState,
    current_price: float,
    current_date: pd.Timestamp,
    absolute_threshold: float,
    trailing_threshold: float,
) -> Optional[StopLossSignal]:
    """
    Check both stop-loss rules for a single position
    
    Args:
        symbol: Asset symbol
        position: Position state
        current_price: Current price
        current_date: Current date
        absolute_threshold: Absolute stop threshold
        trailing_threshold: Trailing stop threshold
    
    Returns:
        StopLossSignal if triggered, None otherwise
    
    Note:
        If both stops trigger, absolute takes priority
    """
    # Check absolute stop
    abs_triggered, loss_from_entry = check_absolute_stop(
        position.entry_price,
        current_price,
        absolute_threshold
    )
    
    if abs_triggered:
        return StopLossSignal(
            symbol=symbol,
            trigger_type="absolute",
            trigger_date=current_date,
            entry_price=position.entry_price,
            current_price=current_price,
            peak_price=position.peak_price,
            loss_from_entry_pct=loss_from_entry,
            loss_from_peak_pct=(current_price - position.peak_price) / position.peak_price,
            threshold_breached=absolute_threshold,
        )
    
    # Check trailing stop
    trail_triggered, loss_from_peak = check_trailing_stop(
        position.peak_price,
        current_price,
        trailing_threshold
    )
    
    if trail_triggered:
        return StopLossSignal(
            symbol=symbol,
            trigger_type="trailing",
            trigger_date=current_date,
            entry_price=position.entry_price,
            current_price=current_price,
            peak_price=position.peak_price,
            loss_from_entry_pct=(current_price - position.entry_price) / position.entry_price,
            loss_from_peak_pct=loss_from_peak,
            threshold_breached=trailing_threshold,
        )
    
    return None


# ============================================================================
# Peak Price Management
# ============================================================================

def update_position_peaks(
    positions: Dict[str, PositionState],
    current_prices: Dict[str, float],
    current_date: pd.Timestamp,
) -> Dict[str, PositionState]:
    """
    Update peak prices for all positions
    
    Args:
        positions: Current positions
        current_prices: Current market prices
        current_date: Current date
    
    Returns:
        Updated positions dictionary
    """
    updated_positions = {}
    
    for symbol, position in positions.items():
        # Create copy
        updated_position = PositionState(
            symbol=position.symbol,
            entry_date=position.entry_date,
            entry_price=position.entry_price,
            peak_price=position.peak_price,
            peak_date=position.peak_date,
            in_cooldown=position.in_cooldown,
            cooldown_until=position.cooldown_until,
        )
        
        # Update peak if price available
        if symbol in current_prices:
            updated_position.update_peak(current_prices[symbol], current_date)
        
        updated_positions[symbol] = updated_position
    
    return updated_positions


# ============================================================================
# Cooldown Management
# ============================================================================

def activate_cooldown(
    symbol: str,
    trigger_date: pd.Timestamp,
    cooldown_weeks: int,
) -> pd.Timestamp:
    """
    Activate cooldown period for a symbol
    
    Args:
        symbol: Asset symbol
        trigger_date: Date when stop was triggered
        cooldown_weeks: Cooldown duration in weeks
    
    Returns:
        Cooldown expiration date
    """
    cooldown_until = trigger_date + timedelta(weeks=cooldown_weeks)
    return cooldown_until


def is_symbol_in_cooldown(
    symbol: str,
    cooldowns: Dict[str, pd.Timestamp],
    as_of_date: pd.Timestamp,
) -> bool:
    """
    Check if symbol is in cooldown period
    
    Args:
        symbol: Asset symbol
        cooldowns: Dict of symbol → cooldown_until
        as_of_date: Current date
    
    Returns:
        True if in cooldown
    """
    if symbol not in cooldowns:
        return False
    
    cooldown_until = cooldowns[symbol]
    return as_of_date < cooldown_until


# ============================================================================
# High-Level API
# ============================================================================

class RiskManager:
    """
    Manages stop-loss rules and position risk
    
    Examples:
        >>> risk_mgr = RiskManager(
        ...     absolute_threshold=-0.05,
        ...     trailing_threshold=-0.10,
        ...     cooldown_weeks=2
        ... )
        >>> 
        >>> # Check stops for current positions
        >>> result = risk_mgr.check_stops(
        ...     positions=positions_dict,
        ...     current_prices=prices_dict,
        ...     as_of_date=pd.Timestamp("2024-02-01")
        ... )
        >>> 
        >>> if result.has_stops():
        ...     print(f"Stops triggered: {result.get_stopped_symbols()}")
    """
    
    def __init__(
        self,
        absolute_threshold: float = -0.05,
        trailing_threshold: float = -0.10,
        cooldown_weeks: int = 2,
    ):
        """
        Initialize risk manager
        
        Args:
            absolute_threshold: Absolute stop-loss threshold (e.g., -0.05 for -5%)
            trailing_threshold: Trailing stop threshold (e.g., -0.10 for -10%)
            cooldown_weeks: Cooldown period in weeks after stop
        """
        self.absolute_threshold = absolute_threshold
        self.trailing_threshold = trailing_threshold
        self.cooldown_weeks = cooldown_weeks
    
    def check_stops(
        self,
        positions: Dict[str, PositionState],
        current_prices: Dict[str, float],
        as_of_date: pd.Timestamp,
    ) -> RiskCheckResult:
        """
        Check all positions for stop-loss triggers
        
        Args:
            positions: Current positions
            current_prices: Current market prices
            as_of_date: Current date
        
        Returns:
            RiskCheckResult object
        """
        triggered_stops = []
        cooldowns = {}
        
        # Update peaks first
        updated_positions = update_position_peaks(positions, current_prices, as_of_date)
        
        # Check each position
        for symbol, position in updated_positions.items():
            # Skip if already in cooldown
            if position.is_in_cooldown(as_of_date):
                cooldowns[symbol] = position.cooldown_until
                continue
            
            # Get current price
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # Check stops
            stop_signal = check_position_stops(
                symbol,
                position,
                current_price,
                as_of_date,
                self.absolute_threshold,
                self.trailing_threshold,
            )
            
            if stop_signal is not None:
                triggered_stops.append(stop_signal)
                
                # Activate cooldown
                cooldown_until = activate_cooldown(
                    symbol,
                    as_of_date,
                    self.cooldown_weeks
                )
                cooldowns[symbol] = cooldown_until
                
                # Update position cooldown state
                updated_positions[symbol].in_cooldown = True
                updated_positions[symbol].cooldown_until = cooldown_until
        
        return RiskCheckResult(
            as_of_date=as_of_date,
            triggered_stops=triggered_stops,
            updated_positions=updated_positions,
            cooldowns_active=cooldowns,
        )
    
    def create_position(
        self,
        symbol: str,
        entry_date: pd.Timestamp,
        entry_price: float,
    ) -> PositionState:
        """
        Create a new position state
        
        Args:
            symbol: Asset symbol
            entry_date: Entry date
            entry_price: Entry price
        
        Returns:
            PositionState object
        """
        return PositionState(
            symbol=symbol,
            entry_date=entry_date,
            entry_price=entry_price,
            peak_price=entry_price,  # Initially peak = entry
            peak_date=entry_date,
        )
    
    @classmethod
    def from_config(cls, config: AdaptiveRotationConfig) -> "RiskManager":
        """
        Create risk manager from configuration
        
        Args:
            config: Strategy configuration
        
        Returns:
            RiskManager instance
        """
        # Convert cooldown days to weeks (round up)
        cooldown_days = config.cooldown.after_stop_days
        cooldown_weeks = (cooldown_days + 6) // 7  # Round up to nearest week
        
        return cls(
            absolute_threshold=config.stop_loss.absolute.threshold,
            trailing_threshold=config.stop_loss.trailing.threshold,
            cooldown_weeks=cooldown_weeks,
        )


if __name__ == "__main__":
    """Quick test of risk management"""
    
    print("Testing Risk Management")
    print("=" * 60)
    
    # Create risk manager
    risk_mgr = RiskManager(
        absolute_threshold=-0.05,
        trailing_threshold=-0.10,
        cooldown_weeks=2
    )
    
    # Create sample positions
    test_date = pd.Timestamp("2024-02-01")
    
    positions = {
        "AAPL": risk_mgr.create_position("AAPL", test_date, 100.0),
        "MSFT": risk_mgr.create_position("MSFT", test_date, 200.0),
    }
    
    # Update AAPL peak to 110
    positions["AAPL"].peak_price = 110.0
    
    # Test 1: No stops triggered
    print("\n[Test 1: Normal prices]")
    current_prices = {"AAPL": 105.0, "MSFT": 205.0}
    
    result = risk_mgr.check_stops(positions, current_prices, test_date)
    
    print(f"  Stops triggered: {len(result.triggered_stops)}")
    assert len(result.triggered_stops) == 0, "Should have no stops"
    print("  ✓ PASS")
    
    # Test 2: Absolute stop triggered
    print("\n[Test 2: Absolute stop]")
    current_prices = {"AAPL": 94.0, "MSFT": 205.0}  # AAPL down 6%
    
    result = risk_mgr.check_stops(positions, current_prices, test_date)
    
    print(f"  Stops triggered: {len(result.triggered_stops)}")
    if len(result.triggered_stops) > 0:
        for stop in result.triggered_stops:
            print(f"    {stop}")
    
    assert len(result.triggered_stops) == 1, "Should have 1 stop"
    assert result.triggered_stops[0].trigger_type == "absolute"
    print("  ✓ PASS")
    
    # Test 3: Trailing stop triggered
    print("\n[Test 3: Trailing stop]")
    # Reset positions, AAPL had peak at 110, now at 98 (-10.9% from peak)
    positions["AAPL"] = risk_mgr.create_position("AAPL", test_date, 100.0)
    positions["AAPL"].peak_price = 110.0
    
    current_prices = {"AAPL": 98.0, "MSFT": 205.0}
    
    result = risk_mgr.check_stops(positions, current_prices, test_date)
    
    print(f"  Stops triggered: {len(result.triggered_stops)}")
    if len(result.triggered_stops) > 0:
        for stop in result.triggered_stops:
            print(f"    {stop}")
    
    assert len(result.triggered_stops) == 1, "Should have 1 stop"
    assert result.triggered_stops[0].trigger_type == "trailing"
    print("  ✓ PASS")
    
    print(f"\n{'='*60}")
    print("[PASS] Risk management test complete!")
