"""
Market Regime Detection
=======================

Implements two-layer market regime system:
1. Slow Regime Gate (weekly, structural)
2. Fast Risk-Off Overlay (daily, emergency)

Market regime controls risk budget, not asset selection.

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Literal
from dataclasses import dataclass
from enum import Enum

from .utils.robust_stats import robust_zscore
from .config_loader import AdaptiveRotationConfig


# ============================================================================
# Enums and Data Classes
# ============================================================================

class SlowRegimeState(Enum):
    """Slow regime states"""
    RISK_ON = "risk_on"
    NEUTRAL = "neutral"
    RISK_OFF = "risk_off"


@dataclass
class SlowRegimeSignals:
    """Slow regime component signals"""
    trend_deterioration: bool
    drawdown_stress: bool
    volatility_stress: bool
    risk_score: int
    
    # Raw metrics for debugging
    spx_price: float
    spx_ma_26w: float
    spx_drawdown_13w: float
    vix_z_score: float


@dataclass
class SlowRegimeResult:
    """Slow regime detection result"""
    state: SlowRegimeState
    signals: SlowRegimeSignals
    group_cap: float
    cash_floor: float
    is_persistent: bool  # Whether state has persisted for required weeks
    metadata: Dict


@dataclass
class FastRiskOffResult:
    """Fast risk-off overlay result"""
    is_active: bool
    days_remaining: int  # Days until expiry
    trigger_date: Optional[pd.Timestamp]
    
    # Signals
    price_shock: bool
    volatility_shock: bool
    
    # Effective constraints when active
    effective_group_cap: Optional[float]
    effective_cash_floor: Optional[float]
    
    metadata: Dict


@dataclass
class MarketRegimeResult:
    """Complete market regime result"""
    slow_regime: SlowRegimeResult
    fast_risk_off: FastRiskOffResult
    
    # Effective constraints (after fast overlay)
    effective_group_cap: float
    effective_cash_floor: float
    effective_state: str  # "risk_on", "neutral", "risk_off", "fast_risk_off"
    
    as_of_date: pd.Timestamp


# ============================================================================
# Slow Regime Detection
# ============================================================================

def compute_slow_regime_signals(
    spx_prices: pd.Series,
    vix_prices: pd.Series,
    as_of_date: pd.Timestamp,
    trend_ma_weeks: int = 26,
    drawdown_weeks: int = 13,
    drawdown_threshold: float = 0.10,
    vix_lookback_years: int = 3,
    vix_z_threshold: float = 3.0,
) -> SlowRegimeSignals:
    """
    Compute slow regime component signals
    
    Args:
        spx_prices: SPX weekly close prices (DatetimeIndex)
        vix_prices: VIX weekly close prices (DatetimeIndex)
        as_of_date: Current decision date
        trend_ma_weeks: MA period for trend (default 26)
        drawdown_weeks: Drawdown lookback (default 13)
        drawdown_threshold: Drawdown threshold (default 0.10)
        vix_lookback_years: VIX normalization lookback (default 3 years)
        vix_z_threshold: VIX Z-score threshold (default 3.0)
    
    Returns:
        SlowRegimeSignals object
    
    Note:
        All inputs should be point-in-time (≤ as_of_date)
    """
    # Get data as of date
    spx = spx_prices[spx_prices.index <= as_of_date].copy()
    vix = vix_prices[vix_prices.index <= as_of_date].copy()
    
    if len(spx) == 0 or len(vix) == 0:
        raise ValueError(f"No data available as of {as_of_date}")
    
    # Current values
    spx_current = spx.iloc[-1]
    vix_current = vix.iloc[-1]
    
    # 1. Trend Signal: SPX < 26-week MA
    if len(spx) >= trend_ma_weeks:
        spx_ma = spx.rolling(window=trend_ma_weeks, min_periods=trend_ma_weeks).mean().iloc[-1]
        trend_deterioration = spx_current < spx_ma
    else:
        # Not enough data
        spx_ma = np.nan
        trend_deterioration = False
    
    # 2. Drawdown Signal: 13-week drawdown >= 10%
    if len(spx) >= drawdown_weeks:
        recent_spx = spx.iloc[-drawdown_weeks:]
        peak = recent_spx.max()
        drawdown = (spx_current - peak) / peak
        drawdown_stress = drawdown <= -drawdown_threshold
        spx_drawdown_13w = drawdown
    else:
        spx_drawdown_13w = 0.0
        drawdown_stress = False
    
    # 3. Volatility Signal: VIX robust Z-score >= 3.0
    # Use 3-year rolling window for MAD normalization
    vix_lookback_periods = vix_lookback_years * 52  # weeks
    
    if len(vix) >= 52:  # At least 1 year
        # Compute robust Z-score
        vix_z = robust_zscore(
            vix,
            window=min(vix_lookback_periods, len(vix)),
            center_metric="median"
        )
        vix_z_current = vix_z.iloc[-1]
        
        # Handle NaN (insufficient data for Z-score)
        if pd.isna(vix_z_current):
            vix_z_current = 0.0
        
        volatility_stress = vix_z_current >= vix_z_threshold
    else:
        vix_z_current = 0.0
        volatility_stress = False
    
    # Calculate risk score
    risk_score = int(trend_deterioration) + int(drawdown_stress) + int(volatility_stress)
    
    return SlowRegimeSignals(
        trend_deterioration=bool(trend_deterioration),
        drawdown_stress=bool(drawdown_stress),
        volatility_stress=bool(volatility_stress),
        risk_score=int(risk_score),
        spx_price=float(spx_current),
        spx_ma_26w=float(spx_ma),
        spx_drawdown_13w=float(spx_drawdown_13w),
        vix_z_score=float(vix_z_current),
    )


def map_risk_score_to_regime(
    risk_score: int,
    config: AdaptiveRotationConfig,
) -> Tuple[SlowRegimeState, float, float]:
    """
    Map risk score to regime state and constraints
    
    Args:
        risk_score: Risk score (0-3)
        config: Strategy configuration
    
    Returns:
        (state, group_cap, cash_floor)
    """
    mapping = config.market_regime.slow_regime.mapping
    
    if risk_score == 0:
        return (
            SlowRegimeState.RISK_ON,
            mapping.risk_on.group_cap,
            mapping.risk_on.cash_floor,
        )
    elif risk_score == 1:
        return (
            SlowRegimeState.NEUTRAL,
            mapping.neutral.group_cap,
            mapping.neutral.cash_floor,
        )
    else:  # risk_score >= 2
        return (
            SlowRegimeState.RISK_OFF,
            mapping.risk_off.group_cap,
            mapping.risk_off.cash_floor,
        )


def check_regime_persistence(
    current_state: SlowRegimeState,
    previous_states: list[SlowRegimeState],
    persistence_weeks: int = 2,
) -> bool:
    """
    Check if regime has persisted for required weeks
    
    Args:
        current_state: Current regime state
        previous_states: List of previous states (most recent first)
        persistence_weeks: Required persistence (default 2)
    
    Returns:
        True if state has persisted
    
    Examples:
        >>> states = [SlowRegimeState.RISK_OFF, SlowRegimeState.RISK_OFF]
        >>> check_regime_persistence(SlowRegimeState.RISK_OFF, states, 2)
        True
    """
    if len(previous_states) < persistence_weeks - 1:
        # Not enough history
        return False
    
    # Check if all required previous states match current
    for i in range(persistence_weeks - 1):
        if previous_states[i] != current_state:
            return False
    
    return True


def detect_slow_regime(
    spx_prices: pd.Series,
    vix_prices: pd.Series,
    as_of_date: pd.Timestamp,
    config: AdaptiveRotationConfig,
    previous_states: Optional[list[SlowRegimeState]] = None,
) -> SlowRegimeResult:
    """
    Detect slow regime state
    
    Args:
        spx_prices: SPX weekly close prices
        vix_prices: VIX weekly close prices
        as_of_date: Current decision date
        config: Strategy configuration
        previous_states: Previous regime states for persistence check
    
    Returns:
        SlowRegimeResult object
    
    Examples:
        >>> result = detect_slow_regime(spx_prices, vix_prices, date, config)
        >>> print(f"Regime: {result.state.value}, Group Cap: {result.group_cap}")
    """
    # Get configuration
    slow_config = config.market_regime.slow_regime
    
    # Compute signals
    signals = compute_slow_regime_signals(
        spx_prices=spx_prices,
        vix_prices=vix_prices,
        as_of_date=as_of_date,
        trend_ma_weeks=slow_config.trend_ma_weeks,
        drawdown_weeks=slow_config.drawdown_weeks,
        drawdown_threshold=slow_config.drawdown_threshold,
        vix_lookback_years=slow_config.volatility.vix_lookback_years,
        vix_z_threshold=slow_config.volatility.vix_z_threshold,
    )
    
    # Map to regime state
    state, group_cap, cash_floor = map_risk_score_to_regime(signals.risk_score, config)
    
    # Check persistence
    previous_states = previous_states or []
    is_persistent = check_regime_persistence(
        current_state=state,
        previous_states=previous_states,
        persistence_weeks=slow_config.persistence_weeks,
    )
    
    return SlowRegimeResult(
        state=state,
        signals=signals,
        group_cap=group_cap,
        cash_floor=cash_floor,
        is_persistent=is_persistent,
        metadata={
            "as_of_date": as_of_date,
            "persistence_weeks_required": slow_config.persistence_weeks,
            "persistence_weeks_observed": len(previous_states) + 1,
        }
    )


# ============================================================================
# Fast Risk-Off Overlay
# ============================================================================

def detect_price_shock(
    prices: pd.Series,
    as_of_date: pd.Timestamp,
    lookback_days: int = 3,
    threshold: float = -0.03,
) -> Tuple[bool, float]:
    """
    Detect price shock (3-day return <= -3%)
    
    Args:
        prices: Daily close prices
        as_of_date: Current date
        lookback_days: Lookback period (default 3)
        threshold: Return threshold (default -0.03)
    
    Returns:
        (is_shock, return_value)
    """
    # Get data as of date
    prices_filtered = prices[prices.index <= as_of_date]
    
    if len(prices_filtered) < lookback_days + 1:
        return False, 0.0
    
    # Calculate N-day return
    current_price = prices_filtered.iloc[-1]
    past_price = prices_filtered.iloc[-(lookback_days + 1)]
    
    ret = (current_price - past_price) / past_price
    
    is_shock = ret <= threshold
    
    return bool(is_shock), float(ret)


def detect_volatility_shock(
    vix_prices: pd.Series,
    as_of_date: pd.Timestamp,
    vix_z_threshold: float = 3.0,
    delta_vix_z_threshold: float = 3.5,
    lookback_years: int = 3,
) -> Tuple[bool, float, float]:
    """
    Detect volatility shock (VIX_Z >= 3.0 AND ΔVIX_Z >= 3.5)
    
    Args:
        vix_prices: Daily VIX close prices
        as_of_date: Current date
        vix_z_threshold: VIX Z-score threshold (default 3.0)
        delta_vix_z_threshold: VIX change Z-score threshold (default 3.5)
        lookback_years: Lookback for normalization (default 3 years)
    
    Returns:
        (is_shock, vix_z, delta_vix_z)
    """
    # Get data as of date
    vix = vix_prices[vix_prices.index <= as_of_date]
    
    if len(vix) < 252:  # At least 1 year of daily data
        return False, 0.0, 0.0
    
    # Compute VIX Z-score
    lookback_periods = lookback_years * 252  # trading days
    vix_z = robust_zscore(
        vix,
        window=min(lookback_periods, len(vix)),
        center_metric="median"
    )
    
    if pd.isna(vix_z.iloc[-1]):
        return False, 0.0, 0.0
    
    vix_z_current = vix_z.iloc[-1]
    
    # Compute ΔVIX Z-score
    if len(vix) >= 2:
        vix_delta = vix.diff()
        delta_vix_z = robust_zscore(
            vix_delta,
            window=min(lookback_periods, len(vix_delta)),
            center_metric="median"
        )
        
        if pd.isna(delta_vix_z.iloc[-1]):
            delta_vix_z_current = 0.0
        else:
            delta_vix_z_current = delta_vix_z.iloc[-1]
    else:
        delta_vix_z_current = 0.0
    
    # Check thresholds
    is_shock = (vix_z_current >= vix_z_threshold) and (delta_vix_z_current >= delta_vix_z_threshold)
    
    return is_shock, vix_z_current, delta_vix_z_current


def check_fast_risk_off_trigger(
    spx_daily: pd.Series,
    qqq_daily: pd.Series,
    vix_daily: pd.Series,
    as_of_date: pd.Timestamp,
    config: AdaptiveRotationConfig,
) -> Tuple[bool, Dict]:
    """
    Check if fast risk-off should be triggered
    
    Args:
        spx_daily: SPX daily close prices
        qqq_daily: QQQ daily close prices
        vix_daily: VIX daily close prices
        as_of_date: Current date
        config: Strategy configuration
    
    Returns:
        (should_trigger, signal_details)
    """
    fro_config = config.fast_risk_off
    
    # Check price shock for SPX
    spx_shock, spx_ret = detect_price_shock(
        spx_daily,
        as_of_date,
        lookback_days=fro_config.price_shock.lookback_days,
        threshold=fro_config.price_shock.drawdown_threshold,
    )
    
    # Check price shock for QQQ
    qqq_shock, qqq_ret = detect_price_shock(
        qqq_daily,
        as_of_date,
        lookback_days=fro_config.price_shock.lookback_days,
        threshold=fro_config.price_shock.drawdown_threshold,
    )
    
    # Price shock = SPX OR QQQ
    price_shock = spx_shock or qqq_shock
    
    # Check volatility shock
    vol_shock, vix_z, delta_vix_z = detect_volatility_shock(
        vix_daily,
        as_of_date,
        vix_z_threshold=fro_config.volatility_shock.vix_z_threshold,
        delta_vix_z_threshold=fro_config.volatility_shock.delta_vix_z_threshold,
    )
    
    # Trigger = Price Shock OR Volatility Shock
    # Either condition alone is sufficient for early protection
    should_trigger = price_shock or vol_shock
    
    signal_details = {
        "price_shock": price_shock,
        "spx_shock": spx_shock,
        "qqq_shock": qqq_shock,
        "spx_return": spx_ret,
        "qqq_return": qqq_ret,
        "volatility_shock": vol_shock,
        "vix_z": vix_z,
        "delta_vix_z": delta_vix_z,
    }
    
    return should_trigger, signal_details


def check_fast_risk_off_exit(
    spx_daily: pd.Series,
    as_of_date: pd.Timestamp,
    trigger_date: pd.Timestamp,
    duration_days: int = 5,
    exit_return_threshold: float = 0.01,
) -> Tuple[bool, str]:
    """
    Check if fast risk-off should exit
    
    Exit conditions:
    1. Duration elapsed (5 trading days)
    2. SPX 3-day return > +1%
    
    Args:
        spx_daily: SPX daily close prices
        as_of_date: Current date
        trigger_date: Date when fast risk-off was triggered
        duration_days: Maximum duration (default 5)
        exit_return_threshold: Recovery threshold (default 0.01)
    
    Returns:
        (should_exit, reason)
    """
    # Get trading days since trigger
    spx = spx_daily[(spx_daily.index > trigger_date) & (spx_daily.index <= as_of_date)]
    days_elapsed = len(spx)
    
    # Check duration
    if days_elapsed >= duration_days:
        return True, "duration_expired"
    
    # Check recovery (3-day return > +1%)
    if len(spx) >= 3:
        current_price = spx.iloc[-1]
        past_price = spx.iloc[-4] if len(spx) >= 4 else spx.iloc[0]
        recovery_return = (current_price - past_price) / past_price
        
        if recovery_return > exit_return_threshold:
            return True, "recovery"
    
    return False, "active"


def update_fast_risk_off_state(
    spx_daily: pd.Series,
    qqq_daily: pd.Series,
    vix_daily: pd.Series,
    as_of_date: pd.Timestamp,
    config: AdaptiveRotationConfig,
    current_fast_state: Optional[FastRiskOffResult] = None,
) -> FastRiskOffResult:
    """
    Update fast risk-off overlay state
    
    Args:
        spx_daily: SPX daily prices
        qqq_daily: QQQ daily prices
        vix_daily: VIX daily prices
        as_of_date: Current date
        config: Strategy configuration
        current_fast_state: Current fast risk-off state (if any)
    
    Returns:
        FastRiskOffResult object
    """
    fro_config = config.fast_risk_off
    
    # If currently active, check exit conditions
    if current_fast_state and current_fast_state.is_active:
        should_exit, reason = check_fast_risk_off_exit(
            spx_daily,
            as_of_date,
            current_fast_state.trigger_date,
            duration_days=fro_config.behavior.duration_days,
        )
        
        if should_exit:
            # Exit fast risk-off
            return FastRiskOffResult(
                is_active=False,
                days_remaining=0,
                trigger_date=None,
                price_shock=False,
                volatility_shock=False,
                effective_group_cap=None,
                effective_cash_floor=None,
                metadata={"exit_reason": reason, "previous_trigger": current_fast_state.trigger_date},
            )
        else:
            # Still active
            days_elapsed = len(spx_daily[
                (spx_daily.index > current_fast_state.trigger_date) &
                (spx_daily.index <= as_of_date)
            ])
            days_remaining = fro_config.behavior.duration_days - days_elapsed
            
            return FastRiskOffResult(
                is_active=True,
                days_remaining=max(0, days_remaining),
                trigger_date=current_fast_state.trigger_date,
                price_shock=current_fast_state.price_shock,
                volatility_shock=current_fast_state.volatility_shock,
                effective_group_cap=fro_config.behavior.group_cap,
                effective_cash_floor=fro_config.behavior.cash_floor,
                metadata={"status": "active", "days_elapsed": days_elapsed},
            )
    
    # Not currently active, check trigger
    should_trigger, signal_details = check_fast_risk_off_trigger(
        spx_daily, qqq_daily, vix_daily, as_of_date, config
    )
    
    if should_trigger:
        # Trigger fast risk-off
        return FastRiskOffResult(
            is_active=True,
            days_remaining=fro_config.behavior.duration_days,
            trigger_date=as_of_date,
            price_shock=signal_details["price_shock"],
            volatility_shock=signal_details["volatility_shock"],
            effective_group_cap=fro_config.behavior.group_cap,
            effective_cash_floor=fro_config.behavior.cash_floor,
            metadata={"status": "triggered", "signals": signal_details},
        )
    else:
        # Not active, not triggered
        return FastRiskOffResult(
            is_active=False,
            days_remaining=0,
            trigger_date=None,
            price_shock=signal_details["price_shock"],
            volatility_shock=signal_details["volatility_shock"],
            effective_group_cap=None,
            effective_cash_floor=None,
            metadata={"status": "inactive", "signals": signal_details},
        )


# ============================================================================
# Complete Market Regime Detection
# ============================================================================

def detect_market_regime(
    spx_weekly: pd.Series,
    vix_weekly: pd.Series,
    spx_daily: pd.Series,
    qqq_daily: pd.Series,
    vix_daily: pd.Series,
    as_of_date: pd.Timestamp,
    config: AdaptiveRotationConfig,
    previous_slow_states: Optional[list[SlowRegimeState]] = None,
    current_fast_state: Optional[FastRiskOffResult] = None,
) -> MarketRegimeResult:
    """
    Detect complete market regime (slow + fast overlay)
    
    Args:
        spx_weekly: SPX weekly close prices
        vix_weekly: VIX weekly close prices
        spx_daily: SPX daily close prices
        qqq_daily: QQQ daily close prices
        vix_daily: VIX daily close prices
        as_of_date: Current decision date
        config: Strategy configuration
        previous_slow_states: Previous slow regime states
        current_fast_state: Current fast risk-off state
    
    Returns:
        MarketRegimeResult object
    
    Examples:
        >>> regime = detect_market_regime(
        ...     spx_weekly, vix_weekly,
        ...     spx_daily, qqq_daily, vix_daily,
        ...     pd.Timestamp("2024-02-01"),
        ...     config
        ... )
        >>> print(f"Effective state: {regime.effective_state}")
        >>> print(f"Group cap: {regime.effective_group_cap}")
    """
    # Detect slow regime
    slow_result = detect_slow_regime(
        spx_weekly, vix_weekly, as_of_date, config, previous_slow_states
    )
    
    # Update fast risk-off overlay
    fast_result = update_fast_risk_off_state(
        spx_daily, qqq_daily, vix_daily, as_of_date, config, current_fast_state
    )
    
    # Compute effective constraints
    if fast_result.is_active:
        # Fast overlay tightens constraints
        effective_group_cap = min(slow_result.group_cap, fast_result.effective_group_cap)
        effective_cash_floor = max(slow_result.cash_floor, fast_result.effective_cash_floor)
        effective_state = "fast_risk_off"
    else:
        # Use slow regime constraints
        effective_group_cap = slow_result.group_cap
        effective_cash_floor = slow_result.cash_floor
        effective_state = slow_result.state.value
    
    return MarketRegimeResult(
        slow_regime=slow_result,
        fast_risk_off=fast_result,
        effective_group_cap=effective_group_cap,
        effective_cash_floor=effective_cash_floor,
        effective_state=effective_state,
        as_of_date=as_of_date,
    )


if __name__ == "__main__":
    """Quick test of market regime detection"""
    
    print("Testing Market Regime Detection")
    print("=" * 60)
    
    # Create sample data
    dates_weekly = pd.date_range("2020-01-01", "2024-12-31", freq="W-FRI")
    dates_daily = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    
    # Simulate SPX (uptrend with volatility)
    spx_weekly = pd.Series(
        2000 + np.random.randn(len(dates_weekly)).cumsum() * 50,
        index=dates_weekly
    )
    spx_daily = pd.Series(
        2000 + np.random.randn(len(dates_daily)).cumsum() * 10,
        index=dates_daily
    )
    
    # Simulate VIX (mean-reverting)
    vix_weekly = pd.Series(
        20 + np.random.randn(len(dates_weekly)) * 5,
        index=dates_weekly
    ).clip(lower=10)
    
    vix_daily = pd.Series(
        20 + np.random.randn(len(dates_daily)) * 5,
        index=dates_daily
    ).clip(lower=10)
    
    # QQQ similar to SPX
    qqq_daily = spx_daily * 0.8
    
    # Load config
    from config_loader import load_config
    config = load_config("src/strategies/AdaptiveRotationConf_v1.2.1.yaml")
    
    # Test detection
    test_date = pd.Timestamp("2024-06-30")
    
    regime = detect_market_regime(
        spx_weekly, vix_weekly,
        spx_daily, qqq_daily, vix_daily,
        test_date,
        config
    )
    
    print(f"\n[Test Date: {test_date.date()}]")
    print(f"\nSlow Regime:")
    print(f"  State: {regime.slow_regime.state.value}")
    print(f"  Risk Score: {regime.slow_regime.signals.risk_score}")
    print(f"  Trend Deterioration: {regime.slow_regime.signals.trend_deterioration}")
    print(f"  Drawdown Stress: {regime.slow_regime.signals.drawdown_stress}")
    print(f"  Volatility Stress: {regime.slow_regime.signals.volatility_stress}")
    print(f"  Group Cap: {regime.slow_regime.group_cap}")
    print(f"  Cash Floor: {regime.slow_regime.cash_floor}")
    
    print(f"\nFast Risk-Off:")
    print(f"  Active: {regime.fast_risk_off.is_active}")
    print(f"  Price Shock: {regime.fast_risk_off.price_shock}")
    print(f"  Volatility Shock: {regime.fast_risk_off.volatility_shock}")
    
    print(f"\nEffective Constraints:")
    print(f"  State: {regime.effective_state}")
    print(f"  Group Cap: {regime.effective_group_cap}")
    print(f"  Cash Floor: {regime.effective_cash_floor}")
    
    print(f"\n{'='*60}")
    print("[PASS] Market regime detection test complete!")
