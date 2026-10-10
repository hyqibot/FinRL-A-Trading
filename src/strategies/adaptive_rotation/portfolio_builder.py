"""
Portfolio Builder
=================

Constructs target portfolio from all strategy signals.

Key Features:
- Integrates market regime, group strength, intra-group ranking, and exceptions
- Equal-weight allocation within groups
- Exception asset priority and weight multiplier
- Market regime risk budget adjustment
- Cash as residual

Author: Adaptive Rotation Strategy Team
Version: 1.2.1
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config_loader import AdaptiveRotationConfig
from .market_regime import MarketRegimeResult
from .group_strength import GroupStrengthResult
from .intra_group_ranking import GroupRankingResult
from .exception_framework import ExceptionDetectionResult


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PortfolioWeights:
    """Portfolio construction output"""
    as_of_date: pd.Timestamp
    weights: Dict[str, float]  # symbol → weight (0 to 1)
    
    # Metadata for audit
    active_groups: List[str]
    exception_symbols: List[str]
    cash_weight: float
    
    # Regime impact
    regime_state: str
    risk_budget: float  # Total risk budget (0 to 1)
    
    # Allocation details
    group_budgets: Dict[str, float]  # group → budget
    asset_allocations: Dict[str, Dict[str, float]]  # group → {symbol: weight}
    
    def validate(self) -> bool:
        """Validate that weights sum to <= 1.0"""
        total = sum(self.weights.values())
        return total <= 1.0 + 1e-6  # Small tolerance for floating point
    
    def get_weight(self, symbol: str) -> float:
        """Get weight for a specific symbol"""
        return self.weights.get(symbol, 0.0)
    
    def get_invested_weight(self) -> float:
        """Get total invested weight (excluding cash)"""
        return sum(self.weights.values())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'date': self.as_of_date.strftime('%Y-%m-%d'),
            'weights': self.weights,
            'cash_weight': self.cash_weight,
            'regime_state': self.regime_state,
            'risk_budget': self.risk_budget,
            'active_groups': self.active_groups,
            'exception_symbols': self.exception_symbols,
        }


@dataclass
class PortfolioBuildResult:
    """Complete portfolio build result with diagnostics"""
    portfolio: PortfolioWeights
    
    # Input summaries
    regime_result: Optional[MarketRegimeResult] = None
    group_strength: Optional[GroupStrengthResult] = None
    group_rankings: Optional[Dict[str, GroupRankingResult]] = None
    exceptions: Optional[ExceptionDetectionResult] = None
    
    # Build diagnostics
    constraints_applied: Dict[str, any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Risk Budget Calculation
# ============================================================================

def calculate_risk_budget(
    regime_result: MarketRegimeResult,
    risk_on_budget: float = 1.0,
    risk_off_budget: float = 0.5,
) -> Tuple[float, str]:
    """
    Calculate risk budget based on market regime
    
    Args:
        regime_result: Market regime detection result
        risk_on_budget: Budget in risk-on regime
        risk_off_budget: Budget in risk-off regime
    
    Returns:
        Tuple of (risk_budget, regime_state)
    
    Examples:
        >>> budget, state = calculate_risk_budget(regime_result)
        >>> print(f"Risk budget: {budget:.1%} ({state})")
    """
    # Determine effective regime state
    if regime_result.fast_risk_off.is_active:
        return risk_off_budget, "fast_risk_off"
    
    # Use slow regime
    slow_state = regime_result.slow_regime.state
    
    if slow_state.value == "risk_off":
        return risk_off_budget, "risk_off"
    elif slow_state.value == "neutral":
        # Interpolate between risk_on and risk_off
        neutral_budget = (risk_on_budget + risk_off_budget) / 2
        return neutral_budget, "neutral"
    else:  # risk_on
        return risk_on_budget, "risk_on"


# ============================================================================
# Group Budget Allocation
# ============================================================================

def allocate_group_budgets(
    active_groups: List[str],
    risk_budget: float,
    equal_weight: bool = True,
) -> Dict[str, float]:
    """
    Allocate budget to active groups
    
    Args:
        active_groups: List of active group names
        risk_budget: Total available budget
        equal_weight: Whether to use equal weighting
    
    Returns:
        Dict mapping group → budget
    
    Examples:
        >>> budgets = allocate_group_budgets(["group_a", "group_b"], 1.0)
        >>> # Each group gets 50%
    """
    if not active_groups:
        return {}
    
    if equal_weight:
        # Equal allocation
        budget_per_group = risk_budget / len(active_groups)
        return {group: budget_per_group for group in active_groups}
    else:
        # TODO: Could implement risk-parity or other weighting schemes
        raise NotImplementedError("Only equal_weight is currently supported")


# ============================================================================
# Asset Weight Calculation
# ============================================================================

def calculate_asset_weights_in_group(
    group_name: str,
    group_budget: float,
    top_assets: List[str],
    equal_weight: bool = True,
) -> Dict[str, float]:
    """
    Calculate asset weights within a group
    
    Args:
        group_name: Group name
        group_budget: Budget allocated to this group
        top_assets: List of selected assets
        equal_weight: Whether to use equal weighting
    
    Returns:
        Dict mapping symbol → weight
    """
    if not top_assets:
        return {}
    
    if equal_weight:
        weight_per_asset = group_budget / len(top_assets)
        return {symbol: weight_per_asset for symbol in top_assets}
    else:
        raise NotImplementedError("Only equal_weight is currently supported")


def apply_exception_multiplier(
    weights: Dict[str, float],
    exception_symbols: List[str],
    multiplier: float,
) -> Dict[str, float]:
    """
    Apply weight multiplier to exception assets
    
    Args:
        weights: Current asset weights
        exception_symbols: List of exception symbols
        multiplier: Weight multiplier (e.g., 1.5)
    
    Returns:
        Updated weights (not normalized)
    
    Note:
        Weights are not normalized here. Caller must normalize.
    """
    updated_weights = weights.copy()
    
    for symbol in exception_symbols:
        if symbol in updated_weights:
            updated_weights[symbol] *= multiplier
    
    return updated_weights


# ============================================================================
# Portfolio Construction
# ============================================================================

def normalize_weights(
    weights: Dict[str, float],
    max_total: float = 1.0,
) -> Dict[str, float]:
    """
    Normalize weights to not exceed max_total
    
    Args:
        weights: Asset weights
        max_total: Maximum total weight
    
    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    
    if total == 0:
        return weights
    
    if total > max_total:
        # Scale down proportionally
        scale_factor = max_total / total
        return {sym: w * scale_factor for sym, w in weights.items()}
    
    return weights


def build_fallback_portfolio(
    fallback_symbols: List[str],
    risk_budget: float,
    regime_state: str,
    as_of_date: pd.Timestamp,
) -> PortfolioWeights:
    """
    Build fallback portfolio when no groups qualify
    
    Fallback strategy allocates equal weight to benchmark ETFs (SPY, QQQ)
    to maintain market exposure and avoid missing rallies.
    
    Args:
        fallback_symbols: List of fallback symbols (e.g., ["SPY", "QQQ"])
        risk_budget: Total risk budget from regime
        regime_state: Current regime state
        as_of_date: Current date
    
    Returns:
        PortfolioWeights with fallback allocation
    
    Examples:
        >>> portfolio = build_fallback_portfolio(
        ...     fallback_symbols=["SPY", "QQQ"],
        ...     risk_budget=1.0,
        ...     regime_state="risk_on",
        ...     as_of_date=pd.Timestamp("2024-01-01")
        ... )
        >>> print(portfolio.weights)  # {"SPY": 0.5, "QQQ": 0.5}
    """
    # Equal weight allocation
    n_symbols = len(fallback_symbols)
    if n_symbols == 0:
        # No fallback symbols, return empty portfolio
        return PortfolioWeights(
            as_of_date=as_of_date,
            weights={},
            active_groups=["FALLBACK"],
            exception_symbols=[],
            cash_weight=1.0,
            regime_state=regime_state,
            risk_budget=risk_budget,
            group_budgets={},
            asset_allocations={}
        )
    
    # Calculate equal weights
    weight_per_symbol = risk_budget / n_symbols
    
    # Build weights dict
    weights = {symbol: weight_per_symbol for symbol in fallback_symbols}
    
    # Calculate cash
    invested = sum(weights.values())
    cash_weight = 1.0 - invested
    
    return PortfolioWeights(
        as_of_date=as_of_date,
        weights=weights,
        active_groups=["FALLBACK"],  # Special marker
        exception_symbols=[],
        cash_weight=cash_weight,
        regime_state=regime_state,
        risk_budget=risk_budget,
        group_budgets={"FALLBACK": risk_budget},
        asset_allocations={"FALLBACK": weights}
    )


def build_portfolio_weights(
    active_groups: List[str],
    group_rankings: Dict[str, GroupRankingResult],
    exception_symbols: List[str],
    risk_budget: float,
    regime_state: str,
    as_of_date: pd.Timestamp,
    max_assets_per_group: int = 2,
    exception_weight_multiplier: float = 1.5,
) -> PortfolioWeights:
    """
    Build portfolio weights from all inputs
    
    Args:
        active_groups: List of active group names
        group_rankings: Dict of group → ranking results
        exception_symbols: List of exception asset symbols
        risk_budget: Total risk budget
        regime_state: Market regime state
        as_of_date: Current date
        max_assets_per_group: Max assets per group
        exception_weight_multiplier: Exception weight multiplier
    
    Returns:
        PortfolioWeights object
    """
    # Step 1: Allocate budgets to groups
    group_budgets = allocate_group_budgets(active_groups, risk_budget)
    
    # Step 2: Select assets and calculate weights
    all_weights = {}
    asset_allocations = {}
    
    for group_name in active_groups:
        if group_name not in group_rankings:
            continue
        
        ranking_result = group_rankings[group_name]
        
        # Get top N assets
        top_assets = ranking_result.get_top_n(max_assets_per_group)
        
        # Calculate weights for this group
        group_weights = calculate_asset_weights_in_group(
            group_name,
            group_budgets[group_name],
            top_assets,
        )
        
        # Add to total
        all_weights.update(group_weights)
        asset_allocations[group_name] = group_weights
    
    # Step 3: Apply exception multiplier
    if exception_symbols:
        all_weights = apply_exception_multiplier(
            all_weights,
            exception_symbols,
            exception_weight_multiplier
        )
    
    # Step 4: Normalize to ensure sum <= 1.0
    all_weights = normalize_weights(all_weights, max_total=1.0)
    
    # Step 5: Calculate cash weight
    invested_weight = sum(all_weights.values())
    cash_weight = max(0.0, 1.0 - invested_weight)
    
    return PortfolioWeights(
        as_of_date=as_of_date,
        weights=all_weights,
        active_groups=active_groups,
        exception_symbols=exception_symbols,
        cash_weight=cash_weight,
        regime_state=regime_state,
        risk_budget=risk_budget,
        group_budgets=group_budgets,
        asset_allocations=asset_allocations,
    )


# ============================================================================
# High-Level API
# ============================================================================

class PortfolioBuilder:
    """
    Builds target portfolio from strategy signals
    
    Examples:
        >>> builder = PortfolioBuilder(config)
        >>> 
        >>> result = builder.build(
        ...     regime_result=regime_result,
        ...     group_strength=group_strength,
        ...     group_rankings=rankings,
        ...     exceptions=exceptions,
        ...     as_of_date=pd.Timestamp("2024-02-01")
        ... )
        >>> 
        >>> portfolio = result.portfolio
        >>> print(f"Invested: {portfolio.get_invested_weight():.1%}")
        >>> print(f"Cash: {portfolio.cash_weight:.1%}")
    """
    
    def __init__(self, config: AdaptiveRotationConfig):
        """
        Initialize portfolio builder
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        
        # Portfolio constraints
        self.max_active_groups = config.portfolio.max_active_groups
        self.max_assets_per_group = config.ranking.top_n_per_group
        self.allow_exception = config.portfolio.allow_exception
        self.exception_weight_multiplier = config.portfolio.exception_weight_multiplier
        
        # Regime budgets
        self.risk_on_budget = 1.0 - config.market_regime.slow_regime.mapping.risk_on.cash_floor
        self.risk_off_budget = 1.0 - config.market_regime.slow_regime.mapping.risk_off.cash_floor
    
    def build(
        self,
        regime_result: MarketRegimeResult,
        group_strength: GroupStrengthResult,
        group_rankings: Dict[str, GroupRankingResult],
        exceptions: ExceptionDetectionResult,
        as_of_date: pd.Timestamp,
    ) -> PortfolioBuildResult:
        """
        Build portfolio from all strategy signals
        
        Args:
            regime_result: Market regime detection
            group_strength: Group strength analysis
            group_rankings: Intra-group rankings
            exceptions: Exception detection
            as_of_date: Current date
        
        Returns:
            PortfolioBuildResult with portfolio and diagnostics
        """
        warnings = []
        constraints = {}
        
        # Step 1: Calculate risk budget from regime
        risk_budget, regime_state = calculate_risk_budget(
            regime_result,
            self.risk_on_budget,
            self.risk_off_budget
        )
        constraints['risk_budget'] = risk_budget
        constraints['regime_state'] = regime_state
        
        # Step 2: Get active groups
        active_groups = group_strength.active_groups[:self.max_active_groups]
        
        # Check if fallback is needed
        use_fallback = False
        fallback_symbols = []
        
        if len(active_groups) == 0:
            # Check if fallback is enabled
            fallback_config = getattr(self.config.portfolio, 'fallback', None)
            if fallback_config is not None and fallback_config.enabled:
                use_fallback = True
                fallback_symbols = fallback_config.symbols
                warnings.append(f"No active groups selected - using fallback: {', '.join(fallback_symbols)}")
            else:
                warnings.append("No active groups selected")
        
        constraints['max_active_groups'] = self.max_active_groups
        constraints['actual_active_groups'] = len(active_groups)
        constraints['fallback_enabled'] = use_fallback
        
        # Step 3: Get exception symbols
        exception_symbols = []
        if self.allow_exception:
            exception_symbols = exceptions.get_qualified_symbols()
        
        constraints['allow_exception'] = self.allow_exception
        constraints['exception_count'] = len(exception_symbols)
        
        # Step 4: Build portfolio
        if use_fallback:
            # Build fallback portfolio
            portfolio = build_fallback_portfolio(
                fallback_symbols=fallback_symbols,
                risk_budget=risk_budget,
                regime_state=regime_state,
                as_of_date=as_of_date,
            )
        else:
            # Normal portfolio construction
            portfolio = build_portfolio_weights(
                active_groups=active_groups,
                group_rankings=group_rankings,
                exception_symbols=exception_symbols,
                risk_budget=risk_budget,
                regime_state=regime_state,
                as_of_date=as_of_date,
                max_assets_per_group=self.max_assets_per_group,
                exception_weight_multiplier=self.exception_weight_multiplier,
            )
        
        # Step 5: Validate
        if not portfolio.validate():
            warnings.append(f"Portfolio weights sum to {sum(portfolio.weights.values()):.4f} > 1.0")
        
        return PortfolioBuildResult(
            portfolio=portfolio,
            regime_result=regime_result,
            group_strength=group_strength,
            group_rankings=group_rankings,
            exceptions=exceptions,
            constraints_applied=constraints,
            warnings=warnings,
        )


if __name__ == "__main__":
    """Quick test of portfolio builder"""
    
    print("Testing Portfolio Builder")
    print("=" * 60)
    
    # This is a placeholder - full integration test requires all modules
    print("\n[Note: Full integration testing requires all strategy modules]")
    print("      Portfolio builder will be tested via unit tests")
    
    print(f"\n{'='*60}")
    print("[INFO] Portfolio builder module loaded successfully!")
