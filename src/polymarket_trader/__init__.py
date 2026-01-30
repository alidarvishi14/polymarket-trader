"""Polymarket Trader: Event-time term structure arbitrage system for binary markets.

This package implements a survival/hazard-rate framework for trading binary contracts
indexed by maturities, with proper no-arbitrage curve fitting, portfolio optimization,
and risk management.
"""

from polymarket_trader.execution.exit_rules import ExitRuleEngine
from polymarket_trader.execution.rebalancing import RebalancingPolicy
from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.curve_fitting import HazardCurveFitter
from polymarket_trader.optimization.portfolio import PortfolioOptimizer
from polymarket_trader.risk.hedging import HedgingEngine

__version__ = "0.1.0"

__all__ = [
    "HazardRateModel",
    "HazardCurveFitter",
    "PortfolioOptimizer",
    "HedgingEngine",
    "RebalancingPolicy",
    "ExitRuleEngine",
]
