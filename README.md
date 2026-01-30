# Polymarket Trader

Event-time term structure arbitrage system for binary prediction markets (Polymarket-style "Will X happen by T?" contracts).

## Overview

This package implements a survival/hazard-rate framework for trading binary contracts indexed by maturities, with:

1. **Hazard curve fitting** - Convex QP to fit no-arbitrage cumulative hazard curves
2. **Portfolio optimization** - LP to maximize expected edge with hedging constraints
3. **Risk management** - Factor-neutral hedging (delta, theta, curve factors)
4. **Rebalancing policy** - Band-based rebalancing to avoid over-trading
5. **Exit rules** - Explicit convergence, PnL target, and time-to-expiry rules
6. **Microstructure** - Bid/ask spreads, order book depth, transaction costs

## Installation

```bash
# Install with pip
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Core Concepts

### Survival/Hazard Framework

- **τ** = random event time
- **S(T) = P(τ > T)** = survival probability
- **P(T) = 1 - S(T)** = cumulative probability (YES price)
- **H(T) = -log(S(T))** = cumulative hazard
- **λ(t) = dH(t)/dt** = hazard rate

Market prices are interpreted as cumulative probabilities. All no-arbitrage conditions live in **hazard space** (H must be non-negative and non-decreasing).

### No-Arbitrage Constraints

For maturities T₁ < T₂ < ... < Tₙ:
- Monotonicity: P_{i+1} ≥ P_i
- Bounds: 0 ≤ P_i ≤ 1
- Non-negative hazard: H_i ≥ H_{i-1}

Violations indicate **calendar arbitrage**.

## Quick Start

```python
import numpy as np
from polymarket_trader import (
    HazardCurveFitter,
    PortfolioOptimizer,
    HedgingEngine,
    RebalancingPolicy,
    ExitRuleEngine,
)
from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.optimization.curve_fitting import LossType
from polymarket_trader.optimization.portfolio import OptimizationConfig, HedgeType
from polymarket_trader.risk.hedging import HedgingConfig
from polymarket_trader.execution.rebalancing import RebalanceConfig
from polymarket_trader.execution.exit_rules import ExitConfig

# 1. Create market data
market_data = [
    MaturityData(
        maturity=30.0,
        market_price=0.15,
        bid_price=0.14,
        ask_price=0.16,
        volume=1000.0,
        contract_id="contract-30d",
    ),
    MaturityData(
        maturity=60.0,
        market_price=0.30,
        bid_price=0.29,
        ask_price=0.31,
        volume=800.0,
        contract_id="contract-60d",
    ),
    MaturityData(
        maturity=90.0,
        market_price=0.45,
        bid_price=0.44,
        ask_price=0.46,
        volume=600.0,
        contract_id="contract-90d",
    ),
]

# 2. Fit the hazard curve (convex QP)
fitter = HazardCurveFitter(
    smoothness_alpha=0.1,
    loss_type=LossType.QUADRATIC,
)
fit_result = fitter.fit(market_data)
model = fit_result.model

# 3. Check mispricings
market_prices = np.array([d.market_price for d in sorted(market_data, key=lambda x: x.maturity)])
mispricings = model.mispricings(market_prices)
print(f"Mispricings: {mispricings}")

# 4. Optimize portfolio (LP with hedging)
config = OptimizationConfig(
    budget=10000.0,
    min_edge_per_leg=0.01,
    max_concentration=0.25,
    turnover_penalty=0.001,
    hedge_types={HedgeType.DELTA_NEUTRAL, HedgeType.THETA_NEUTRAL},
)
optimizer = PortfolioOptimizer(model=model, config=config)
portfolio = optimizer.optimize(market_data)

print(f"Total edge: {portfolio.total_edge:.4f}")
print(f"Delta exposure: {portfolio.delta_exposure:.4f}")
print(f"Theta exposure: {portfolio.theta_exposure:.4f}")

# 5. Set up risk monitoring
hedge_config = HedgingConfig(
    target_delta=0.0,
    target_theta=0.0,
    delta_threshold=0.1,
    theta_threshold=0.05,
    factor_threshold=0.1,
)
hedging = HedgingEngine(model=model, config=hedge_config)
exposures = hedging.compute_exposures(portfolio.positions)
drift = hedging.check_drift(exposures)

# 6. Set up rebalancing policy
rebal_config = RebalanceConfig(
    min_days_to_expiry=5.0,
    roll_buffer_days=2.0,
    max_rebalance_frequency_hours=24.0,
    drift_band_pct=0.15,
    transaction_cost_bps=10.0,
    min_edge_after_costs=0.01,
)
rebalancer = RebalancingPolicy(
    model=model,
    hedging_engine=hedging,
    config=rebal_config,
)

# 7. Set up exit rules
exit_config = ExitConfig(
    convergence_threshold=0.01,
    pnl_capture_target=0.7,
    min_days_to_expiry=5.0,
    stop_loss_pct=0.2,
    model_stability_threshold=0.8,
)
exit_engine = ExitRuleEngine(config=exit_config)
```

## Module Structure

```
src/polymarket_trader/
├── models/
│   └── hazard.py          # HazardRateModel, MaturityData
├── optimization/
│   ├── curve_fitting.py   # HazardCurveFitter (convex QP)
│   └── portfolio.py       # PortfolioOptimizer (LP)
├── risk/
│   └── hedging.py         # HedgingEngine, RiskExposures
├── execution/
│   ├── rebalancing.py     # RebalancingPolicy
│   └── exit_rules.py      # ExitRuleEngine
└── microstructure/
    └── constraints.py     # OrderBook, TransactionCostModel
```

## Key Features

### Curve Fitting (Convex QP)

Fits cumulative hazard H(T) using:
```
min_H  Σ w_i (H_i + log(1 - P_i^mkt))² + α Σ (H_{i+1} - 2H_i + H_{i-1})²
s.t.   H_1 ≥ 0
       H_i ≥ H_{i-1}  for all i
```

- Works in log-survival space (convex!)
- Enforces no-arbitrage via linear constraints
- Smoothness penalty controlled by α
- Supports quadratic and Huber loss

### Portfolio Optimization (LP)

Maximizes expected edge:
```
max Σ x_i (P_i^theo - P_i^ask) + Σ y_i (P_i^bid - P_i^theo)
s.t. Budget, concentration, and hedging constraints
```

Hedging options:
- Delta neutral: Σ z_i * S_i = 0
- Theta neutral: Σ z_i * λ_i * S_i = 0
- Factor neutral: front-end, belly, back-end

### Rebalancing Policy

- **Band rebalancing**: Only rebalance when exposures drift beyond tolerance
- **Roll hedges**: Don't top up, roll to next maturity
- **Time triggers**: Auto-roll positions approaching expiry
- **Frequency limits**: Prevent over-trading

### Exit Rules

- **Convergence**: Close when mispricing < threshold
- **PnL target**: Close after capturing 70% of initial edge
- **Time-to-expiry**: Close or roll inside k days
- **Stop loss**: Close if loss exceeds threshold

## Testing

```bash
# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=polymarket_trader

# Run linter
python -m ruff check src/ tests/
```

## Dependencies

- numpy >= 1.24.0
- scipy >= 1.11.0
- cvxpy >= 1.4.0
- pandas >= 2.0.0

## License

MIT
