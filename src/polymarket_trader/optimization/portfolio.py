"""Linear programming portfolio optimizer for binary market arbitrage.

This module constructs optimal portfolios of YES/NO contracts that maximize
expected value under the fitted hazard curve, subject to:
- Budget constraints
- Hedging constraints (delta-neutral, theta-neutral, factor-neutral)
- Liquidity constraints
- Minimum edge requirements

The optimization is:
    max Σ x_i (P_i^theo - P_i^ask) + Σ y_i (P_i^bid - P_i^theo)
    s.t. Σ x_i * P_i^ask + Σ y_i * (1 - P_i^bid) ≤ Budget
         Hedging constraints (factor neutrality)
         0 ≤ x_i ≤ max_position_i
         0 ≤ y_i ≤ max_position_i
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel, MaturityData

logger = logging.getLogger(__name__)


class HedgeType(Enum):
    """Types of hedging constraints."""

    DELTA_NEUTRAL = auto()  # Σ z_i * (1 - P_i) = 0
    THETA_NEUTRAL = auto()  # Σ z_i * λ_i * (1 - P_i) = 0
    FRONT_END_NEUTRAL = auto()  # Neutralize front-end hazard bump
    BELLY_NEUTRAL = auto()  # Neutralize belly hazard bump
    BACK_END_NEUTRAL = auto()  # Neutralize back-end hazard bump


@dataclass(frozen=True)
class LiquidityConstraint:
    """Liquidity constraint for a maturity.

    Attributes:
        maturity_index: Index of the maturity in the sorted array.
        max_yes_position: Maximum YES contracts that can be held.
        max_no_position: Maximum NO contracts that can be held.
        order_book_depth: Available depth in the order book.

    """

    maturity_index: int
    max_yes_position: float
    max_no_position: float
    order_book_depth: float


@dataclass(frozen=True)
class PortfolioPosition:
    """Position in a single contract.

    Attributes:
        maturity: Time to maturity.
        contract_id: Contract identifier.
        yes_quantity: Number of YES contracts (≥ 0).
        no_quantity: Number of NO contracts (≥ 0).
        net_exposure: Net YES exposure (yes_quantity - no_quantity).
        cost_basis: Total cost to acquire the position.
        expected_value: Expected value under the fitted curve.
        edge: Expected profit (EV - cost).

    """

    maturity: float
    contract_id: str
    yes_quantity: float
    no_quantity: float
    net_exposure: float
    cost_basis: float
    expected_value: float
    edge: float


@dataclass(frozen=True)
class PortfolioResult:
    """Result of portfolio optimization.

    Attributes:
        positions: List of positions in each contract.
        total_cost: Total portfolio cost.
        total_expected_value: Total expected value under the curve.
        total_edge: Total expected profit.
        delta_exposure: Portfolio delta to cumulative hazard.
        theta_exposure: Portfolio theta (time decay).
        solver_status: Solver status string.
        objective_value: Final objective value.

    """

    positions: list[PortfolioPosition]
    total_cost: float
    total_expected_value: float
    total_edge: float
    delta_exposure: float
    theta_exposure: float
    solver_status: str
    objective_value: float


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization.

    Attributes:
        budget: Maximum capital to deploy.
        min_edge_per_leg: Minimum edge required to include a leg (spread + fees).
        max_concentration: Maximum fraction of budget in any single maturity.
        turnover_penalty: Penalty for total position size (reduces turnover).
        hedge_types: Set of hedging constraints to apply.
        hedge_tolerance: Tolerance for hedging constraints (soft constraints if > 0).

    """

    budget: float
    min_edge_per_leg: float
    max_concentration: float
    turnover_penalty: float
    hedge_types: set[HedgeType] = field(default_factory=set)
    hedge_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.budget <= 0:
            raise ValueError(f"Budget must be positive, got {self.budget}")
        if self.min_edge_per_leg < 0:
            raise ValueError(f"min_edge_per_leg must be non-negative, got {self.min_edge_per_leg}")
        if not 0 < self.max_concentration <= 1:
            raise ValueError(f"max_concentration must be in (0, 1], got {self.max_concentration}")
        if self.turnover_penalty < 0:
            raise ValueError(f"turnover_penalty must be non-negative, got {self.turnover_penalty}")


class PortfolioOptimizer:
    """Linear programming portfolio optimizer for binary market arbitrage.

    Constructs optimal YES/NO positions that maximize expected edge under
    the fitted hazard curve, subject to budget, hedging, and liquidity constraints.
    """

    def __init__(
        self,
        model: HazardRateModel,
        config: OptimizationConfig,
    ) -> None:
        """Initialize the portfolio optimizer.

        Args:
            model: Fitted HazardRateModel for theoretical pricing.
            config: Optimization configuration.

        """
        self._model = model
        self._config = config

        logger.info(
            "PortfolioOptimizer initialized: budget=%.2f, hedge_types=%s",
            config.budget,
            [h.name for h in config.hedge_types],
        )

    def optimize(
        self,
        maturity_data: Sequence[MaturityData],
        liquidity_constraints: Sequence[LiquidityConstraint] | None = None,
        solver: str | None = None,
    ) -> PortfolioResult:
        """Optimize portfolio allocation.

        Args:
            maturity_data: Market data for each maturity (must match model maturities).
            liquidity_constraints: Optional per-maturity position limits.
            solver: Optional CVXPY solver to use.

        Returns:
            PortfolioResult with optimal positions and diagnostics.

        Raises:
            ValueError: If data is invalid or optimization fails.

        """
        # Sort and extract data
        sorted_data = sorted(maturity_data, key=lambda x: x.maturity)
        n = len(sorted_data)

        maturities = np.array([d.maturity for d in sorted_data])
        bid_prices = np.array([d.bid_price for d in sorted_data])
        ask_prices = np.array([d.ask_price for d in sorted_data])
        contract_ids = [d.contract_id for d in sorted_data]

        # Validate maturities match model
        if not np.allclose(maturities, self._model.maturities, rtol=1e-6):
            raise ValueError("Maturity data maturities don't match model maturities")

        theo_prices = self._model.theoretical_prices
        survival_probs = self._model.survival_probabilities
        bucket_hazards = self._model.bucket_hazards

        # Decision variables: YES and NO positions
        x = cp.Variable(n, nonneg=True, name="yes_positions")  # YES
        y = cp.Variable(n, nonneg=True, name="no_positions")  # NO
        z = x - y  # Net exposure

        # Edge calculation
        # YES edge: P^theo - P^ask (buy YES when underpriced)
        # NO edge: P^bid - P^theo (buy NO when overpriced)
        yes_edge = theo_prices - ask_prices
        no_edge = bid_prices - theo_prices

        # Objective: maximize total edge minus turnover penalty
        objective_expr = cp.sum(cp.multiply(yes_edge, x)) + cp.sum(cp.multiply(no_edge, y))
        if self._config.turnover_penalty > 0:
            objective_expr = objective_expr - self._config.turnover_penalty * (
                cp.sum(x) + cp.sum(y)
            )
        objective = cp.Maximize(objective_expr)

        # Constraints
        constraints = []

        # Budget constraint: cost of all positions
        yes_costs = ask_prices  # Cost to buy YES
        no_costs = 1.0 - bid_prices  # Cost to buy NO (pay 1-bid to seller)
        total_cost = cp.sum(cp.multiply(yes_costs, x)) + cp.sum(cp.multiply(no_costs, y))
        constraints.append(total_cost <= self._config.budget)

        # Concentration constraints
        max_per_maturity = self._config.max_concentration * self._config.budget
        for i in range(n):
            position_cost = yes_costs[i] * x[i] + no_costs[i] * y[i]
            constraints.append(position_cost <= max_per_maturity)

        # Minimum edge constraints (only trade if edge exceeds threshold)
        for i in range(n):
            # If YES edge < min_edge, force x[i] = 0
            if yes_edge[i] < self._config.min_edge_per_leg:
                constraints.append(x[i] == 0)
            # If NO edge < min_edge, force y[i] = 0
            if no_edge[i] < self._config.min_edge_per_leg:
                constraints.append(y[i] == 0)

        # Hedging constraints
        hedge_tolerance = self._config.hedge_tolerance

        if HedgeType.DELTA_NEUTRAL in self._config.hedge_types:
            # Σ z_i * (1 - P_i) = 0
            delta_exposure = cp.sum(cp.multiply(survival_probs, z))
            if hedge_tolerance > 0:
                constraints.append(cp.abs(delta_exposure) <= hedge_tolerance)
            else:
                constraints.append(delta_exposure == 0)

        if HedgeType.THETA_NEUTRAL in self._config.hedge_types:
            # Σ z_i * λ_i * (1 - P_i) = 0
            theta_weights = bucket_hazards * survival_probs
            theta_exposure = cp.sum(cp.multiply(theta_weights, z))
            if hedge_tolerance > 0:
                constraints.append(cp.abs(theta_exposure) <= hedge_tolerance)
            else:
                constraints.append(theta_exposure == 0)

        # Factor neutrality (front/belly/back)
        if HedgeType.FRONT_END_NEUTRAL in self._config.hedge_types:
            front_weights = self._compute_factor_weights(n, "front")
            front_exposure = cp.sum(cp.multiply(front_weights * survival_probs, z))
            if hedge_tolerance > 0:
                constraints.append(cp.abs(front_exposure) <= hedge_tolerance)
            else:
                constraints.append(front_exposure == 0)

        if HedgeType.BELLY_NEUTRAL in self._config.hedge_types:
            belly_weights = self._compute_factor_weights(n, "belly")
            belly_exposure = cp.sum(cp.multiply(belly_weights * survival_probs, z))
            if hedge_tolerance > 0:
                constraints.append(cp.abs(belly_exposure) <= hedge_tolerance)
            else:
                constraints.append(belly_exposure == 0)

        if HedgeType.BACK_END_NEUTRAL in self._config.hedge_types:
            back_weights = self._compute_factor_weights(n, "back")
            back_exposure = cp.sum(cp.multiply(back_weights * survival_probs, z))
            if hedge_tolerance > 0:
                constraints.append(cp.abs(back_exposure) <= hedge_tolerance)
            else:
                constraints.append(back_exposure == 0)

        # Liquidity constraints
        if liquidity_constraints:
            liq_map = {lc.maturity_index: lc for lc in liquidity_constraints}
            for i in range(n):
                if i in liq_map:
                    lc = liq_map[i]
                    constraints.append(x[i] <= lc.max_yes_position)
                    constraints.append(y[i] <= lc.max_no_position)

        # Solve
        problem = cp.Problem(objective, constraints)
        try:
            if solver:
                problem.solve(solver=solver)
            else:
                problem.solve()
        except cp.SolverError as e:
            raise ValueError(f"Portfolio optimization solver failed: {e}") from e

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(f"Portfolio optimization failed: {problem.status}")

        # Extract results
        x_opt = x.value if x.value is not None else np.zeros(n)
        y_opt = y.value if y.value is not None else np.zeros(n)

        # Clip tiny negative values from numerical precision
        x_opt = np.maximum(x_opt, 0)
        y_opt = np.maximum(y_opt, 0)

        # Build positions
        positions = []
        total_cost_val = 0.0
        total_ev = 0.0

        for i in range(n):
            yes_q = float(x_opt[i])
            no_q = float(y_opt[i])
            net = yes_q - no_q

            cost = yes_q * ask_prices[i] + no_q * (1 - bid_prices[i])
            ev = yes_q * theo_prices[i] + no_q * (1 - theo_prices[i])
            edge = ev - cost

            positions.append(
                PortfolioPosition(
                    maturity=float(maturities[i]),
                    contract_id=contract_ids[i],
                    yes_quantity=yes_q,
                    no_quantity=no_q,
                    net_exposure=net,
                    cost_basis=cost,
                    expected_value=ev,
                    edge=edge,
                )
            )

            total_cost_val += cost
            total_ev += ev

        # Compute portfolio Greeks
        z_opt = x_opt - y_opt
        delta_exp = float(np.sum(z_opt * survival_probs))
        theta_exp = float(np.sum(z_opt * bucket_hazards * survival_probs))

        logger.info(
            "Portfolio optimized: cost=%.4f, edge=%.4f, delta=%.4f, theta=%.4f",
            total_cost_val,
            total_ev - total_cost_val,
            delta_exp,
            theta_exp,
        )

        return PortfolioResult(
            positions=positions,
            total_cost=total_cost_val,
            total_expected_value=total_ev,
            total_edge=total_ev - total_cost_val,
            delta_exposure=delta_exp,
            theta_exposure=theta_exp,
            solver_status=str(problem.status),
            objective_value=float(problem.value) if problem.value is not None else 0.0,
        )

    @staticmethod
    def _compute_factor_weights(n: int, factor: str) -> NDArray[np.float64]:
        """Compute weights for hazard factor exposure.

        Creates bump weights for front-end, belly, and back-end of the curve.

        Args:
            n: Number of maturities.
            factor: One of 'front', 'belly', 'back'.

        Returns:
            Array of weights for each maturity.

        """
        x = np.linspace(0, 1, n)

        if factor == "front":
            # Weight decays from 1 at front to 0 at back
            weights = 1.0 - x
        elif factor == "belly":
            # Bell curve centered in middle
            weights = np.exp(-((x - 0.5) ** 2) / 0.1)
        elif factor == "back":
            # Weight grows from 0 at front to 1 at back
            weights = x
        else:
            raise ValueError(f"Unknown factor: {factor}")

        # Normalize
        return weights / (weights.sum() + 1e-10)
