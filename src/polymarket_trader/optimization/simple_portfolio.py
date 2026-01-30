"""Simple portfolio optimizer for binary market arbitrage.

Supports both from-scratch and incremental optimization with position closing.

Formulation uses 4 decision variables per maturity:
    - buy_yes_i:  YES contracts to buy (≥ 0)
    - sell_yes_i: YES contracts to sell (≥ 0, limited by current YES holdings)
    - buy_no_i:   NO contracts to buy (≥ 0)
    - sell_no_i:  NO contracts to sell (≥ 0, limited by current NO holdings)

Position change: Δz_i = buy_yes_i - sell_yes_i - buy_no_i + sell_no_i
Final position:  z_new_i = z_current_i + Δz_i

Cash flows (with spread):
    - Buy YES:  pay P_ask
    - Sell YES: receive P_bid
    - Buy NO:   pay (1 - P_bid)
    - Sell NO:  receive (1 - P_ask)

Budget constraint:
    net_cash_outflow = cost_of_buying - revenue_from_selling ≤ budget
    (can be negative if closing positions returns more than new positions cost)
"""

import logging
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimplePosition:
    """Position in a single contract.

    Attributes:
        maturity: Time to maturity.
        buy_yes: YES contracts bought in this optimization.
        sell_yes: YES contracts sold in this optimization.
        buy_no: NO contracts bought in this optimization.
        sell_no: NO contracts sold in this optimization.
        net_exposure: Final net YES exposure after trades.
        market_price: Market mid price.
        model_price: Model theoretical price.
        edge_per_contract: Edge per contract for increasing YES exposure.

    """

    maturity: float
    buy_yes: float
    sell_yes: float
    buy_no: float
    sell_no: float
    net_exposure: float
    market_price: float
    model_price: float
    edge_per_contract: float


@dataclass(frozen=True)
class SimplePortfolioResult:
    """Result of simple portfolio optimization.

    Attributes:
        positions: List of positions.
        total_cost: Net cash outflow (negative means net inflow from closing).
        total_edge: Total expected profit from trades.
        delta_exposure: Final portfolio delta.
        theta_exposure: Final portfolio theta.
        solver_status: Solver status.

    """

    positions: list[SimplePosition]
    total_cost: float
    total_edge: float
    delta_exposure: float
    theta_exposure: float
    solver_status: str


def optimize_simple_portfolio(
    model: HazardRateModel,
    market_prices: NDArray[np.float64],
    budget: float,
    min_theta: float,
    bid_prices: NDArray[np.float64] | None = None,
    ask_prices: NDArray[np.float64] | None = None,
    use_spread: bool = False,
    current_positions: NDArray[np.float64] | None = None,
    target_delta: float = 0.0,
    target_theta: float | None = None,
    delta_tolerance: float | None = None,
    theta_tolerance: float | None = None,
) -> SimplePortfolioResult:
    """Optimize portfolio with 4-variable formulation supporting position closing.

    This formulation allows:
    - Buying new YES/NO positions
    - Selling existing positions (with cash returned to budget)

    Args:
        model: Fitted HazardRateModel with theoretical prices.
        market_prices: Array of mid-market prices.
        budget: Maximum net cash outflow for trades.
        min_theta: Minimum theta constraint for final portfolio.
        bid_prices: Array of bid prices (required if use_spread=True).
        ask_prices: Array of ask prices (required if use_spread=True).
        use_spread: If True, use bid/ask for pricing instead of mid.
        current_positions: Array of current net positions (positive=long YES).
            If None, assumes from-scratch optimization.
        target_delta: Target delta exposure (default 0).
        target_theta: Target theta exposure. If None, only min_theta is used.
        delta_tolerance: Tolerance band for delta. If None, uses strict equality.
        theta_tolerance: Tolerance band for theta. If None, only min_theta is used.

    Returns:
        SimplePortfolioResult with optimal trades and final positions.

    Raises:
        ValueError: If optimization fails or required prices missing.

    """
    n = len(market_prices)

    if use_spread:
        if bid_prices is None or ask_prices is None:
            raise ValueError("bid_prices and ask_prices required when use_spread=True")
        if len(bid_prices) != n or len(ask_prices) != n:
            raise ValueError(
                f"bid/ask prices length mismatch: bid={len(bid_prices)}, "
                f"ask={len(ask_prices)}, expected={n}"
            )

    if len(model.maturities) != n:
        raise ValueError(
            f"Market prices length {n} doesn't match model maturities {len(model.maturities)}"
        )

    # Handle current positions
    if current_positions is None:
        current_positions = np.zeros(n, dtype=np.float64)
    else:
        current_positions = np.asarray(current_positions, dtype=np.float64)
        if len(current_positions) != n:
            raise ValueError(
                f"current_positions length {len(current_positions)} doesn't match "
                f"market prices length {n}"
            )

    # Model quantities
    maturities = model.maturities
    model_prices = model.theoretical_prices  # P_model
    survival_probs = model.survival_probabilities  # S = 1 - P_model
    bucket_hazards = model.bucket_hazards  # λ

    # ==========================================================================
    # PRICING: Determine costs and revenues for each action
    # ==========================================================================
    if use_spread:
        # With spread: different prices for buying vs selling
        buy_yes_price = ask_prices  # Pay ask to buy YES
        sell_yes_price = bid_prices  # Receive bid when selling YES
        buy_no_price = 1.0 - bid_prices  # Pay (1 - bid) to buy NO
        sell_no_price = 1.0 - ask_prices  # Receive (1 - ask) when selling NO
    else:
        # Mid-market: symmetric pricing
        buy_yes_price = market_prices
        sell_yes_price = market_prices
        buy_no_price = 1.0 - market_prices
        sell_no_price = 1.0 - market_prices

    # ==========================================================================
    # EDGE CALCULATION
    # ==========================================================================
    # Edge = expected value change - cash paid (or + cash received)
    # Buy YES:  EV = P_model, Cost = buy_yes_price, Edge = P_model - buy_yes_price
    # Sell YES: EV lost = P_model, Revenue = sell_yes_price, Edge = sell_yes_price - P_model
    # Buy NO:   EV = 1 - P_model, Cost = buy_no_price, Edge = (1-P_model) - buy_no_price
    # Sell NO:  EV lost = 1 - P_model, Revenue = sell_no_price, Edge = sell_no_price - (1-P_model)

    edge_buy_yes = model_prices - buy_yes_price
    edge_sell_yes = sell_yes_price - model_prices
    edge_buy_no = (1.0 - model_prices) - buy_no_price  # = bid - model (with spread)
    edge_sell_no = sell_no_price - (1.0 - model_prices)  # = model - ask (with spread)

    # ==========================================================================
    # DECISION VARIABLES (4 per maturity)
    # ==========================================================================
    buy_yes = cp.Variable(n, nonneg=True, name="buy_yes")
    sell_yes = cp.Variable(n, nonneg=True, name="sell_yes")
    buy_no = cp.Variable(n, nonneg=True, name="buy_no")
    sell_no = cp.Variable(n, nonneg=True, name="sell_no")

    # Position change: Δz = buy_yes - sell_yes - buy_no + sell_no
    delta_z = buy_yes - sell_yes - buy_no + sell_no

    # ==========================================================================
    # OBJECTIVE: Maximize total edge from all trades
    # ==========================================================================
    objective = cp.Maximize(
        cp.sum(cp.multiply(edge_buy_yes, buy_yes))
        + cp.sum(cp.multiply(edge_sell_yes, sell_yes))
        + cp.sum(cp.multiply(edge_buy_no, buy_no))
        + cp.sum(cp.multiply(edge_sell_no, sell_no))
    )

    # ==========================================================================
    # CONSTRAINTS
    # ==========================================================================
    constraints = []

    # 1. Selling constraints: can only sell what you own
    #    - If z_current > 0 (long YES): can sell up to z_current YES, cannot sell NO
    #    - If z_current < 0 (long NO): can sell up to |z_current| NO, cannot sell YES
    #    - If z_current = 0: cannot sell anything
    for i in range(n):
        if current_positions[i] > 0:
            # Long YES: can sell YES up to current holding, cannot sell NO
            constraints.append(sell_yes[i] <= current_positions[i])
            constraints.append(sell_no[i] == 0)
        elif current_positions[i] < 0:
            # Long NO: can sell NO up to current holding, cannot sell YES
            constraints.append(sell_yes[i] == 0)
            constraints.append(sell_no[i] <= -current_positions[i])
        else:
            # No position: cannot sell anything
            constraints.append(sell_yes[i] == 0)
            constraints.append(sell_no[i] == 0)

    # 2. Budget constraint: net cash outflow ≤ budget
    #    Cash outflow = cost of buying - revenue from selling
    cost_buying = cp.sum(cp.multiply(buy_yes_price, buy_yes)) + cp.sum(
        cp.multiply(buy_no_price, buy_no)
    )
    revenue_selling = cp.sum(cp.multiply(sell_yes_price, sell_yes)) + cp.sum(
        cp.multiply(sell_no_price, sell_no)
    )
    net_cash_outflow = cost_buying - revenue_selling
    constraints.append(net_cash_outflow <= budget)

    # Compute current exposures from existing positions
    current_delta = float(np.sum(current_positions * survival_probs))
    theta_weights = bucket_hazards * survival_probs
    current_theta = float(-np.sum(current_positions * theta_weights))

    # 3. Delta constraint
    #    Final delta = current_delta + Σ Δz_i * S_i
    trade_delta = cp.sum(cp.multiply(survival_probs, delta_z))
    final_delta = current_delta + trade_delta

    if delta_tolerance is not None:
        constraints.append(final_delta >= target_delta - delta_tolerance)
        constraints.append(final_delta <= target_delta + delta_tolerance)
    else:
        constraints.append(final_delta == target_delta)

    # 4. Theta constraint
    #    Final theta = current_theta + Σ Δz_i * (-λ_i * S_i)
    trade_theta = -cp.sum(cp.multiply(theta_weights, delta_z))
    final_theta = current_theta + trade_theta

    if theta_tolerance is not None and target_theta is not None:
        constraints.append(final_theta >= target_theta - theta_tolerance)
        constraints.append(final_theta <= target_theta + theta_tolerance)
    else:
        constraints.append(final_theta >= min_theta)

    # ==========================================================================
    # SOLVE
    # ==========================================================================
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve()
    except cp.SolverError as e:
        raise ValueError(f"Solver failed: {e}") from e

    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        raise ValueError(f"Optimization failed: {problem.status}")

    # ==========================================================================
    # EXTRACT RESULTS
    # ==========================================================================
    buy_yes_opt = np.maximum(buy_yes.value, 0) if buy_yes.value is not None else np.zeros(n)
    sell_yes_opt = np.maximum(sell_yes.value, 0) if sell_yes.value is not None else np.zeros(n)
    buy_no_opt = np.maximum(buy_no.value, 0) if buy_no.value is not None else np.zeros(n)
    sell_no_opt = np.maximum(sell_no.value, 0) if sell_no.value is not None else np.zeros(n)

    # Position change and final positions
    delta_z_opt = buy_yes_opt - sell_yes_opt - buy_no_opt + sell_no_opt
    final_positions = current_positions + delta_z_opt

    # Build positions list
    positions = []
    for i in range(n):
        positions.append(
            SimplePosition(
                maturity=float(maturities[i]),
                buy_yes=float(buy_yes_opt[i]),
                sell_yes=float(sell_yes_opt[i]),
                buy_no=float(buy_no_opt[i]),
                sell_no=float(sell_no_opt[i]),
                net_exposure=float(final_positions[i]),
                market_price=float(market_prices[i]),
                model_price=float(model_prices[i]),
                edge_per_contract=float(edge_buy_yes[i]),
            )
        )

    # Compute totals
    total_cost_buying = float(
        np.sum(buy_yes_opt * buy_yes_price) + np.sum(buy_no_opt * buy_no_price)
    )
    total_revenue_selling = float(
        np.sum(sell_yes_opt * sell_yes_price) + np.sum(sell_no_opt * sell_no_price)
    )
    net_cost = total_cost_buying - total_revenue_selling

    total_edge = float(
        np.sum(buy_yes_opt * edge_buy_yes)
        + np.sum(sell_yes_opt * edge_sell_yes)
        + np.sum(buy_no_opt * edge_buy_no)
        + np.sum(sell_no_opt * edge_sell_no)
    )

    # Final portfolio exposures
    delta_val = float(np.sum(final_positions * survival_probs))
    theta_val = float(-np.sum(final_positions * theta_weights))

    is_incremental = np.any(current_positions != 0)
    logger.info(
        "Portfolio optimized (incremental=%s): net_cost=%.2f, edge=%.4f, "
        "final_delta=%.4f, final_theta=%.4f",
        is_incremental,
        net_cost,
        total_edge,
        delta_val,
        theta_val,
    )

    return SimplePortfolioResult(
        positions=positions,
        total_cost=net_cost,
        total_edge=total_edge,
        delta_exposure=delta_val,
        theta_exposure=theta_val,
        solver_status=str(problem.status),
    )
