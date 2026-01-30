#!/usr/bin/env python3
"""Simulation demonstrating Bayesian estimator behavior with isolated price moves.

This script shows how the Bayesian estimator:
1. Builds confidence over stable periods
2. Resists isolated/suspicious price moves
3. Creates larger mispricings (trading opportunities) when moves are inconsistent

Scenario:
- Market prices stable for 10 iterations
- Then mar31 suddenly moves up +1.5%, feb28 moves down -1.5%
- We observe the model's response and trade recommendations
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from polymarket_trader.models.bayesian_hazard import (
    BayesianHazardConfig,
    BayesianHazardEstimator,
)
from polymarket_trader.models.hazard import HazardRateModel, MaturityData
from polymarket_trader.optimization.simple_portfolio import optimize_simple_portfolio

logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner output
    format="%(message)s",
)

EASTERN_TZ = ZoneInfo("America/New_York")


def make_market_data(prices: dict[str, float], spreads: float = 0.005) -> list[MaturityData]:
    """Create MaturityData from price dict.

    Args:
        prices: Dict mapping contract_id to mid price.
        spreads: Half-spread (bid = mid - spread, ask = mid + spread).

    Returns:
        List of MaturityData sorted by maturity.

    """
    # Maturities based on the terminal output
    maturities = {
        "jan30": 1.0,
        "jan31": 2.0,
        "feb6": 8.0,
        "feb13": 15.0,
        "feb28": 30.0,
        "mar31": 61.0,
        "jun30": 152.0,
    }

    data = []
    for contract_id, mid_price in prices.items():
        data.append(
            MaturityData(
                maturity=maturities[contract_id],
                market_price=mid_price,
                bid_price=mid_price - spreads,
                ask_price=mid_price + spreads,
                volume=10000.0,
                contract_id=contract_id,
            )
        )

    return sorted(data, key=lambda x: x.maturity)


def print_prices_table(
    iteration: int,
    market_data: list[MaturityData],
    model: HazardRateModel,
    title: str = "",
) -> None:
    """Print a formatted price comparison table."""
    market_prices = np.array([d.market_price for d in market_data])
    model_prices = model.theoretical_prices
    mispricings = model.mispricings(market_prices)

    print(f"\n{'=' * 80}")
    print(f"ITERATION {iteration}: {title}")
    print(f"{'=' * 80}")
    print(f"{'Contract':<10} {'Market':>10} {'Model':>10} {'Mispricing':>12} {'Signal':<15}")
    print("-" * 80)

    for i, d in enumerate(market_data):
        mispricing = mispricings[i]
        if mispricing > 0.005:
            signal = "BUY YES ↑"
        elif mispricing < -0.005:
            signal = "BUY NO ↓"
        else:
            signal = "-"

        print(
            f"{d.contract_id:<10} "
            f"{d.market_price:>10.2%} "
            f"{model_prices[i]:>10.2%} "
            f"{mispricing:>+12.2%} "
            f"{signal:<15}"
        )


def run_portfolio_optimization(
    model: HazardRateModel,
    market_data: list[MaturityData],
    budget: float = 1000.0,
) -> dict:
    """Run portfolio optimization and return summary."""
    market_prices = np.array([d.market_price for d in market_data])
    bid_prices = np.array([d.bid_price for d in market_data])
    ask_prices = np.array([d.ask_price for d in market_data])

    result = optimize_simple_portfolio(
        model=model,
        market_prices=market_prices,
        budget=budget,
        min_theta=-1000.0,  # No theta constraint for simplicity
        bid_prices=bid_prices,
        ask_prices=ask_prices,
        use_spread=True,
        target_delta=0.0,
        target_theta=0.0,
        delta_tolerance=50.0,  # Wide tolerance
        theta_tolerance=50.0,
    )

    return {
        "positions": result.positions,
        "total_cost": result.total_cost,
        "total_edge": result.total_edge,
        "delta": result.delta_exposure,
        "theta": result.theta_exposure,
    }


def print_trade_recommendations(
    market_data: list[MaturityData],
    result: dict,
) -> None:
    """Print trade recommendations."""
    print(f"\n{'─' * 80}")
    print("TRADE RECOMMENDATIONS")
    print(f"{'─' * 80}")
    print(f"{'Contract':<10} {'Action':<12} {'Quantity':>10} {'Net Position':>12}")
    print("-" * 80)

    has_trades = False
    for i, d in enumerate(market_data):
        pos = result["positions"][i]

        cid, net = d.contract_id, pos.net_exposure
        if pos.buy_yes > 0.1:
            print(f"{cid:<10} {'BUY YES':<12} {pos.buy_yes:>10.2f} {net:>+12.2f}")
            has_trades = True
        if pos.sell_yes > 0.1:
            print(f"{cid:<10} {'SELL YES':<12} {pos.sell_yes:>10.2f} {net:>+12.2f}")
            has_trades = True
        if pos.buy_no > 0.1:
            print(f"{cid:<10} {'BUY NO':<12} {pos.buy_no:>10.2f} {net:>+12.2f}")
            has_trades = True
        if pos.sell_no > 0.1:
            print(f"{cid:<10} {'SELL NO':<12} {pos.sell_no:>10.2f} {net:>+12.2f}")
            has_trades = True

    if not has_trades:
        print("  No significant trades recommended.")

    print("-" * 80)
    print(f"  Total cost:     ${result['total_cost']:>10.2f}")
    print(f"  Expected edge:  ${result['total_edge']:>10.4f}")
    print(f"  Final delta:    {result['delta']:>+10.2f}")
    print(f"  Final theta:    {result['theta']:>+10.4f}")


def main() -> None:
    """Run the simulation."""
    print("=" * 80)
    print("BAYESIAN ESTIMATOR SIMULATION: ISOLATED PRICE MOVE")
    print("=" * 80)
    print()
    print("Scenario:")
    print("  1. Market prices stable for 10 iterations")
    print("  2. Then mar31 moves UP +1.5%, feb28 moves DOWN -1.5%")
    print("  3. Observe how the Bayesian model responds")
    print()

    # Baseline prices (from the terminal output)
    baseline_prices = {
        "jan30": 0.058,
        "jan31": 0.105,
        "feb6": 0.255,
        "feb13": 0.355,
        "feb28": 0.495,
        "mar31": 0.595,
        "jun30": 0.655,
    }

    # Initialize Bayesian estimator with high correlation (resists isolated moves)
    config = BayesianHazardConfig(
        rho=0.90,  # High correlation - strongly resists isolated moves
        log_lambda_std=0.3,
        obs_noise_scale=0.1,  # Trust observations somewhat
        process_noise_std=0.005,  # Low process noise - beliefs are stable
        min_variance=1e-4,
        isolation_penalty=3.0,  # Penalize isolated moves
    )

    estimator = BayesianHazardEstimator(config)
    now = datetime.now(EASTERN_TZ)

    # --- Phase 1: Stable period (10 iterations) ---
    print("\n" + "=" * 80)
    print("PHASE 1: STABLE PERIOD (10 iterations)")
    print("=" * 80)

    for i in range(10):
        # Add small noise to prices to simulate market microstructure
        noisy_prices = {k: v + np.random.uniform(-0.002, 0.002) for k, v in baseline_prices.items()}
        market_data = make_market_data(noisy_prices)

        if i == 0:
            estimator.initialize(market_data, now)
            print(f"Iteration {i + 1}: Initialized estimator")
        else:
            diagnostics = estimator.update(market_data, now)
            if i % 3 == 0 or i == 9:  # Print every 3rd iteration
                print(
                    f"Iteration {i + 1}: uncertainty={diagnostics.posterior_uncertainty:.4f}, "
                    f"innovation_norm={np.linalg.norm(diagnostics.innovation):.4f}"
                )

    # Show model after stable period
    stable_market_data = make_market_data(baseline_prices)
    stable_model = estimator.get_model()
    print_prices_table(10, stable_market_data, stable_model, "End of stable period")

    # Get confidence intervals
    lower, upper = estimator.get_confidence_intervals()
    print("\n95% Confidence Intervals (after 10 stable iterations):")
    for i, d in enumerate(stable_market_data):
        print(f"  {d.contract_id}: [{lower[i]:.2%}, {upper[i]:.2%}]")

    # --- Phase 2: Sudden isolated move ---
    print("\n" + "=" * 80)
    print("PHASE 2: SUDDEN ISOLATED MOVE")
    print("=" * 80)
    print()
    print("Price changes:")
    print("  mar31: +1.5% (0.595 → 0.610)")
    print("  feb28: -1.5% (0.495 → 0.480)")
    print("  Others: unchanged")
    print()

    # Create the "shocked" prices
    shocked_prices = baseline_prices.copy()
    shocked_prices["mar31"] = 0.610  # +1.5%
    shocked_prices["feb28"] = 0.480  # -1.5%

    shocked_market_data = make_market_data(shocked_prices)

    # Update Bayesian estimator with shocked prices
    diagnostics = estimator.update(shocked_market_data, now)
    bayesian_model = estimator.get_model()

    print("Bayesian update diagnostics:")
    print("  Innovation (how much prices deviate from prediction):")
    for i, d in enumerate(shocked_market_data):
        print(f"    {d.contract_id}: {diagnostics.innovation[i]:+.4f}")
    print("  Kalman gain diagonal (how much we trust observations):")
    for i, d in enumerate(shocked_market_data):
        print(f"    {d.contract_id}: {diagnostics.kalman_gain_diag[i]:.4f}")

    # --- Compare Bayesian vs QP fitter ---
    print("\n" + "=" * 80)
    print("COMPARISON: BAYESIAN vs QP FITTER")
    print("=" * 80)

    # QP fitter (fits exactly to current prices)
    from polymarket_trader.optimization.curve_fitting import HazardCurveFitter, LossType

    qp_fitter = HazardCurveFitter(smoothness_alpha=0.01, loss_type=LossType.QUADRATIC)
    qp_result = qp_fitter.fit(shocked_market_data)
    qp_model = qp_result.model

    print_prices_table(
        11, shocked_market_data, bayesian_model, "BAYESIAN MODEL (resists isolated moves)"
    )
    print_prices_table(11, shocked_market_data, qp_model, "QP FITTER (fits to current prices)")

    # --- Highlight the key difference ---
    print("\n" + "=" * 80)
    print("KEY INSIGHT: MISPRICING COMPARISON")
    print("=" * 80)

    market_prices = np.array([d.market_price for d in shocked_market_data])
    bayesian_mispricings = bayesian_model.mispricings(market_prices)
    qp_mispricings = qp_model.mispricings(market_prices)

    print(f"\n{'Contract':<10} {'Bayesian':>15} {'QP Fitter':>15} {'Difference':>15}")
    print("-" * 60)
    for i, d in enumerate(shocked_market_data):
        diff = abs(bayesian_mispricings[i]) - abs(qp_mispricings[i])
        marker = "← LARGER" if diff > 0.002 else ""
        print(
            f"{d.contract_id:<10} "
            f"{bayesian_mispricings[i]:>+15.2%} "
            f"{qp_mispricings[i]:>+15.2%} "
            f"{marker:>15}"
        )

    print()
    print("The Bayesian model shows LARGER mispricings for the contracts that moved")
    print("in isolation (mar31, feb28) because it doesn't fully believe the move.")
    print()

    # --- Trade recommendations ---
    print("\n" + "=" * 80)
    print("PORTFOLIO OPTIMIZATION (Starting fresh with $1000 budget)")
    print("=" * 80)

    print("\n--- Using BAYESIAN model ---")
    bayesian_result = run_portfolio_optimization(bayesian_model, shocked_market_data)
    print_trade_recommendations(shocked_market_data, bayesian_result)

    print("\n--- Using QP model ---")
    qp_result = run_portfolio_optimization(qp_model, shocked_market_data)
    print_trade_recommendations(shocked_market_data, qp_result)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The Bayesian model:")
    print("  1. Maintains beliefs from the stable period")
    print("  2. Doesn't fully trust the isolated move in mar31/feb28")
    print("  3. Creates larger mispricings → stronger trading signals")
    print("  4. Recommends fading the move (sell mar31, buy feb28)")
    print()
    print("The QP fitter:")
    print("  1. Fits exactly to current prices")
    print("  2. Has minimal mispricings")
    print("  3. Weaker trading signals")
    print()


if __name__ == "__main__":
    main()
