"""Term structure analysis for binary markets."""

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel, MaturityData
from polymarket_trader.optimization.curve_fitting import (
    CurveFitResult,
    HazardCurveFitter,
    LossType,
)
from polymarket_trader.optimization.simple_portfolio import (
    SimplePortfolioResult,
    optimize_simple_portfolio,
)

logger = logging.getLogger(__name__)


@dataclass
class TermStructureResult:
    """Result of term structure analysis.

    Attributes:
        market_data: Input market data.
        fit_result: Hazard curve fit result.
        model: Fitted hazard rate model.
        mispricings: Array of mispricings (theo - market).
        portfolio: Optimal portfolio result.
        market_prices: Array of market prices.
        model_prices: Array of model theoretical prices.

    """

    market_data: list[MaturityData]
    fit_result: CurveFitResult
    model: HazardRateModel
    mispricings: NDArray[np.float64]
    portfolio: SimplePortfolioResult
    market_prices: NDArray[np.float64]
    model_prices: NDArray[np.float64]

    def print_report(self) -> None:
        """Print a formatted analysis report."""
        print("=" * 70)
        print("TERM STRUCTURE ANALYSIS REPORT")
        print("=" * 70)

        # Market data
        print("\n" + "-" * 70)
        print("MARKET DATA")
        print("-" * 70)
        print(f"{'Contract':<12} {'DTE':>6} {'Price':>8} {'Bid':>8} {'Ask':>8} {'Volume':>12}")
        print("-" * 70)
        for d in self.market_data:
            print(
                f"{d.contract_id:<12} {d.maturity:>6.0f} {d.market_price:>8.2%} "
                f"{d.bid_price:>8.2%} {d.ask_price:>8.2%} ${d.volume:>11,.0f}"
            )

        # Fitted curve
        print("\n" + "-" * 70)
        print("FITTED HAZARD CURVE")
        print("-" * 70)
        print(
            f"{'DTE':>6} {'Market':>10} {'Model':>10} {'Mispricing':>10} "
            f"{'Cum.Hazard':>10} {'λ (rate)':>10}"
        )
        print("-" * 70)

        for i in range(len(self.market_data)):
            print(
                f"{self.model.maturities[i]:>6.0f} {self.market_prices[i]:>10.2%} "
                f"{self.model_prices[i]:>10.2%} {self.mispricings[i]:>+10.2%} "
                f"{self.model.cumulative_hazards[i]:>10.4f} "
                f"{self.model.bucket_hazards[i]:>10.4f}"
            )

        # Portfolio
        print("\n" + "-" * 70)
        print("OPTIMAL PORTFOLIO")
        print("-" * 70)
        print(
            f"{'DTE':>6} {'P_mkt':>8} {'P_model':>8} {'Edge':>8} {'YES':>10} {'NO':>10} {'Net':>10}"
        )
        print("-" * 70)

        for pos in self.portfolio.positions:
            if abs(pos.yes_quantity) > 0.01 or abs(pos.no_quantity) > 0.01:
                print(
                    f"{pos.maturity:>6.0f} {pos.market_price:>8.2%} "
                    f"{pos.model_price:>8.2%} {pos.edge_per_contract:>+8.2%} "
                    f"{pos.yes_quantity:>10.2f} {pos.no_quantity:>10.2f} "
                    f"{pos.net_exposure:>+10.2f}"
                )

        print("-" * 70)

        # Summary
        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"Total Cost:      ${self.portfolio.total_cost:>10.2f}")
        print(f"Total Edge:      ${self.portfolio.total_edge:>10.4f}")
        print(f"Delta exposure:   {self.portfolio.delta_exposure:>+10.6f}")
        print(f"Theta exposure:   {self.portfolio.theta_exposure:>+10.6f}")
        if self.portfolio.total_cost > 0:
            print(
                f"Expected return:  {self.portfolio.total_edge / self.portfolio.total_cost:>10.2%}"
            )

        print("=" * 70)


class TermStructureAnalysis:
    """Analysis engine for term structure arbitrage.

    Combines curve fitting and portfolio optimization into a single workflow.
    """

    def __init__(
        self,
        smoothness_alpha: float = 0.01,
        loss_type: LossType = LossType.QUADRATIC,
        huber_delta: float | None = None,
    ) -> None:
        """Initialize the analysis engine.

        Args:
            smoothness_alpha: Smoothness penalty for curve fitting.
            loss_type: Loss function type for curve fitting.
            huber_delta: Delta parameter for Huber loss.

        """
        self._smoothness_alpha = smoothness_alpha
        self._loss_type = loss_type
        self._huber_delta = huber_delta

        logger.info(
            "TermStructureAnalysis initialized: alpha=%.4f, loss=%s",
            smoothness_alpha,
            loss_type.value,
        )

    def analyze(
        self,
        market_data: list[MaturityData],
        budget: float,
        delta_neutral: bool = True,
        min_theta: float = 0.0,
        use_spread: bool = False,
    ) -> TermStructureResult:
        """Run full term structure analysis.

        Args:
            market_data: List of MaturityData from the market.
            budget: Portfolio budget.
            delta_neutral: Whether to enforce delta neutrality.
            min_theta: Minimum theta constraint (0 = non-negative).
            use_spread: If True, use bid/ask prices for edge calculation.

        Returns:
            TermStructureResult with all analysis outputs.

        Raises:
            ValueError: If analysis fails.

        """
        if len(market_data) < 2:
            raise ValueError(f"Need at least 2 maturities, got {len(market_data)}")

        logger.info(
            "Running analysis on %d contracts, budget=$%.2f, use_spread=%s",
            len(market_data),
            budget,
            use_spread,
        )

        # Step 1: Fit the hazard curve
        fitter = HazardCurveFitter(
            smoothness_alpha=self._smoothness_alpha,
            loss_type=self._loss_type,
            huber_delta=self._huber_delta,
        )
        fit_result = fitter.fit(market_data)
        model = fit_result.model

        logger.info(
            "Curve fit complete: status=%s, objective=%.6f",
            fit_result.solver_status,
            fit_result.objective_value,
        )

        # Extract prices (sorted by maturity)
        sorted_data = sorted(market_data, key=lambda x: x.maturity)
        market_prices = np.array([d.market_price for d in sorted_data])
        bid_prices = np.array([d.bid_price for d in sorted_data])
        ask_prices = np.array([d.ask_price for d in sorted_data])
        model_prices = model.theoretical_prices
        mispricings = model.mispricings(market_prices)

        # Step 2: Optimize portfolio
        portfolio = optimize_simple_portfolio(
            model=model,
            market_prices=market_prices,
            budget=budget,
            min_theta=min_theta,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            use_spread=use_spread,
        )

        logger.info(
            "Portfolio optimized: cost=%.2f, edge=%.4f, delta=%.6f, theta=%.6f",
            portfolio.total_cost,
            portfolio.total_edge,
            portfolio.delta_exposure,
            portfolio.theta_exposure,
        )

        # Sort market_data to match model order
        sorted_data = sorted(market_data, key=lambda x: x.maturity)

        return TermStructureResult(
            market_data=sorted_data,
            fit_result=fit_result,
            model=model,
            mispricings=mispricings,
            portfolio=portfolio,
            market_prices=market_prices,
            model_prices=model_prices,
        )


def analyze_term_structure(
    market_data: list[MaturityData],
    budget: float,
    smoothness_alpha: float = 0.01,
    delta_neutral: bool = True,
    min_theta: float = 0.0,
    use_spread: bool = False,
) -> TermStructureResult:
    """Run term structure analysis with default settings.

    Args:
        market_data: List of MaturityData from the market.
        budget: Portfolio budget.
        smoothness_alpha: Smoothness penalty for curve fitting.
        delta_neutral: Whether to enforce delta neutrality.
        min_theta: Minimum theta constraint.
        use_spread: If True, use bid/ask prices for edge calculation.

    Returns:
        TermStructureResult with all analysis outputs.

    """
    analyzer = TermStructureAnalysis(smoothness_alpha=smoothness_alpha)
    return analyzer.analyze(
        market_data=market_data,
        budget=budget,
        delta_neutral=delta_neutral,
        min_theta=min_theta,
        use_spread=use_spread,
    )
