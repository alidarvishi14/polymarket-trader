"""Convex QP curve fitting for cumulative hazard.

This module fits the cumulative hazard H(T) to market prices using a convex
quadratic program that:
1. Minimizes weighted squared errors in log-survival space
2. Adds a smoothness penalty (second-difference regularization)
3. Enforces no-arbitrage constraints (non-negative, non-decreasing hazards)

The optimization is:
    min_H  Σ w_i (H_i + log(1 - P_i^mkt))² + α Σ (H_{i+1} - 2H_i + H_{i-1})²
    s.t.   H_1 ≥ 0
           H_i ≥ H_{i-1}  for all i

This is a convex QP because the objective is quadratic convex and constraints are linear.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel, MaturityData

logger = logging.getLogger(__name__)


class LossType(Enum):
    """Loss function type for curve fitting."""

    QUADRATIC = "quadratic"
    HUBER = "huber"


@dataclass(frozen=True)
class CurveFitResult:
    """Result of hazard curve fitting.

    Attributes:
        model: The fitted HazardRateModel.
        objective_value: Final objective function value.
        fit_residuals: Residuals in log-survival space for each maturity.
        solver_status: Status string from the solver.
        iterations: Number of solver iterations.

    """

    model: HazardRateModel
    objective_value: float
    fit_residuals: NDArray[np.float64]
    solver_status: str
    iterations: int | None


class HazardCurveFitter:
    """Convex QP curve fitter for cumulative hazard.

    Fits a no-arbitrage hazard curve to market prices using convex optimization.
    The key insight is to work in log-survival (cumulative hazard) space where
    the no-arbitrage constraints become linear.
    """

    def __init__(
        self,
        smoothness_alpha: float,
        loss_type: LossType,
        huber_delta: float | None = None,
        min_hazard_increment: float = 0.0,
    ) -> None:
        """Initialize the curve fitter.

        Args:
            smoothness_alpha: Smoothness penalty weight (α). Higher values give
                smoother curves but worse fit. Typical range: 0.001 to 1.0.
            loss_type: Type of loss function (QUADRATIC or HUBER).
            huber_delta: Delta parameter for Huber loss (required if loss_type is HUBER).
            min_hazard_increment: Minimum increment between consecutive hazards
                (for numerical stability). Default 0.

        Raises:
            ValueError: If parameters are invalid.

        """
        if smoothness_alpha < 0:
            raise ValueError(f"smoothness_alpha must be non-negative, got {smoothness_alpha}")

        if loss_type == LossType.HUBER and huber_delta is None:
            raise ValueError("huber_delta required when loss_type is HUBER")

        if huber_delta is not None and huber_delta <= 0:
            raise ValueError(f"huber_delta must be positive, got {huber_delta}")

        self._smoothness_alpha = smoothness_alpha
        self._loss_type = loss_type
        self._huber_delta = huber_delta
        self._min_hazard_increment = min_hazard_increment

        logger.info(
            "HazardCurveFitter initialized: alpha=%.4f, loss=%s, huber_delta=%s",
            smoothness_alpha,
            loss_type.value,
            huber_delta,
        )

    def fit(
        self,
        maturity_data: Sequence[MaturityData],
        weights: NDArray[np.float64] | None = None,
        solver: str | None = None,
    ) -> CurveFitResult:
        """Fit cumulative hazard curve to market data.

        Args:
            maturity_data: Sequence of MaturityData objects, must be sorted by maturity.
            weights: Optional weights for each maturity (e.g., based on liquidity).
                If None, uses sqrt(volume) as weights.
            solver: Optional CVXPY solver to use (e.g., 'ECOS', 'OSQP', 'SCS').

        Returns:
            CurveFitResult containing the fitted model and diagnostics.

        Raises:
            ValueError: If maturity data is invalid or optimization fails.

        """
        if len(maturity_data) < 2:
            raise ValueError(f"Need at least 2 maturities, got {len(maturity_data)}")

        # Sort by maturity and extract arrays
        sorted_data = sorted(maturity_data, key=lambda x: x.maturity)
        maturities = np.array([d.maturity for d in sorted_data], dtype=np.float64)
        market_prices = np.array([d.market_price for d in sorted_data], dtype=np.float64)

        # Validate monotonicity of market prices (soft check - we'll fit anyway)
        price_diffs = np.diff(market_prices)
        if np.any(price_diffs < -0.01):  # Allow small tolerance
            logger.warning(
                "Market prices not monotonic - this suggests calendar arbitrage: %s",
                dict(zip(maturities, market_prices, strict=True)),
            )

        # Compute weights
        if weights is None:
            weights = np.ones(len(maturities), dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
            if len(weights) != len(maturities):
                raise ValueError(
                    f"Weights length {len(weights)} doesn't match data length {len(maturities)}"
                )

        # Normalize weights
        weights = weights / weights.sum()

        # Target log-survival: -log(1 - P^mkt)
        # Clamp prices to avoid log(0)
        clamped_prices = np.clip(market_prices, 1e-6, 1.0 - 1e-6)
        target_hazards = -np.log(1.0 - clamped_prices)

        # Setup optimization
        n = len(maturities)
        h_var = cp.Variable(n, name="cumulative_hazard")

        # Fit loss: weighted squared error in hazard space
        residuals = h_var - target_hazards
        if self._loss_type == LossType.QUADRATIC:
            fit_loss = cp.sum(cp.multiply(weights, cp.square(residuals)))
        else:  # HUBER
            fit_loss = cp.sum(cp.multiply(weights, cp.huber(residuals, M=self._huber_delta)))

        # Smoothness penalty: second-difference (curvature)
        if n >= 3:
            second_diff = h_var[2:] - 2 * h_var[1:-1] + h_var[:-2]
            smoothness_penalty = cp.sum_squares(second_diff)
        else:
            smoothness_penalty = 0.0

        objective = cp.Minimize(fit_loss + self._smoothness_alpha * smoothness_penalty)

        # Constraints: non-negative, non-decreasing
        constraints = [
            h_var[0] >= 0,  # First hazard non-negative
        ]

        # h_var_i >= h_var_{i-1} + min_increment for all i > 0
        for i in range(1, n):
            constraints.append(h_var[i] >= h_var[i - 1] + self._min_hazard_increment)

        # Solve
        problem = cp.Problem(objective, constraints)

        try:
            if solver:
                problem.solve(solver=solver)
            else:
                problem.solve()
        except cp.SolverError as e:
            raise ValueError(f"Optimization solver failed: {e}") from e

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError(
                f"Optimization failed with status: {problem.status}. "
                f"This may indicate infeasible no-arbitrage constraints."
            )

        fitted_hazards = h_var.value
        if fitted_hazards is None:
            raise ValueError("Solver returned None for hazard values")

        # Ensure numerical constraints are satisfied (clip tiny negative values)
        fitted_hazards = np.maximum(fitted_hazards, 0.0)

        # Ensure monotonicity (fix any tiny violations from numerical precision)
        for i in range(1, len(fitted_hazards)):
            fitted_hazards[i] = max(fitted_hazards[i], fitted_hazards[i - 1])

        # Create the model
        model = HazardRateModel(
            maturities=maturities,
            cumulative_hazards=fitted_hazards,
        )

        # Compute fit residuals
        fit_residuals = fitted_hazards - target_hazards

        # Get iteration count if available
        iterations = None
        if hasattr(problem, "solver_stats") and problem.solver_stats is not None:
            iterations = getattr(problem.solver_stats, "num_iters", None)

        logger.info(
            "Curve fit complete: status=%s, objective=%.6f, max_residual=%.4f",
            problem.status,
            problem.value,
            np.max(np.abs(fit_residuals)),
        )

        return CurveFitResult(
            model=model,
            objective_value=float(problem.value),
            fit_residuals=fit_residuals,
            solver_status=str(problem.status),
            iterations=iterations,
        )

    def stress_test_alpha(
        self,
        maturity_data: Sequence[MaturityData],
        alpha_values: Sequence[float],
    ) -> dict[float, CurveFitResult]:
        """Fit curves across multiple smoothness values for robustness analysis.

        This helps identify which mispricings are robust to model choice
        vs. artifacts of a particular smoothness setting.

        Args:
            maturity_data: Market data to fit.
            alpha_values: Sequence of alpha values to test.

        Returns:
            Dictionary mapping alpha values to CurveFitResults.

        """
        results = {}
        for alpha in alpha_values:
            fitter = HazardCurveFitter(
                smoothness_alpha=alpha,
                loss_type=self._loss_type,
                huber_delta=self._huber_delta,
                min_hazard_increment=self._min_hazard_increment,
            )
            try:
                results[alpha] = fitter.fit(maturity_data)
                logger.info(
                    "Alpha=%.4f: objective=%.6f",
                    alpha,
                    results[alpha].objective_value,
                )
            except ValueError:
                logger.warning("Fit failed for alpha=%.4f", alpha, exc_info=True)

        return results
