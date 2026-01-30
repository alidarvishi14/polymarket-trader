"""Hedging engine for managing portfolio risk exposure.

This module provides tools to:
1. Compute portfolio risk exposures (Greeks) in hazard space
2. Decompose risk into interpretable factors
3. Monitor exposure drift and trigger rebalancing
4. Construct hedge adjustments
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.portfolio import PortfolioPosition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskExposures:
    """Portfolio risk exposures in hazard space.

    Attributes:
        delta: Sensitivity to parallel hazard shift (Σ z_i * S_i).
        theta: Time decay (Σ z_i * λ_i * S_i).
        front_end: Exposure to front-end hazard bump.
        belly: Exposure to belly hazard bump.
        back_end: Exposure to back-end hazard bump.
        per_maturity_delta: Delta contribution by maturity.
        per_maturity_theta: Theta contribution by maturity.

    """

    delta: float
    theta: float
    front_end: float
    belly: float
    back_end: float
    per_maturity_delta: NDArray[np.float64]
    per_maturity_theta: NDArray[np.float64]


@dataclass(frozen=True)
class DriftReport:
    """Report on exposure drift from target.

    Attributes:
        delta_drift: Absolute drift in delta.
        theta_drift: Absolute drift in theta.
        front_end_drift: Absolute drift in front-end exposure.
        belly_drift: Absolute drift in belly exposure.
        back_end_drift: Absolute drift in back-end exposure.
        max_drift_pct: Maximum drift as percentage of threshold.
        requires_rebalance: Whether any drift exceeds threshold.
        triggering_factors: List of factors that triggered rebalance.

    """

    delta_drift: float
    theta_drift: float
    front_end_drift: float
    belly_drift: float
    back_end_drift: float
    max_drift_pct: float
    requires_rebalance: bool
    triggering_factors: list[str]


@dataclass
class HedgingConfig:
    """Configuration for hedging engine.

    Attributes:
        target_delta: Target delta exposure (typically 0 for neutral).
        target_theta: Target theta exposure (typically 0 for neutral).
        delta_threshold: Maximum allowed delta drift before rebalance.
        theta_threshold: Maximum allowed theta drift before rebalance.
        factor_threshold: Maximum allowed factor drift before rebalance.

    """

    target_delta: float
    target_theta: float
    delta_threshold: float
    theta_threshold: float
    factor_threshold: float

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.delta_threshold <= 0:
            raise ValueError(f"delta_threshold must be positive, got {self.delta_threshold}")
        if self.theta_threshold <= 0:
            raise ValueError(f"theta_threshold must be positive, got {self.theta_threshold}")
        if self.factor_threshold <= 0:
            raise ValueError(f"factor_threshold must be positive, got {self.factor_threshold}")


class HedgingEngine:
    """Engine for computing and managing portfolio hedges.

    Provides risk decomposition in hazard space and monitors exposure
    drift to trigger banded rebalancing.
    """

    def __init__(
        self,
        model: HazardRateModel,
        config: HedgingConfig,
    ) -> None:
        """Initialize the hedging engine.

        Args:
            model: HazardRateModel for computing sensitivities.
            config: Hedging configuration.

        """
        self._model = model
        self._config = config
        self._last_exposures: RiskExposures | None = None

        # Precompute factor weights
        n = len(model.maturities)
        self._front_weights = self._compute_factor_weights(n, "front")
        self._belly_weights = self._compute_factor_weights(n, "belly")
        self._back_weights = self._compute_factor_weights(n, "back")

        logger.info(
            "HedgingEngine initialized: delta_threshold=%.4f, theta_threshold=%.4f",
            config.delta_threshold,
            config.theta_threshold,
        )

    def compute_exposures(
        self,
        positions: Sequence[PortfolioPosition],
    ) -> RiskExposures:
        """Compute current portfolio risk exposures.

        Args:
            positions: Current portfolio positions.

        Returns:
            RiskExposures with all computed metrics.

        Raises:
            ValueError: If positions don't align with model maturities.

        """
        # Build position vector aligned with model maturities
        maturities = self._model.maturities
        n = len(maturities)
        z = np.zeros(n)

        pos_map = {p.maturity: p for p in positions}
        for i, mat in enumerate(maturities):
            if mat in pos_map:
                z[i] = pos_map[mat].net_exposure

        # Get model sensitivities
        survival = self._model.survival_probabilities
        bucket_hazards = self._model.bucket_hazards

        # Per-maturity contributions
        per_mat_delta = z * survival
        per_mat_theta = z * bucket_hazards * survival

        # Aggregate exposures
        delta = float(np.sum(per_mat_delta))
        theta = float(np.sum(per_mat_theta))

        # Factor exposures
        front_end = float(np.sum(self._front_weights * per_mat_delta))
        belly = float(np.sum(self._belly_weights * per_mat_delta))
        back_end = float(np.sum(self._back_weights * per_mat_delta))

        exposures = RiskExposures(
            delta=delta,
            theta=theta,
            front_end=front_end,
            belly=belly,
            back_end=back_end,
            per_maturity_delta=per_mat_delta,
            per_maturity_theta=per_mat_theta,
        )

        self._last_exposures = exposures

        logger.info(
            "Computed exposures: delta=%.4f, theta=%.4f, front=%.4f, belly=%.4f, back=%.4f",
            delta,
            theta,
            front_end,
            belly,
            back_end,
        )

        return exposures

    def check_drift(
        self,
        current_exposures: RiskExposures,
    ) -> DriftReport:
        """Check if exposures have drifted beyond thresholds.

        Args:
            current_exposures: Current portfolio exposures.

        Returns:
            DriftReport with drift metrics and rebalance recommendation.

        """
        config = self._config

        # Compute absolute drifts from targets
        delta_drift = abs(current_exposures.delta - config.target_delta)
        theta_drift = abs(current_exposures.theta - config.target_theta)

        # Factor drifts (target is always 0 for factor neutrality)
        front_drift = abs(current_exposures.front_end)
        belly_drift = abs(current_exposures.belly)
        back_drift = abs(current_exposures.back_end)

        # Check thresholds
        triggering = []
        drift_pcts = []

        delta_pct = delta_drift / config.delta_threshold
        drift_pcts.append(delta_pct)
        if delta_pct > 1.0:
            triggering.append("delta")

        theta_pct = theta_drift / config.theta_threshold
        drift_pcts.append(theta_pct)
        if theta_pct > 1.0:
            triggering.append("theta")

        front_pct = front_drift / config.factor_threshold
        drift_pcts.append(front_pct)
        if front_pct > 1.0:
            triggering.append("front_end")

        belly_pct = belly_drift / config.factor_threshold
        drift_pcts.append(belly_pct)
        if belly_pct > 1.0:
            triggering.append("belly")

        back_pct = back_drift / config.factor_threshold
        drift_pcts.append(back_pct)
        if back_pct > 1.0:
            triggering.append("back_end")

        max_drift = max(drift_pcts)
        requires_rebal = len(triggering) > 0

        report = DriftReport(
            delta_drift=delta_drift,
            theta_drift=theta_drift,
            front_end_drift=front_drift,
            belly_drift=belly_drift,
            back_end_drift=back_drift,
            max_drift_pct=max_drift,
            requires_rebalance=requires_rebal,
            triggering_factors=triggering,
        )

        if requires_rebal:
            logger.warning(
                "Rebalance triggered by factors: %s (max drift %.1f%%)",
                triggering,
                max_drift * 100,
            )
        else:
            logger.info("No rebalance needed, max drift %.1f%%", max_drift * 100)

        return report

    def compute_hedge_adjustment(
        self,
        current_exposures: RiskExposures,
        target_maturities: Sequence[int] | None = None,
    ) -> NDArray[np.float64]:
        """Compute position adjustments to restore hedge targets.

        Uses least-squares to find minimal adjustments that neutralize
        the specified factors.

        Args:
            current_exposures: Current portfolio exposures.
            target_maturities: Indices of maturities available for hedging.
                If None, uses all maturities.

        Returns:
            Array of position adjustments (positive = buy YES, negative = buy NO).

        """
        n = len(self._model.maturities)
        survival = self._model.survival_probabilities
        bucket_hazards = self._model.bucket_hazards

        if target_maturities is None:
            target_maturities = list(range(n))

        m = len(target_maturities)

        # Build sensitivity matrix: each row is a factor, each column is a maturity
        # Factors: delta, theta, front, belly, back
        sens_matrix = np.zeros((5, m))
        for j, i in enumerate(target_maturities):
            sens_matrix[0, j] = survival[i]  # Delta
            sens_matrix[1, j] = bucket_hazards[i] * survival[i]  # Theta
            sens_matrix[2, j] = self._front_weights[i] * survival[i]  # Front
            sens_matrix[3, j] = self._belly_weights[i] * survival[i]  # Belly
            sens_matrix[4, j] = self._back_weights[i] * survival[i]  # Back

        # Target: bring each factor to its target
        target_vec = np.array(
            [
                self._config.target_delta - current_exposures.delta,
                self._config.target_theta - current_exposures.theta,
                -current_exposures.front_end,
                -current_exposures.belly,
                -current_exposures.back_end,
            ]
        )

        # Solve least-squares: sens_matrix @ dz = target_vec
        result, _, _, _ = np.linalg.lstsq(
            sens_matrix.T @ sens_matrix,
            sens_matrix.T @ target_vec,
            rcond=None,
        )

        # Map back to full maturity array
        adjustments = np.zeros(n)
        for j, i in enumerate(target_maturities):
            adjustments[i] = result[j] if j < len(result) else 0.0

        logger.info(
            "Computed hedge adjustments: sum=%.4f, max_abs=%.4f",
            np.sum(adjustments),
            np.max(np.abs(adjustments)),
        )

        return adjustments

    @staticmethod
    def _compute_factor_weights(n: int, factor: str) -> NDArray[np.float64]:
        """Compute weights for hazard factor exposure.

        Args:
            n: Number of maturities.
            factor: One of 'front', 'belly', 'back'.

        Returns:
            Normalized weight array.

        """
        x = np.linspace(0, 1, n)

        if factor == "front":
            weights = 1.0 - x
        elif factor == "belly":
            weights = np.exp(-((x - 0.5) ** 2) / 0.1)
        elif factor == "back":
            weights = x
        else:
            raise ValueError(f"Unknown factor: {factor}")

        return weights / (weights.sum() + 1e-10)
