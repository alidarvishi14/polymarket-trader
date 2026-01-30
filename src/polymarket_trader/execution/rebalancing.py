"""Rebalancing policy for portfolio management.

This module implements banded rebalancing that:
1. Only rebalances when exposures drift beyond tolerance bands
2. Rolls hedges rather than topping them up
3. Handles time-based roll triggers for near-expiry positions
4. Provides clear decision logic for when and how to rebalance
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.portfolio import PortfolioPosition
from polymarket_trader.risk.hedging import HedgingEngine

logger = logging.getLogger(__name__)


class RebalanceReason(Enum):
    """Reason for triggering a rebalance."""

    DELTA_DRIFT = auto()
    THETA_DRIFT = auto()
    FACTOR_DRIFT = auto()
    TIME_TO_EXPIRY = auto()
    NEWS_EVENT = auto()
    LIQUIDITY_CHANGE = auto()
    SCHEDULED = auto()


@dataclass(frozen=True)
class RollInstruction:
    """Instruction to roll a position from one maturity to another.

    Attributes:
        from_maturity: Maturity to close/reduce.
        to_maturity: Maturity to open/increase.
        quantity: Size to roll (positive = net YES exposure).
        reason: Why this roll is recommended.

    """

    from_maturity: float
    to_maturity: float
    quantity: float
    reason: str


@dataclass(frozen=True)
class RebalanceDecision:
    """Decision output from rebalancing policy.

    Attributes:
        should_rebalance: Whether to rebalance.
        reasons: List of reasons for rebalancing.
        position_adjustments: Recommended position changes by maturity.
        roll_instructions: Specific roll instructions.
        urgency: How urgent the rebalance is (0-1, 1 = immediate).
        cost_estimate: Estimated transaction cost of rebalancing.

    """

    should_rebalance: bool
    reasons: list[RebalanceReason]
    position_adjustments: dict[float, float]
    roll_instructions: list[RollInstruction]
    urgency: float
    cost_estimate: float


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing policy.

    Attributes:
        min_days_to_expiry: Minimum days before expiry to hold a position.
        roll_buffer_days: Days before min_days_to_expiry to start rolling.
        max_rebalance_frequency_hours: Minimum hours between rebalances.
        drift_band_pct: Percentage drift band before triggering rebalance.
        transaction_cost_bps: Assumed transaction cost in basis points.
        min_edge_after_costs: Minimum edge required after costs to trade.

    """

    min_days_to_expiry: float
    roll_buffer_days: float
    max_rebalance_frequency_hours: float
    drift_band_pct: float
    transaction_cost_bps: float
    min_edge_after_costs: float

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_days_to_expiry < 0:
            raise ValueError(
                f"min_days_to_expiry must be non-negative, got {self.min_days_to_expiry}"
            )
        if self.roll_buffer_days < 0:
            raise ValueError(f"roll_buffer_days must be non-negative, got {self.roll_buffer_days}")
        if self.max_rebalance_frequency_hours <= 0:
            raise ValueError(
                f"max_rebalance_frequency_hours must be positive, "
                f"got {self.max_rebalance_frequency_hours}"
            )
        if not 0 < self.drift_band_pct < 1:
            raise ValueError(f"drift_band_pct must be in (0, 1), got {self.drift_band_pct}")


class RebalancingPolicy:
    """Policy engine for determining when and how to rebalance.

    Implements banded rebalancing that avoids over-trading while
    maintaining risk limits.
    """

    def __init__(
        self,
        model: HazardRateModel,
        hedging_engine: HedgingEngine,
        config: RebalanceConfig,
    ) -> None:
        """Initialize the rebalancing policy.

        Args:
            model: HazardRateModel for pricing.
            hedging_engine: HedgingEngine for exposure calculations.
            config: Rebalancing configuration.

        """
        self._model = model
        self._hedging = hedging_engine
        self._config = config
        self._last_rebalance_time: datetime | None = None

        logger.info(
            "RebalancingPolicy initialized: min_dte=%.1f, roll_buffer=%.1f, drift_band=%.1f%%",
            config.min_days_to_expiry,
            config.roll_buffer_days,
            config.drift_band_pct * 100,
        )

    def evaluate(
        self,
        positions: Sequence[PortfolioPosition],
        current_time: datetime,
        days_to_expiry: NDArray[np.float64],
        market_prices: NDArray[np.float64],
    ) -> RebalanceDecision:
        """Evaluate whether rebalancing is needed.

        Args:
            positions: Current portfolio positions.
            current_time: Current timestamp.
            days_to_expiry: Days to expiry for each maturity.
            market_prices: Current market prices.

        Returns:
            RebalanceDecision with recommendation and details.

        """
        reasons: list[RebalanceReason] = []
        adjustments: dict[float, float] = {}
        rolls: list[RollInstruction] = []
        urgency = 0.0

        # Check minimum time since last rebalance
        if self._last_rebalance_time is not None:
            hours_since = (current_time - self._last_rebalance_time).total_seconds() / 3600
            if hours_since < self._config.max_rebalance_frequency_hours:
                logger.info(
                    "Skipping rebalance: %.1f hours since last (min %.1f)",
                    hours_since,
                    self._config.max_rebalance_frequency_hours,
                )
                return RebalanceDecision(
                    should_rebalance=False,
                    reasons=[],
                    position_adjustments={},
                    roll_instructions=[],
                    urgency=0.0,
                    cost_estimate=0.0,
                )

        # Compute current exposures
        exposures = self._hedging.compute_exposures(positions)

        # Check drift
        drift_report = self._hedging.check_drift(exposures)
        if drift_report.requires_rebalance:
            if "delta" in drift_report.triggering_factors:
                reasons.append(RebalanceReason.DELTA_DRIFT)
            if "theta" in drift_report.triggering_factors:
                reasons.append(RebalanceReason.THETA_DRIFT)
            if any(
                f in drift_report.triggering_factors for f in ["front_end", "belly", "back_end"]
            ):
                reasons.append(RebalanceReason.FACTOR_DRIFT)

            urgency = max(urgency, min(drift_report.max_drift_pct, 1.0))

            # Get hedge adjustments
            adj = self._hedging.compute_hedge_adjustment(exposures)
            for i, mat in enumerate(self._model.maturities):
                if abs(adj[i]) > 1e-6:
                    adjustments[float(mat)] = float(adj[i])

        # Check time-to-expiry triggers
        maturities = self._model.maturities
        roll_threshold = self._config.min_days_to_expiry + self._config.roll_buffer_days

        for pos in positions:
            mat_idx = np.searchsorted(maturities, pos.maturity)
            if mat_idx >= len(days_to_expiry):
                continue

            dte = days_to_expiry[mat_idx]
            if dte <= roll_threshold and abs(pos.net_exposure) > 1e-6:
                reasons.append(RebalanceReason.TIME_TO_EXPIRY)

                # Find next maturity to roll into
                if mat_idx + 1 < len(maturities):
                    to_mat = float(maturities[mat_idx + 1])
                    rolls.append(
                        RollInstruction(
                            from_maturity=pos.maturity,
                            to_maturity=to_mat,
                            quantity=pos.net_exposure,
                            reason=f"DTE {dte:.1f} below threshold {roll_threshold:.1f}",
                        )
                    )

                    # Increase urgency as we approach hard expiry
                    dte_urgency = 1.0 - (dte / roll_threshold) if roll_threshold > 0 else 1.0
                    urgency = max(urgency, dte_urgency)

        # Estimate transaction costs
        total_turnover = sum(abs(v) for v in adjustments.values())
        for roll in rolls:
            total_turnover += 2 * abs(roll.quantity)  # Close + open

        cost_estimate = total_turnover * self._config.transaction_cost_bps / 10000

        should_rebal = len(reasons) > 0

        if should_rebal:
            logger.info(
                "Rebalance recommended: reasons=%s, urgency=%.2f, cost_est=%.4f",
                [r.name for r in reasons],
                urgency,
                cost_estimate,
            )

        return RebalanceDecision(
            should_rebalance=should_rebal,
            reasons=reasons,
            position_adjustments=adjustments,
            roll_instructions=rolls,
            urgency=urgency,
            cost_estimate=cost_estimate,
        )

    def record_rebalance(self, time: datetime) -> None:
        """Record that a rebalance was executed.

        Args:
            time: Time of rebalance execution.

        """
        self._last_rebalance_time = time
        logger.info("Recorded rebalance at %s", time.isoformat())

    def compute_optimal_roll_maturity(
        self,
        from_maturity: float,
        available_maturities: Sequence[float],
        market_prices: NDArray[np.float64],
        days_to_expiry: NDArray[np.float64],
    ) -> float | None:
        """Find the optimal maturity to roll into.

        Considers:
        - Sufficient time to expiry
        - Liquidity (via spread)
        - Curve steepness (carry)

        Args:
            from_maturity: Maturity being rolled from.
            available_maturities: Candidate maturities.
            market_prices: Market prices for candidates.
            days_to_expiry: Days to expiry for candidates.

        Returns:
            Optimal maturity to roll into, or None if no suitable maturity.

        """
        min_dte = self._config.min_days_to_expiry + self._config.roll_buffer_days
        candidates = []

        for i, mat in enumerate(available_maturities):
            if mat <= from_maturity:
                continue
            if days_to_expiry[i] <= min_dte:
                continue

            # Score by DTE (prefer longer) and implied hazard rate
            dte_score = days_to_expiry[i] / 365.0
            candidates.append((mat, dte_score))

        if not candidates:
            logger.warning("No suitable roll maturity found from %.2f", from_maturity)
            return None

        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0][0]

        logger.info("Optimal roll maturity: %.2f -> %.2f", from_maturity, best)
        return best
