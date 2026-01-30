"""Exit rule engine for position management.

This module implements explicit exit criteria to avoid over-trading:
1. Convergence rule: close when mispricing falls below threshold
2. PnL realization rule: close after capturing target edge percentage
3. Time-to-expiry rule: close or roll positions inside minimum DTE

Without explicit exit rules, relative-value strategies systematically overtrade noise.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.portfolio import PortfolioPosition

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reason for exiting a position."""

    CONVERGENCE = auto()  # Mispricing fell below threshold
    PNL_TARGET = auto()  # Captured target percentage of edge
    TIME_TO_EXPIRY = auto()  # Position too close to expiry
    STOP_LOSS = auto()  # Loss exceeded threshold
    MODEL_CHANGE = auto()  # Model no longer supports position
    MANUAL = auto()  # Manual override


@dataclass(frozen=True)
class ExitSignal:
    """Signal to exit a position.

    Attributes:
        maturity: Maturity of position to exit.
        contract_id: Contract identifier.
        reason: Why exit is recommended.
        urgency: How urgent the exit is (0-1, 1 = immediate).
        quantity_to_close: Quantity to close (positive for closing longs).
        realized_pnl_estimate: Estimated PnL from closing.
        remaining_edge: Remaining edge if position is held.

    """

    maturity: float
    contract_id: str
    reason: ExitReason
    urgency: float
    quantity_to_close: float
    realized_pnl_estimate: float
    remaining_edge: float


@dataclass(frozen=True)
class ExitEvaluation:
    """Result of exit rule evaluation.

    Attributes:
        signals: List of exit signals for positions requiring action.
        total_positions_evaluated: Number of positions checked.
        positions_flagged: Number of positions flagged for exit.
        total_pnl_at_risk: Total PnL at risk from flagged positions.

    """

    signals: list[ExitSignal]
    total_positions_evaluated: int
    positions_flagged: int
    total_pnl_at_risk: float


@dataclass
class ExitConfig:
    """Configuration for exit rules.

    Attributes:
        convergence_threshold: Minimum mispricing to maintain position.
        pnl_capture_target: Target percentage of initial edge to capture (0-1).
        min_days_to_expiry: Minimum DTE before forced exit.
        stop_loss_pct: Maximum loss as percentage of initial cost before exit.
        model_stability_threshold: Minimum edge stability across alphas.

    """

    convergence_threshold: float
    pnl_capture_target: float
    min_days_to_expiry: float
    stop_loss_pct: float
    model_stability_threshold: float

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.convergence_threshold < 0:
            raise ValueError(
                f"convergence_threshold must be non-negative, got {self.convergence_threshold}"
            )
        if not 0 < self.pnl_capture_target <= 1:
            raise ValueError(f"pnl_capture_target must be in (0, 1], got {self.pnl_capture_target}")
        if self.min_days_to_expiry < 0:
            raise ValueError(
                f"min_days_to_expiry must be non-negative, got {self.min_days_to_expiry}"
            )
        if self.stop_loss_pct <= 0:
            raise ValueError(f"stop_loss_pct must be positive, got {self.stop_loss_pct}")


@dataclass
class PositionState:
    """Tracked state for a position.

    Attributes:
        maturity: Position maturity.
        contract_id: Contract identifier.
        entry_price: Price at entry.
        entry_theoretical: Theoretical price at entry.
        initial_edge: Initial expected edge.
        cost_basis: Total cost to acquire.
        quantity: Position size (positive = YES, negative = NO).

    """

    maturity: float
    contract_id: str
    entry_price: float
    entry_theoretical: float
    initial_edge: float
    cost_basis: float
    quantity: float


class ExitRuleEngine:
    """Engine for evaluating exit rules and generating signals.

    Tracks position states and evaluates multiple exit criteria to
    determine when positions should be closed.
    """

    def __init__(
        self,
        config: ExitConfig,
    ) -> None:
        """Initialize the exit rule engine.

        Args:
            config: Exit rule configuration.

        """
        self._config = config
        self._position_states: dict[str, PositionState] = {}

        logger.info(
            "ExitRuleEngine initialized: convergence=%.4f, pnl_target=%.1f%%, min_dte=%.1f",
            config.convergence_threshold,
            config.pnl_capture_target * 100,
            config.min_days_to_expiry,
        )

    def register_position(
        self,
        position: PortfolioPosition,
        entry_price: float,
        entry_theoretical: float,
    ) -> None:
        """Register a new position for tracking.

        Args:
            position: Portfolio position to track.
            entry_price: Market price at entry.
            entry_theoretical: Model theoretical price at entry.

        """
        initial_edge = position.edge
        state = PositionState(
            maturity=position.maturity,
            contract_id=position.contract_id,
            entry_price=entry_price,
            entry_theoretical=entry_theoretical,
            initial_edge=initial_edge,
            cost_basis=position.cost_basis,
            quantity=position.net_exposure,
        )
        self._position_states[position.contract_id] = state

        logger.info(
            "Registered position: %s, maturity=%.2f, initial_edge=%.4f",
            position.contract_id,
            position.maturity,
            initial_edge,
        )

    def unregister_position(self, contract_id: str) -> None:
        """Remove a position from tracking.

        Args:
            contract_id: Contract identifier to remove.

        """
        if contract_id in self._position_states:
            del self._position_states[contract_id]
            logger.info("Unregistered position: %s", contract_id)

    def evaluate(
        self,
        model: HazardRateModel,
        current_positions: Sequence[PortfolioPosition],
        market_prices: NDArray[np.float64],
        days_to_expiry: NDArray[np.float64],
    ) -> ExitEvaluation:
        """Evaluate exit rules for all tracked positions.

        Args:
            model: Current fitted HazardRateModel.
            current_positions: Current portfolio positions.
            market_prices: Current market prices (mid or bid).
            days_to_expiry: Days to expiry for each maturity.

        Returns:
            ExitEvaluation with signals for positions requiring action.

        """
        signals: list[ExitSignal] = []
        maturities = model.maturities
        theo_prices = model.theoretical_prices

        for pos in current_positions:
            if pos.contract_id not in self._position_states:
                continue

            state = self._position_states[pos.contract_id]

            # Find maturity index
            mat_idx = int(np.searchsorted(maturities, pos.maturity))
            if mat_idx >= len(maturities):
                continue

            current_mkt = market_prices[mat_idx]
            current_theo = theo_prices[mat_idx]
            dte = days_to_expiry[mat_idx]

            # Compute current mispricing
            current_mispricing = abs(current_theo - current_mkt)

            # Compute realized PnL (mark-to-market)
            if state.quantity > 0:  # Long YES
                realized_pnl = state.quantity * (current_mkt - state.entry_price)
            else:  # Short YES (long NO)
                realized_pnl = abs(state.quantity) * (state.entry_price - current_mkt)

            # Compute edge captured
            if abs(state.initial_edge) > 1e-10:
                edge_captured_pct = realized_pnl / state.initial_edge
            else:
                edge_captured_pct = 0.0

            # Compute remaining edge
            remaining_edge = pos.edge

            # Check exit rules
            signal = self._check_exit_rules(
                pos=pos,
                state=state,
                current_mispricing=current_mispricing,
                realized_pnl=realized_pnl,
                edge_captured_pct=edge_captured_pct,
                dte=dte,
                remaining_edge=remaining_edge,
            )

            if signal is not None:
                signals.append(signal)

        total_pnl_at_risk = sum(s.remaining_edge for s in signals)

        evaluation = ExitEvaluation(
            signals=signals,
            total_positions_evaluated=len(current_positions),
            positions_flagged=len(signals),
            total_pnl_at_risk=total_pnl_at_risk,
        )

        if signals:
            logger.info(
                "Exit evaluation: %d/%d positions flagged, total PnL at risk: %.4f",
                len(signals),
                len(current_positions),
                total_pnl_at_risk,
            )

        return evaluation

    def _check_exit_rules(
        self,
        pos: PortfolioPosition,
        state: PositionState,
        current_mispricing: float,
        realized_pnl: float,
        edge_captured_pct: float,
        dte: float,
        remaining_edge: float,
    ) -> ExitSignal | None:
        """Check all exit rules for a position.

        Args:
            pos: Current position.
            state: Tracked position state.
            current_mispricing: Current absolute mispricing.
            realized_pnl: Mark-to-market PnL.
            edge_captured_pct: Percentage of initial edge captured.
            dte: Days to expiry.
            remaining_edge: Remaining theoretical edge.

        Returns:
            ExitSignal if exit is recommended, None otherwise.

        """
        config = self._config

        # Rule 1: Convergence
        if current_mispricing < config.convergence_threshold:
            logger.info(
                "Convergence exit for %s: mispricing %.4f < threshold %.4f",
                pos.contract_id,
                current_mispricing,
                config.convergence_threshold,
            )
            return ExitSignal(
                maturity=pos.maturity,
                contract_id=pos.contract_id,
                reason=ExitReason.CONVERGENCE,
                urgency=0.5,
                quantity_to_close=pos.net_exposure,
                realized_pnl_estimate=realized_pnl,
                remaining_edge=remaining_edge,
            )

        # Rule 2: PnL target capture
        if edge_captured_pct >= config.pnl_capture_target:
            logger.info(
                "PnL target exit for %s: captured %.1f%% >= target %.1f%%",
                pos.contract_id,
                edge_captured_pct * 100,
                config.pnl_capture_target * 100,
            )
            return ExitSignal(
                maturity=pos.maturity,
                contract_id=pos.contract_id,
                reason=ExitReason.PNL_TARGET,
                urgency=0.7,
                quantity_to_close=pos.net_exposure,
                realized_pnl_estimate=realized_pnl,
                remaining_edge=remaining_edge,
            )

        # Rule 3: Time to expiry
        if dte <= config.min_days_to_expiry:
            # Urgency increases as we approach expiry
            if config.min_days_to_expiry > 0:
                urgency = 1.0 - (dte / config.min_days_to_expiry)
            else:
                urgency = 1.0
            logger.info(
                "Time exit for %s: DTE %.1f <= threshold %.1f",
                pos.contract_id,
                dte,
                config.min_days_to_expiry,
            )
            return ExitSignal(
                maturity=pos.maturity,
                contract_id=pos.contract_id,
                reason=ExitReason.TIME_TO_EXPIRY,
                urgency=min(urgency, 1.0),
                quantity_to_close=pos.net_exposure,
                realized_pnl_estimate=realized_pnl,
                remaining_edge=remaining_edge,
            )

        # Rule 4: Stop loss
        if state.cost_basis > 0:
            loss_pct = -realized_pnl / state.cost_basis
            if loss_pct >= config.stop_loss_pct:
                logger.warning(
                    "Stop loss exit for %s: loss %.1f%% >= threshold %.1f%%",
                    pos.contract_id,
                    loss_pct * 100,
                    config.stop_loss_pct * 100,
                )
                return ExitSignal(
                    maturity=pos.maturity,
                    contract_id=pos.contract_id,
                    reason=ExitReason.STOP_LOSS,
                    urgency=0.9,
                    quantity_to_close=pos.net_exposure,
                    realized_pnl_estimate=realized_pnl,
                    remaining_edge=remaining_edge,
                )

        return None

    def check_model_stability(
        self,
        position: PortfolioPosition,
        model_results: dict[float, float],
    ) -> ExitSignal | None:
        """Check if position edge is stable across model variations.

        Args:
            position: Position to check.
            model_results: Dictionary mapping alpha values to resulting edge.

        Returns:
            ExitSignal if position is unstable, None otherwise.

        """
        if len(model_results) < 2:
            return None

        edges = list(model_results.values())
        edge_range = max(edges) - min(edges)
        mean_edge = np.mean(edges)

        stability = 1.0 - edge_range / abs(mean_edge) if mean_edge != 0 else 0.0

        if stability < self._config.model_stability_threshold:
            logger.warning(
                "Model instability exit for %s: stability %.2f < threshold %.2f",
                position.contract_id,
                stability,
                self._config.model_stability_threshold,
            )
            return ExitSignal(
                maturity=position.maturity,
                contract_id=position.contract_id,
                reason=ExitReason.MODEL_CHANGE,
                urgency=0.6,
                quantity_to_close=position.net_exposure,
                realized_pnl_estimate=0.0,  # Unknown without current prices
                remaining_edge=mean_edge,
            )

        return None
