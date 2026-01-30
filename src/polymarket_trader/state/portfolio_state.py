"""Portfolio state management for incremental optimization.

This module provides state persistence for portfolio management,
allowing the optimizer to track positions across multiple runs.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization.

    Attributes:
        target_delta: Target delta exposure (usually 0).
        target_theta: Target theta exposure (usually 0).
        delta_tolerance: Allowed deviation from target delta.
        theta_tolerance: Allowed deviation from target theta.
        use_spread: Whether to use bid/ask instead of mid prices.
        smoothness_alpha: Smoothness parameter for curve fitting.

    """

    target_delta: float
    target_theta: float
    delta_tolerance: float
    theta_tolerance: float
    use_spread: bool
    smoothness_alpha: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "target_delta": self.target_delta,
            "target_theta": self.target_theta,
            "delta_tolerance": self.delta_tolerance,
            "theta_tolerance": self.theta_tolerance,
            "use_spread": self.use_spread,
            "smoothness_alpha": self.smoothness_alpha,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioConfig":
        """Create from dictionary."""
        return cls(
            target_delta=float(data["target_delta"]),
            target_theta=float(data["target_theta"]),
            delta_tolerance=float(data["delta_tolerance"]),
            theta_tolerance=float(data["theta_tolerance"]),
            use_spread=bool(data["use_spread"]),
            smoothness_alpha=float(data["smoothness_alpha"]),
        )


@dataclass
class PortfolioState:
    """Portfolio state for incremental optimization.

    Attributes:
        event: Event identifier (URL slug or name).
        positions: Dictionary mapping contract_id to net position.
            Positive = long YES, negative = long NO.
        budget: Remaining budget for new trades.
        config: Portfolio configuration.
        last_updated: Timestamp of last update.

    """

    event: str
    positions: dict[str, float]
    budget: float
    config: PortfolioConfig
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "event": self.event,
            "positions": self.positions,
            "budget": self.budget,
            "config": self.config.to_dict(),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PortfolioState":
        """Create from dictionary."""
        return cls(
            event=str(data["event"]),
            positions={str(k): float(v) for k, v in data["positions"].items()},
            budget=float(data["budget"]),
            config=PortfolioConfig.from_dict(data["config"]),
            last_updated=str(data.get("last_updated", datetime.now(UTC).isoformat())),
        )

    @classmethod
    def create_initial(
        cls,
        event: str,
        budget: float,
        delta_tolerance: float,
        theta_tolerance: float,
        use_spread: bool = True,
        smoothness_alpha: float = 0.01,
    ) -> "PortfolioState":
        """Create initial state with no positions.

        Args:
            event: Event identifier.
            budget: Initial budget.
            delta_tolerance: Delta tolerance band.
            theta_tolerance: Theta tolerance band.
            use_spread: Whether to use bid/ask prices.
            smoothness_alpha: Curve fitting smoothness.

        Returns:
            New PortfolioState with zero positions.

        """
        config = PortfolioConfig(
            target_delta=0.0,
            target_theta=0.0,
            delta_tolerance=delta_tolerance,
            theta_tolerance=theta_tolerance,
            use_spread=use_spread,
            smoothness_alpha=smoothness_alpha,
        )
        return cls(
            event=event,
            positions={},
            budget=budget,
            config=config,
        )

    def get_position(self, contract_id: str) -> float:
        """Get position for a contract, defaulting to 0 if not present."""
        return self.positions.get(contract_id, 0.0)

    def update_positions(self, new_positions: dict[str, float]) -> None:
        """Update positions and timestamp.

        Args:
            new_positions: New positions to set.

        """
        self.positions = new_positions
        self.last_updated = datetime.now(UTC).isoformat()


def load_state(path: Path) -> PortfolioState:
    """Load portfolio state from JSON file.

    Args:
        path: Path to state file.

    Returns:
        Loaded PortfolioState.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is malformed.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in state file: {e}") from e

    try:
        state = PortfolioState.from_dict(data)
    except KeyError as e:
        raise ValueError(f"Missing required field in state file: {e}") from e

    logger.info(
        "Loaded state from %s: event=%s, positions=%d, budget=%.2f",
        path,
        state.event,
        len(state.positions),
        state.budget,
    )
    return state


def save_state(state: PortfolioState, path: Path) -> None:
    """Save portfolio state to JSON file.

    Args:
        state: PortfolioState to save.
        path: Path to save to.

    """
    path = Path(path)

    # Update timestamp before saving
    state.last_updated = datetime.now(UTC).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)

    logger.info(
        "Saved state to %s: event=%s, positions=%d, budget=%.2f",
        path,
        state.event,
        len(state.positions),
        state.budget,
    )
