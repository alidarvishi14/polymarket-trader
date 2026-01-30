"""Persistence for Bayesian hazard estimator state.

This module handles saving and loading the estimator state (posterior over
hazard rates) to/from JSON files, allowing the Bayesian model to maintain
beliefs across multiple runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from polymarket_trader.models.bayesian_hazard import (
    BayesianHazardConfig,
    BayesianHazardState,
)

logger = logging.getLogger(__name__)

# Schema version for future migrations
SCHEMA_VERSION = 1


def save_estimator_state(
    state: BayesianHazardState,
    config: BayesianHazardConfig,
    path: Path,
) -> None:
    """Save estimator state to JSON file.

    Args:
        state: Current estimator state (posterior).
        config: Estimator configuration (hyperparameters).
        path: Output file path.

    Raises:
        OSError: If file cannot be written.

    """
    data = {
        "version": SCHEMA_VERSION,
        "state": state.to_dict(),
        "config": config.to_dict(),
    }

    path = Path(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Saved estimator state to %s: n_buckets=%d, n_updates=%d",
        path,
        state.n_buckets,
        state.n_updates,
    )


def load_estimator_state(
    path: Path,
) -> tuple[BayesianHazardState, BayesianHazardConfig]:
    """Load estimator state from JSON file.

    Args:
        path: Input file path.

    Returns:
        Tuple of (state, config).

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is invalid or incompatible version.

    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Estimator state file not found: {path}")

    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in estimator state file: {e}") from e

    # Check version
    version = data.get("version")
    if version is None:
        raise ValueError("Missing 'version' field in estimator state file")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible estimator state version: {version} (expected {SCHEMA_VERSION})"
        )

    # Check required fields
    if "state" not in data:
        raise ValueError("Missing 'state' field in estimator state file")
    if "config" not in data:
        raise ValueError("Missing 'config' field in estimator state file")

    try:
        state = BayesianHazardState.from_dict(data["state"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid state data: {e}") from e

    try:
        config = BayesianHazardConfig.from_dict(data["config"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid config data: {e}") from e

    logger.info(
        "Loaded estimator state from %s: n_buckets=%d, n_updates=%d",
        path,
        state.n_buckets,
        state.n_updates,
    )

    return state, config
