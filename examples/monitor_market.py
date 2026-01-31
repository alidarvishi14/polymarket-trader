#!/usr/bin/env python3
"""Real-time market monitor using Bayesian hazard estimator.

This script:
1. Fetches market data at configurable intervals (default 30s)
2. Updates Bayesian beliefs about hazard rates
3. Outputs fitted curve and trade recommendations
4. Assumes clean portfolio with $1000 budget for each recommendation

Usage:
    # Run once
    python examples/monitor_market.py --event us-strikes-iran-by

    # Run continuously with config file
    python examples/monitor_market.py --config monitor.yaml --loop

    # CLI args override config file
    python examples/monitor_market.py --config monitor.yaml --interval 60 --loop

Config file format (YAML):
    event: us-strikes-iran-by
    estimator_state: estimator.json
    budget: 5000
    interval: 30
    rho: 0.85
    live: true
    fee: 0.5
"""

import argparse
import logging
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from polymarket_trader.data import (
    AuthenticatedPolymarketClient,
    PolymarketClient,
    transform_to_maturity_data,
)
from polymarket_trader.data.transformer import EASTERN_TZ
from polymarket_trader.models.bayesian_hazard import (
    BayesianHazardConfig,
    BayesianHazardEstimator,
)
from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.optimization.simple_portfolio import optimize_simple_portfolio
from polymarket_trader.state.estimator_state import (
    load_estimator_state,
    save_estimator_state,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Global flag for graceful shutdown
shutdown_requested = False


@dataclass
class MonitorConfig:
    """Configuration for the market monitor.

    Priority: CLI args > config file > defaults.

    Attributes:
        event: Polymarket event slug or URL.
        estimator_state: Path to estimator state file.
        budget: Budget for trade recommendations.
        interval: Update interval in seconds.
        loop: Run continuously.
        rho: Correlation decay parameter.
        live: Fetch live portfolio positions.
        fee: Trading fee per contract in cents.

    """

    event: str | None = None
    estimator_state: Path | None = None
    budget: float = 1000.0
    interval: int = 30
    loop: bool = False
    rho: float = 0.85
    live: bool = False
    fee: float = 0.0

    @classmethod
    def from_yaml(cls, path: Path) -> "MonitorConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            MonitorConfig with values from file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If config file is invalid.

        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Convert estimator_state to Path if present
        if "estimator_state" in data and data["estimator_state"]:
            data["estimator_state"] = Path(data["estimator_state"])

        return cls(**data)

    def merge_cli_args(self, args: argparse.Namespace) -> "MonitorConfig":
        """Merge CLI arguments, CLI takes priority over config file.

        Args:
            args: Parsed CLI arguments.

        Returns:
            New MonitorConfig with CLI overrides applied.

        """
        # Start with current values (from config file or defaults)
        merged = MonitorConfig(
            event=self.event,
            estimator_state=self.estimator_state,
            budget=self.budget,
            interval=self.interval,
            loop=self.loop,
            rho=self.rho,
            live=self.live,
            fee=self.fee,
        )

        # Override with CLI args if explicitly provided
        # We check if the arg was explicitly passed by comparing to argparse defaults
        if args.event is not None:
            merged.event = args.event
        if args.estimator_state is not None:
            merged.estimator_state = args.estimator_state
        if args.budget != 1000.0:  # Default value
            merged.budget = args.budget
        if args.interval != 30:  # Default value
            merged.interval = args.interval
        if args.loop:  # Boolean flag, True if passed
            merged.loop = True
        if args.rho != 0.85:  # Default value
            merged.rho = args.rho
        if args.live:  # Boolean flag, True if passed
            merged.live = True
        if args.fee != 0.0:  # Default value
            merged.fee = args.fee

        return merged

    def validate(self) -> None:
        """Validate config values.

        Raises:
            ValueError: If required fields are missing or invalid.

        """
        if not self.event:
            raise ValueError("Event is required. Pass --event or set 'event' in config file.")
        if self.budget <= 0:
            raise ValueError(f"Budget must be positive, got {self.budget}")
        if self.interval <= 0:
            raise ValueError(f"Interval must be positive, got {self.interval}")
        if not 0 < self.rho < 1:
            raise ValueError(f"Rho must be in (0, 1), got {self.rho}")
        if self.fee < 0:
            raise ValueError(f"Fee must be non-negative, got {self.fee}")


def signal_handler(signum: int, frame: object) -> None:
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    shutdown_requested = True
    print("\n\nShutdown requested. Finishing current iteration...")


def print_header(event: str, n_updates: int) -> None:
    """Print report header."""
    now = datetime.now(EASTERN_TZ)
    print("\n" + "=" * 90)
    print(f"MARKET MONITOR - {event}")
    print(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')} | Updates: {n_updates}")
    print("=" * 90)


def print_hazard_curve(
    market_data: list[MaturityData],
    estimator: BayesianHazardEstimator,
    positions: dict[str, tuple[float, float]] | None = None,
) -> None:
    """Print the fitted hazard curve table.

    Args:
        market_data: List of maturity data.
        estimator: Bayesian hazard estimator.
        positions: Optional dict of contract_id -> (yes_size, no_size).

    """
    model = estimator.get_model()
    market_prices = np.array([d.market_price for d in market_data])
    model_prices = model.theoretical_prices
    mispricings = model.mispricings(market_prices)
    survival_probs = model.survival_probabilities
    bucket_hazards = model.bucket_hazards

    has_positions = positions is not None and any(
        yes > 0.01 or no > 0.01 for yes, no in positions.values()
    )

    print("\n" + "-" * 100)
    print("FITTED HAZARD CURVE (Bayesian)")
    print("-" * 100)

    if has_positions:
        print(
            f"{'Contract':<10} {'DTE':>4} {'Bid':>7} {'Mkt':>7} {'Ask':>7} "
            f"{'Model':>7} {'Misp':>8} {'YES Pos':>10} {'NO Pos':>10}"
        )
    else:
        print(
            f"{'Contract':<12} {'DTE':>5} {'Bid':>8} {'Market':>8} {'Ask':>8} "
            f"{'Model':>8} {'Mispricing':>10}"
        )
    print("-" * 100)

    # Track portfolio totals
    total_delta = 0.0
    total_theta = 0.0

    for i, d in enumerate(market_data):
        yes_pos, no_pos = positions.get(d.contract_id, (0.0, 0.0)) if positions else (0.0, 0.0)

        # Net position for delta/theta: YES is positive, NO is negative
        net_pos = yes_pos - no_pos

        # Calculate delta and theta for this position
        if abs(net_pos) > 0.01:
            # Delta = net_position * survival_probability
            pos_delta = net_pos * survival_probs[i]
            # Theta = -net_position * bucket_hazard * survival_probability
            pos_theta = -net_pos * bucket_hazards[i] * survival_probs[i]
            total_delta += pos_delta
            total_theta += pos_theta

        if has_positions:
            # Format positions
            yes_str = f"{yes_pos:>9.0f}" if yes_pos > 0.01 else "-".rjust(9)
            no_str = f"{no_pos:>9.0f}" if no_pos > 0.01 else "-".rjust(9)
            print(
                f"{d.contract_id:<10} "
                f"{d.maturity:>4.0f} "
                f"{d.bid_price:>7.2%} "
                f"{d.market_price:>7.2%} "
                f"{d.ask_price:>7.2%} "
                f"{model_prices[i]:>7.2%} "
                f"{mispricings[i]:>+8.2%} "
                f"{yes_str} "
                f"{no_str}"
            )
        else:
            print(
                f"{d.contract_id:<12} "
                f"{d.maturity:>5.0f} "
                f"{d.bid_price:>8.2%} "
                f"{d.market_price:>8.2%} "
                f"{d.ask_price:>8.2%} "
                f"{model_prices[i]:>8.2%} "
                f"{mispricings[i]:>+10.2%}"
            )

    # Print portfolio summary if has positions
    if has_positions:
        print("-" * 100)
        print(f"  PORTFOLIO:  Delta = {total_delta:+.2f}  |  Theta = {total_theta:+.4f}")


def print_trade_recommendations(
    market_data: list[MaturityData],
    estimator: BayesianHazardEstimator,
    budget: float,
    positions: dict[str, tuple[float, float]] | None = None,
    trading_fee_cents: float = 0.0,
) -> None:
    """Print trade recommendations.

    Args:
        market_data: List of maturity data.
        estimator: Bayesian hazard estimator.
        budget: Available budget.
        positions: Optional dict of contract_id -> (yes_size, no_size).
                  If provided, uses incremental optimization from current portfolio.
        trading_fee_cents: Trading fee per contract in cents.

    """
    model = estimator.get_model()
    market_prices = np.array([d.market_price for d in market_data])
    bid_prices = np.array([d.bid_price for d in market_data])
    ask_prices = np.array([d.ask_price for d in market_data])

    # Build current positions array if provided (net: YES - NO)
    has_positions = positions is not None and any(
        yes > 0.01 or no > 0.01 for yes, no in positions.values()
    )
    current_positions = None
    if has_positions:
        current_positions = np.array(
            [
                positions.get(d.contract_id, (0.0, 0.0))[0]
                - positions.get(d.contract_id, (0.0, 0.0))[1]
                for d in market_data
            ]
        )

    # Run optimizer
    # Delta: symmetric band around 0 (want to be market-neutral)
    # Theta: one-sided constraint (theta >= -2), positive theta is fine
    result = optimize_simple_portfolio(
        model=model,
        market_prices=market_prices,
        budget=budget,
        min_theta=-2.0,  # Only floor on theta, no ceiling
        bid_prices=bid_prices,
        ask_prices=ask_prices,
        use_spread=True,
        target_delta=0.0,
        delta_tolerance=10.0 if has_positions else 2.0,
        current_positions=current_positions,
        trading_fee_cents=trading_fee_cents,
    )

    # Compute survival probs for delta/theta
    survival_probs = model.survival_probabilities
    theta_weights = model.bucket_hazards * survival_probs
    model_prices = model.theoretical_prices

    print("\n" + "-" * 122)
    if has_positions:
        print(f"RECOMMENDED TRADES (live portfolio, ${budget:.2f} budget)")
    else:
        print(f"RECOMMENDED TRADES (clean portfolio, ${budget:.2f} budget)")
    print("-" * 122)
    print(
        f"{'Contract':<12} {'Action':<10} {'Qty':>8} {'Price':>8} {'Theo':>8} "
        f"{'Net Pos':>10} {'Delta':>10} {'Theta':>10} {'Edge':>12}"
    )
    print("-" * 122)

    # Convert fee from cents to dollars for edge calculation
    fee = trading_fee_cents / 100.0

    has_trades = False
    for i, d in enumerate(market_data):
        pos = result.positions[i]
        net_pos = pos.net_exposure
        delta_contrib = net_pos * survival_probs[i]
        theta_contrib = -net_pos * theta_weights[i]

        # Show trades (buy_yes, sell_yes, buy_no, sell_no)
        # Edge = (theo - price - fee) * qty for buys
        # Edge = (price - theo - fee) * qty for sells
        if pos.buy_yes > 0.01:
            edge = pos.buy_yes * (model_prices[i] - d.ask_price - fee)
            print(
                f"{d.contract_id:<12} "
                f"{'BUY YES':<10} "
                f"{pos.buy_yes:>8.2f} "
                f"{d.ask_price:>8.2%} "
                f"{model_prices[i]:>8.2%} "
                f"{net_pos:>+10.2f} "
                f"{delta_contrib:>+10.2f} "
                f"{theta_contrib:>+10.4f} "
                f"${edge:>+10.4f}"
            )
            has_trades = True
        if pos.sell_yes > 0.01:
            edge = pos.sell_yes * (d.bid_price - model_prices[i] - fee)
            print(
                f"{d.contract_id:<12} "
                f"{'SELL YES':<10} "
                f"{pos.sell_yes:>8.2f} "
                f"{d.bid_price:>8.2%} "
                f"{model_prices[i]:>8.2%} "
                f"{net_pos:>+10.2f} "
                f"{delta_contrib:>+10.2f} "
                f"{theta_contrib:>+10.4f} "
                f"${edge:>+10.4f}"
            )
            has_trades = True
        if pos.buy_no > 0.01:
            edge = pos.buy_no * (d.bid_price - model_prices[i] - fee)
            theo_price = 1.0 - model_prices[i]
            print(
                f"{d.contract_id:<12} "
                f"{'BUY NO':<10} "
                f"{pos.buy_no:>8.2f} "
                f"{1 - d.bid_price:>8.2%} "
                f"{theo_price:>8.2%} "
                f"{net_pos:>+10.2f} "
                f"{delta_contrib:>+10.2f} "
                f"{theta_contrib:>+10.4f} "
                f"${edge:>+10.4f}"
            )
            has_trades = True
        if pos.sell_no > 0.01:
            edge = pos.sell_no * (model_prices[i] - d.ask_price - fee)
            theo_price = 1.0 - model_prices[i]
            print(
                f"{d.contract_id:<12} "
                f"{'SELL NO':<10} "
                f"{pos.sell_no:>8.2f} "
                f"{1 - d.ask_price:>8.2%} "
                f"{theo_price:>8.2%} "
                f"{net_pos:>+10.2f} "
                f"{delta_contrib:>+10.2f} "
                f"{theta_contrib:>+10.4f} "
                f"${edge:>+10.4f}"
            )
            has_trades = True

    if not has_trades:
        print("  No trades recommended.")

    print("-" * 122)
    print(f"  Net cost:       ${result.total_cost:>10.2f}")
    print(f"  Expected edge:  ${result.total_edge:>10.4f}")
    print(f"  Final delta:    {result.delta_exposure:>+10.2f}")
    print(f"  Final theta:    {result.theta_exposure:>+10.4f}")


def print_estimator_status(estimator: BayesianHazardEstimator) -> None:
    """Print brief estimator status."""
    state = estimator.state
    print(f"\n  Estimator: n_updates={state.n_updates}, trace(Σ)={np.trace(state.sigma):.4f}")


def fetch_with_retry(
    event: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list | None:
    """Fetch market data with retry logic for transient errors.

    Args:
        event: Event slug to fetch.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        List of markets or None if all retries failed.

    """
    import requests

    for attempt in range(max_retries):
        try:
            with PolymarketClient() as client:
                return client.fetch_event(event)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            if attempt < max_retries - 1:
                print(
                    f"  Connection error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}"
                )
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"  All {max_retries} attempts failed: {e}")
                return None
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None

    return None


def run_single_update(
    event: str,
    estimator: BayesianHazardEstimator,
    budget: float,
    fetch_positions_fn: Callable | None = None,
    trading_fee_cents: float = 0.0,
) -> bool:
    """Run a single update cycle.

    Args:
        event: Event slug.
        estimator: Bayesian hazard estimator.
        budget: Budget for trade recommendations.
        fetch_positions_fn: Optional function(market_data) -> positions dict.
        trading_fee_cents: Trading fee per contract in cents.

    Returns:
        True if successful, False if failed.

    """
    try:
        # Fetch market data with retry
        markets = fetch_with_retry(event)

        if not markets:
            print("  Failed to fetch market data")
            return False

        now_et = datetime.now(EASTERN_TZ)
        market_data = transform_to_maturity_data(markets, reference_time=now_et)

        if len(market_data) < 2:
            print(f"ERROR: Need at least 2 contracts, found {len(market_data)}")
            return False

        # Fetch positions if function provided (only needs market_data now since
        # token_ids are stored directly in MaturityData)
        positions = None
        if fetch_positions_fn:
            positions = fetch_positions_fn(market_data)

        # Update estimator
        if not estimator.is_initialized:
            estimator.initialize(market_data, now_et)
        else:
            estimator.update(market_data, now_et)

        # Print report
        print_header(event, estimator.state.n_updates)
        print_hazard_curve(market_data, estimator, positions)
        print_trade_recommendations(market_data, estimator, budget, positions, trading_fee_cents)
        print_estimator_status(estimator)

        return True

    except Exception as e:
        logger.error("Update failed: %s", e, exc_info=True)
        print(f"\nERROR: {e}")
        return False


def main() -> int:
    """Run the market monitor CLI."""
    parser = argparse.ArgumentParser(
        description="Real-time market monitor using Bayesian hazard estimator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file example (YAML):
  event: us-strikes-iran-by
  estimator_state: estimator.json
  budget: 5000
  interval: 30
  rho: 0.85
  live: true
  fee: 0.5

Priority: CLI args > config file > defaults
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--event",
        type=str,
        help="Polymarket event slug or URL",
    )
    parser.add_argument(
        "--estimator-state",
        type=Path,
        help="Path to estimator state file (will be created if doesn't exist)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1000.0,
        help="Budget for trade recommendations (default: 1000)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously (otherwise run once)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.85,
        help="Correlation decay parameter (default: 0.85)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Fetch live portfolio positions from Polymarket API (requires .env)",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.0,
        help="Trading fee per contract in cents (default: 0.0)",
    )

    args = parser.parse_args()

    # Load config: defaults -> config file -> CLI args
    if args.config:
        try:
            cfg = MonitorConfig.from_yaml(args.config)
            print(f"Loaded config from: {args.config}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Error loading config: {e}")
            return 1
    else:
        cfg = MonitorConfig()

    # Merge CLI args (CLI takes priority)
    cfg = cfg.merge_cli_args(args)

    # Validate final config
    try:
        cfg.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1

    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize or load estimator
    if cfg.estimator_state and cfg.estimator_state.exists():
        state, bayesian_config = load_estimator_state(cfg.estimator_state)
        estimator = BayesianHazardEstimator(bayesian_config)
        estimator.state = state
        print(f"Loaded estimator state: n_updates={state.n_updates}")
    else:
        bayesian_config = BayesianHazardConfig(
            rho=cfg.rho,
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.01,
            min_variance=1e-4,
            isolation_penalty=2.0,
        )
        estimator = BayesianHazardEstimator(bayesian_config)
        print(f"Initialized new estimator: rho={cfg.rho}")

    # Initialize live portfolio if requested
    auth_client: AuthenticatedPolymarketClient | None = None

    if cfg.live:
        try:
            auth_client = AuthenticatedPolymarketClient.from_env()
            print(
                f"Live mode enabled: wallet={auth_client.address[:6]}...{auth_client.address[-4:]}"
            )
        except Exception as e:
            logger.warning("Failed to initialize live mode: %s", e)
            print(f"Warning: Could not initialize live mode: {e}")
            print("Continuing without portfolio data...")

    def fetch_positions_mapped(
        market_data: list[MaturityData],
    ) -> dict[str, tuple[float, float]] | None:
        """Fetch current positions and map to contract_ids.

        Args:
            market_data: Transformed maturity data (with contract_ids and token_ids).

        Returns:
            Dict mapping contract_id -> (yes_size, no_size) tuple.
            User can hold both YES and NO tokens separately.

        """
        if not auth_client:
            return None

        try:
            # Build token→(contract_id, outcome) mapping directly from MaturityData
            # Now MaturityData stores yes_token_id and no_token_id - no price matching needed!
            token_to_contract: dict[str, tuple[str, str]] = {}

            for md in market_data:
                if md.yes_token_id:
                    token_to_contract[md.yes_token_id] = (md.contract_id, "YES")
                if md.no_token_id:
                    token_to_contract[md.no_token_id] = (md.contract_id, "NO")

            logger.info(
                "Built token mapping: %d tokens → %d contracts",
                len(token_to_contract),
                len({c for c, _ in token_to_contract.values()}),
            )

            # Fetch positions from API
            api_positions = auth_client.fetch_positions()

            # Map to contract_ids: track YES and NO separately
            pos_dict: dict[str, tuple[float, float]] = {}  # contract_id -> (yes_size, no_size)
            for p in api_positions:
                mapping = token_to_contract.get(p.token_id)
                if mapping:
                    contract_id, outcome = mapping
                    yes_size, no_size = pos_dict.get(contract_id, (0.0, 0.0))
                    if outcome == "YES":
                        yes_size += p.size
                    else:
                        no_size += p.size
                    pos_dict[contract_id] = (yes_size, no_size)
                else:
                    # Position not in this event
                    pass

            return pos_dict

        except Exception as e:
            logger.warning("Failed to fetch positions: %s", e)
            return None

    def fetch_live_balance() -> float:
        """Fetch live USDC balance.

        Falls back to config budget if API returns 0 (API often doesn't reflect actual balance).
        """
        if not auth_client or not auth_client.has_private_key:
            return cfg.budget
        try:
            balance = auth_client.fetch_balance()
            # CLOB balance = unspent USDC (not in positions)
            # $0 is normal if all USDC is in positions - use budget as buying power
            if balance < 0.01:
                logger.info(
                    "CLOB balance: $0 (cash is in positions). Using budget $%.2f as buying power",
                    cfg.budget,
                )
                return cfg.budget
            return balance
        except Exception as e:
            logger.warning("Failed to fetch balance: %s", e)
            return cfg.budget

    # Create position fetch function if live mode enabled
    fetch_pos_fn = fetch_positions_mapped if cfg.live else None

    # Get budget (live or from config)
    budget = fetch_live_balance() if cfg.live else cfg.budget

    try:
        if cfg.loop:
            # Continuous monitoring
            print(f"Starting continuous monitoring (interval: {cfg.interval}s)")
            if cfg.fee > 0:
                print(f"Trading fee: {cfg.fee} cents per contract")
            print("Press Ctrl+C to stop\n")

            while not shutdown_requested:
                # Refresh balance each iteration if in live mode
                if cfg.live:
                    budget = fetch_live_balance()

                success = run_single_update(
                    cfg.event, estimator, budget, fetch_pos_fn, cfg.fee
                )

                if cfg.estimator_state and success:
                    save_estimator_state(estimator.state, estimator.config, cfg.estimator_state)

                if shutdown_requested:
                    break

                # Wait for next interval
                for _ in range(cfg.interval):
                    if shutdown_requested:
                        break
                    time.sleep(1)

            print("\nMonitoring stopped.")
        else:
            # Single run
            success = run_single_update(cfg.event, estimator, budget, fetch_pos_fn, cfg.fee)

            if cfg.estimator_state and success:
                save_estimator_state(estimator.state, estimator.config, cfg.estimator_state)

            return 0 if success else 1

    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
        print(f"\nFATAL ERROR: {e}")
        return 1

    # Save state on exit
    if cfg.estimator_state and estimator.is_initialized:
        save_estimator_state(estimator.state, estimator.config, cfg.estimator_state)
        print(f"Saved estimator state to: {cfg.estimator_state}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
