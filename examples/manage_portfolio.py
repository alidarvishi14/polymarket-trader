#!/usr/bin/env python3
"""Incremental portfolio management for Polymarket term structures.

This script manages a portfolio across multiple runs, reading state from a JSON file,
optimizing trades, and writing updated state back.

Usage:
    # First run (initialize state file)
    python examples/manage_portfolio.py --init us-strikes-iran-by \
        --budget 1000 --delta-tol 10 --theta-tol 10 --state portfolio.json

    # Subsequent runs (read existing state, optimize, write new state)
    python examples/manage_portfolio.py --state portfolio.json

    # Dry run (show trades without updating state)
    python examples/manage_portfolio.py --state portfolio.json --dry-run

    # Live mode (fetch positions from Polymarket API)
    python examples/manage_portfolio.py --live us-strikes-iran-by \
        --budget 1000 --delta-tol 10 --theta-tol 10

    # Bayesian mode (maintains beliefs about hazard rates)
    python examples/manage_portfolio.py --state portfolio.json --bayesian \
        --estimator-state estimator.json --show-uncertainty
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from polymarket_trader.data import (
    AuthenticatedPolymarketClient,
    PolymarketClient,
    transform_to_maturity_data,
)
from polymarket_trader.data.transformer import EASTERN_TZ, extract_event_title
from polymarket_trader.models.bayesian_hazard import (
    BayesianHazardConfig,
    BayesianHazardEstimator,
)
from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.optimization.curve_fitting import HazardCurveFitter, LossType
from polymarket_trader.optimization.simple_portfolio import optimize_simple_portfolio
from polymarket_trader.state import (
    PortfolioState,
    load_estimator_state,
    load_state,
    save_estimator_state,
    save_state,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def match_positions_to_market(
    state: PortfolioState,
    market_data: list[MaturityData],
) -> tuple[np.ndarray, list[str]]:
    """Match state positions to current market data.

    Handles:
    - Contracts in state but not in market (expired) -> removed
    - Contracts in market but not in state -> position = 0

    Args:
        state: Current portfolio state.
        market_data: Current market data (sorted by maturity).

    Returns:
        Tuple of (positions array, contract_ids list) aligned with market_data.

    """
    contract_ids = [d.contract_id for d in market_data]
    positions = np.array([state.get_position(cid) for cid in contract_ids], dtype=np.float64)

    # Log any expired contracts
    state_contracts = set(state.positions.keys())
    market_contracts = set(contract_ids)
    expired = state_contracts - market_contracts
    if expired:
        logger.warning(
            "Contracts in state but not in market (expired?): %s",
            expired,
        )

    return positions, contract_ids


def build_token_to_contract_map(
    market_data: list[MaturityData],
) -> dict[str, str]:
    """Build mapping from token_id to contract_id.

    Args:
        market_data: List of MaturityData with contract_ids and token_ids.

    Returns:
        Dict mapping token_id to contract_id.

    """
    # Now MaturityData directly stores yes_token_id and no_token_id
    # No need to match by index or price - just use the token IDs directly

    token_to_contract: dict[str, str] = {}

    for mat_data in market_data:
        # Map YES token
        if mat_data.yes_token_id:
            token_to_contract[mat_data.yes_token_id] = mat_data.contract_id

        # Map NO token (same contract, just different token)
        if mat_data.no_token_id:
            token_to_contract[mat_data.no_token_id] = mat_data.contract_id

    logger.info("Built token→contract mapping: %d tokens", len(token_to_contract))
    return token_to_contract


def fetch_live_positions(
    auth_client: AuthenticatedPolymarketClient,
    token_to_contract: dict[str, str],
    all_contract_ids: list[str],
) -> dict[str, float]:
    """Fetch positions from API and map to contract IDs.

    Args:
        auth_client: Authenticated Polymarket client.
        token_to_contract: Mapping from token_id to contract_id.
        all_contract_ids: List of all contract IDs in the event.

    Returns:
        Dict mapping contract_id to net position (positive=YES, negative=NO).

    """
    # Fetch raw positions from API
    api_positions = auth_client.fetch_positions()

    # Map to contract IDs
    positions: dict[str, float] = dict.fromkeys(all_contract_ids, 0.0)

    for pos in api_positions:
        contract_id = token_to_contract.get(pos.token_id)
        if contract_id is None:
            logger.debug(
                "Position token %s not in this event, skipping",
                pos.token_id[:16] if pos.token_id else "unknown",
            )
            continue

        # Add to position (YES = positive, NO = negative)
        positions[contract_id] = positions.get(contract_id, 0.0) + pos.net_position

    # Log summary
    non_zero = {k: v for k, v in positions.items() if abs(v) > 1e-6}
    if non_zero:
        logger.info("Live positions: %s", non_zero)
    else:
        logger.info("No positions found for this event")

    return positions


def run_optimization(
    state: PortfolioState,
    market_data: list[MaturityData],
    reference_time: datetime,
    bayesian_estimator: BayesianHazardEstimator | None = None,
) -> tuple[PortfolioState, dict]:
    """Run the incremental portfolio optimization.

    Args:
        state: Current portfolio state.
        market_data: Current market data.
        reference_time: Current time (for computing expiry dates from DTE).
        bayesian_estimator: Optional Bayesian estimator (if None, uses QP fitter).

    Returns:
        Tuple of (new_state, report_data).

    """
    config = state.config

    # Sort market data by maturity
    sorted_data = sorted(market_data, key=lambda x: x.maturity)

    # Extract price arrays
    market_prices = np.array([d.market_price for d in sorted_data])
    bid_prices = np.array([d.bid_price for d in sorted_data])
    ask_prices = np.array([d.ask_price for d in sorted_data])

    # Match positions to market data
    current_positions, contract_ids = match_positions_to_market(state, sorted_data)

    # Get hazard model (Bayesian or QP)
    if bayesian_estimator is not None:
        model = bayesian_estimator.get_model()
        model_source = "bayesian"
    else:
        fitter = HazardCurveFitter(
            smoothness_alpha=config.smoothness_alpha,
            loss_type=LossType.QUADRATIC,
        )
        fit_result = fitter.fit(sorted_data)
        model = fit_result.model
        model_source = "qp"

    # Compute current exposures
    survival_probs = model.survival_probabilities
    theta_weights = model.bucket_hazards * survival_probs
    current_delta = float(np.sum(current_positions * survival_probs))
    current_theta = float(-np.sum(current_positions * theta_weights))

    # Run optimization
    result = optimize_simple_portfolio(
        model=model,
        market_prices=market_prices,
        budget=state.budget,
        min_theta=config.target_theta - config.theta_tolerance,
        bid_prices=bid_prices,
        ask_prices=ask_prices,
        use_spread=config.use_spread,
        current_positions=current_positions,
        target_delta=config.target_delta,
        target_theta=config.target_theta,
        delta_tolerance=config.delta_tolerance,
        theta_tolerance=config.theta_tolerance,
    )

    # Build new positions dict (include all contracts, even zeros)
    new_positions = {}
    for i, pos in enumerate(result.positions):
        cid = contract_ids[i]
        new_positions[cid] = round(pos.net_exposure, 2)

    # Compute trades from the 4 action variables
    trades = {}
    for i, pos in enumerate(result.positions):
        cid = contract_ids[i]
        actions = []

        if pos.buy_yes > 1e-6:
            actions.append(
                {
                    "action": "BUY YES",
                    "quantity": round(pos.buy_yes, 2),
                    "price": ask_prices[i] if config.use_spread else market_prices[i],
                }
            )
        if pos.sell_yes > 1e-6:
            actions.append(
                {
                    "action": "SELL YES",
                    "quantity": round(pos.sell_yes, 2),
                    "price": bid_prices[i] if config.use_spread else market_prices[i],
                }
            )
        if pos.buy_no > 1e-6:
            actions.append(
                {
                    "action": "BUY NO",
                    "quantity": round(pos.buy_no, 2),
                    "price": (1 - bid_prices[i]) if config.use_spread else (1 - market_prices[i]),
                }
            )
        if pos.sell_no > 1e-6:
            actions.append(
                {
                    "action": "SELL NO",
                    "quantity": round(pos.sell_no, 2),
                    "price": (1 - ask_prices[i]) if config.use_spread else (1 - market_prices[i]),
                }
            )

        if actions:
            net_change = pos.net_exposure - current_positions[i]
            # Delta contribution = position_change × survival_prob
            delta_contrib = net_change * survival_probs[i]
            # Theta contribution = -position_change × bucket_hazard × survival_prob
            theta_contrib = -net_change * theta_weights[i]
            trades[cid] = {
                "actions": actions,
                "net_pos": round(net_change, 2),
                "delta": round(delta_contrib, 2),
                "theta": round(theta_contrib, 4),
            }

    # Create new state
    new_state = PortfolioState(
        event=state.event,
        positions=new_positions,
        budget=round(state.budget - result.total_cost, 2),
        config=state.config,
    )

    # Compute expiry dates from DTE
    expiry_dates = []
    for dte in model.maturities:
        expiry = reference_time + timedelta(days=float(dte))
        expiry_dates.append(expiry.strftime("%Y-%m-%d"))

    # Build report data
    report = {
        "before": {
            "positions": {
                contract_ids[i]: round(float(current_positions[i]), 2)
                for i in range(len(contract_ids))
            },
            "delta": round(current_delta, 4),
            "theta": round(current_theta, 4),
            "budget": round(state.budget, 2),
        },
        "after": {
            "positions": new_positions,
            "delta": round(result.delta_exposure, 4),
            "theta": round(result.theta_exposure, 4),
            "budget": new_state.budget,
        },
        "trades": trades,
        "total_cost": round(result.total_cost, 2),
        "total_edge": round(result.total_edge, 4),
        "model": {
            "source": model_source,
            "maturities": [float(m) for m in model.maturities],
            "model_prices": [float(p) for p in model.theoretical_prices],
            "market_prices": [float(p) for p in market_prices],
            "bid_prices": [float(p) for p in bid_prices],
            "ask_prices": [float(p) for p in ask_prices],
            "mispricings": [float(m) for m in model.mispricings(market_prices)],
            "expiry_dates": expiry_dates,
        },
        "config": {
            "target_delta": config.target_delta,
            "target_theta": config.target_theta,
            "delta_tolerance": config.delta_tolerance,
            "theta_tolerance": config.theta_tolerance,
        },
        "contract_ids": contract_ids,
    }

    return new_state, report


def print_report(report: dict, show_uncertainty: bool = False) -> None:
    """Print human-readable report to console.

    Args:
        report: Report data dictionary.
        show_uncertainty: Whether to show confidence intervals (Bayesian mode).

    """
    print("=" * 70)
    print("PORTFOLIO UPDATE")
    print(f"Time: {datetime.now(EASTERN_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("=" * 70)

    # Before state
    print("\nBEFORE (input state):")
    before = report["before"]
    if before["positions"]:
        print("  Positions:")
        for cid, pos in before["positions"].items():
            print(f"    {cid}: {pos:+.2f}")
    else:
        print("  Positions: (none)")
    print(f"  Delta:  {before['delta']:+.4f}")
    print(f"  Theta:  {before['theta']:+.4f}")
    print(f"  Budget: ${before['budget']:,.2f}")

    # Tolerances
    config = report["config"]
    print("\nTARGETS:")
    print(f"  Delta: {config['target_delta']:.1f} ± {config['delta_tolerance']:.1f}")
    print(f"  Theta: {config['target_theta']:.1f} ± {config['theta_tolerance']:.1f}")

    # Model fit
    model_data = report["model"]
    model_source = model_data.get("source", "qp")
    has_ci = show_uncertainty and "ci_lower" in model_data

    if has_ci:
        print("\n" + "-" * 110)
        print("FITTED HAZARD CURVE (Bayesian, 95% CI)")
        print("-" * 110)
        print(
            f"{'Expiry':<12} {'DTE':>5} {'Bid':>8} {'Market':>8} {'Ask':>8} "
            f"{'Model':>8} {'[  95% CI  ]':>14} {'Mispricing':>10} {'Contract':<10}"
        )
        print("-" * 110)
        for i in range(len(model_data["maturities"])):
            ci_str = f"[{model_data['ci_lower'][i]:>5.1%}-{model_data['ci_upper'][i]:<5.1%}]"
            print(
                f"{model_data['expiry_dates'][i]:<12} "
                f"{model_data['maturities'][i]:>5.0f} "
                f"{model_data['bid_prices'][i]:>8.2%} "
                f"{model_data['market_prices'][i]:>8.2%} "
                f"{model_data['ask_prices'][i]:>8.2%} "
                f"{model_data['model_prices'][i]:>8.2%} "
                f"{ci_str:>14} "
                f"{model_data['mispricings'][i]:>+10.2%} "
                f"{report['contract_ids'][i]:<10}"
            )
    else:
        title = "FITTED HAZARD CURVE" if model_source == "qp" else "FITTED HAZARD CURVE (Bayesian)"
        print("\n" + "-" * 90)
        print(title)
        print("-" * 90)
        print(
            f"{'Expiry':<12} {'DTE':>5} {'Bid':>8} {'Market':>8} {'Ask':>8} "
            f"{'Model':>8} {'Mispricing':>10} {'Contract':<12}"
        )
        print("-" * 90)
        for i in range(len(model_data["maturities"])):
            print(
                f"{model_data['expiry_dates'][i]:<12} "
                f"{model_data['maturities'][i]:>5.0f} "
                f"{model_data['bid_prices'][i]:>8.2%} "
                f"{model_data['market_prices'][i]:>8.2%} "
                f"{model_data['ask_prices'][i]:>8.2%} "
                f"{model_data['model_prices'][i]:>8.2%} "
                f"{model_data['mispricings'][i]:>+10.2%} "
                f"{report['contract_ids'][i]:<12}"
            )

    # Trades
    print("\n" + "-" * 100)
    print("RECOMMENDED TRADES")
    print("-" * 100)
    trades = report["trades"]
    if trades:
        print(
            f"{'Contract':<12} {'Action':<10} {'Qty':>8} {'Price':>8} "
            f"{'Net Pos':>10} {'Delta':>10} {'Theta':>10}"
        )
        print("-" * 100)
        for cid, trade_info in trades.items():
            first_action = True
            for action in trade_info["actions"]:
                print(
                    f"{cid:<12} "
                    f"{action['action']:<10} "
                    f"{action['quantity']:>8.2f} "
                    f"{action['price']:>8.2%} "
                    f"{trade_info['net_pos']:>+10.2f} "
                    f"{trade_info['delta']:>+10.2f} "
                    f"{trade_info['theta']:>+10.4f}"
                    if first_action
                    else f"{'':<12} "
                    f"{action['action']:<10} "
                    f"{action['quantity']:>8.2f} "
                    f"{action['price']:>8.2%} "
                    f"{'':<10} "
                    f"{'':<10} "
                    f"{'':<10}"
                )
                first_action = False
                cid = ""  # Don't repeat contract name for multiple actions
        print("-" * 100)
        cost_label = "Net cost:" if report["total_cost"] >= 0 else "Net credit:"
        print(f"  {cost_label:<12} ${abs(report['total_cost']):>10.2f}")
        print(f"  Expected edge: ${report['total_edge']:>10.4f}")
    else:
        print("  No trades recommended.")

    # After state
    after = report["after"]
    print("\n" + "-" * 70)
    print("AFTER (output state):")
    print("-" * 70)
    if after["positions"]:
        print("  Positions:")
        for cid, pos in after["positions"].items():
            print(f"    {cid}: {pos:+.2f}")
    else:
        print("  Positions: (none)")

    # Check if within tolerance
    delta_ok = abs(after["delta"] - config["target_delta"]) <= config["delta_tolerance"]
    theta_ok = abs(after["theta"] - config["target_theta"]) <= config["theta_tolerance"]

    print(f"  Delta:  {after['delta']:+.4f}  {'✓' if delta_ok else '⚠️'}")
    print(f"  Theta:  {after['theta']:+.4f}  {'✓' if theta_ok else '⚠️'}")
    print(f"  Budget: ${after['budget']:,.2f}")

    print("=" * 70)


def main() -> int:
    """Run the portfolio management CLI."""
    parser = argparse.ArgumentParser(
        description="Incremental portfolio management for Polymarket term structures"
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Path to portfolio state JSON file (required unless --live)",
    )
    parser.add_argument(
        "--init",
        type=str,
        metavar="EVENT",
        help="Initialize new state file with this event (URL or slug)",
    )
    parser.add_argument(
        "--live",
        type=str,
        metavar="EVENT",
        help="Fetch positions from Polymarket API for this event (requires .env)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1000.0,
        help="Initial budget (only used with --init)",
    )
    parser.add_argument(
        "--delta-tol",
        type=float,
        default=10.0,
        help="Delta tolerance (only used with --init)",
    )
    parser.add_argument(
        "--theta-tol",
        type=float,
        default=10.0,
        help="Theta tolerance (only used with --init)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Smoothness alpha for curve fitting (only used with --init)",
    )
    parser.add_argument(
        "--no-spread",
        action="store_true",
        help="Use mid prices instead of bid/ask (only used with --init)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show trades without updating state file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Bayesian mode arguments
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Use Bayesian hazard estimator instead of QP curve fitting",
    )
    parser.add_argument(
        "--estimator-state",
        type=Path,
        help="Path to Bayesian estimator state file (required with --bayesian)",
    )
    parser.add_argument(
        "--show-uncertainty",
        action="store_true",
        help="Show confidence intervals for model prices (only with --bayesian)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.85,
        help="Correlation decay for Bayesian estimator (only with --bayesian --init)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.bayesian and not args.estimator_state:
        parser.error("--bayesian requires --estimator-state")

    if not args.live and not args.state:
        parser.error("--state is required (unless using --live)")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Track if we're in live mode
    live_mode = args.live is not None
    auth_client = None

    try:
        # Live mode: fetch positions from API
        if live_mode:
            event_slug = args.live

            print(f"\n{'=' * 70}")
            print("LIVE MODE - Fetching positions from Polymarket API")
            print(f"{'=' * 70}")

            try:
                auth_client = AuthenticatedPolymarketClient.from_env()
                print(f"Wallet: {auth_client.address[:6]}...{auth_client.address[-4:]}")
            except ValueError as e:
                print(f"ERROR: {e}")
                print("\nTo use --live mode, create a .env file with:")
                print("  POLYMARKET_ADDRESS=0x...your-wallet-address...")
                print("  POLYMARKET_PRIVATE_KEY=0x...your-private-key... (optional, for balance)")
                return 1

            # Create initial state (positions will be fetched after we get market data)
            state = PortfolioState.create_initial(
                event=event_slug,
                budget=args.budget,
                delta_tolerance=args.delta_tol,
                theta_tolerance=args.theta_tol,
                use_spread=not args.no_spread,
                smoothness_alpha=args.alpha,
            )

        # Initialize from file or load existing state
        elif args.init:
            if args.state.exists():
                print(f"ERROR: State file already exists: {args.state}")
                print("Remove it first or use a different path.")
                return 1

            state = PortfolioState.create_initial(
                event=args.init,
                budget=args.budget,
                delta_tolerance=args.delta_tol,
                theta_tolerance=args.theta_tol,
                use_spread=not args.no_spread,
                smoothness_alpha=args.alpha,
            )
            print(f"Initialized new state for event: {args.init}")
        else:
            if not args.state.exists():
                print(f"ERROR: State file not found: {args.state}")
                print("Use --init EVENT to create a new state file.")
                return 1
            state = load_state(args.state)

        # Fetch market data
        print(f"\nFetching data from Polymarket: {state.event}")
        print("-" * 70)

        with PolymarketClient() as client:
            markets = client.fetch_event(state.event)

        if not markets:
            print("ERROR: No markets found for this event")
            return 1

        event_title = extract_event_title(markets)
        print(f"Event: {event_title}")
        print(f"Found {len(markets)} contracts")

        # Transform to MaturityData
        now_et = datetime.now(EASTERN_TZ)
        market_data = transform_to_maturity_data(markets, reference_time=now_et)

        if len(market_data) < 2:
            print(f"ERROR: Need at least 2 active contracts, found {len(market_data)}")
            return 1

        print(f"Active contracts: {len(market_data)}")

        # In live mode, fetch positions from API
        if live_mode and auth_client is not None:
            # Build token→contract mapping (now uses token IDs stored in MaturityData)
            token_to_contract = build_token_to_contract_map(market_data)

            # Fetch positions from API
            contract_ids = [d.contract_id for d in market_data]
            live_positions = fetch_live_positions(
                auth_client,
                token_to_contract,
                contract_ids,
            )

            # Update state with live positions
            state.positions = live_positions

            # Try to fetch balance (requires private key)
            if auth_client.has_private_key:
                try:
                    balance = auth_client.fetch_balance()
                    state.budget = balance
                    print(f"Live balance: ${balance:,.2f} USDC")
                except Exception as e:
                    logger.warning("Could not fetch balance: %s", e)
                    print(f"Could not fetch balance, using: ${state.budget:,.2f}")
            else:
                print(f"No private key - using budget: ${state.budget:,.2f}")

            # Show live positions summary
            non_zero = {k: v for k, v in state.positions.items() if abs(v) > 1e-6}
            if non_zero:
                print("\nLive positions loaded:")
                for cid, pos in non_zero.items():
                    side = "YES" if pos > 0 else "NO"
                    print(f"  {cid}: {abs(pos):.2f} {side}")
            else:
                print("\nNo positions in this event")

        # Handle Bayesian mode
        estimator = None
        if args.bayesian:
            if args.init or not args.estimator_state.exists():
                # Initialize new Bayesian estimator
                config = BayesianHazardConfig(
                    rho=args.rho,
                    log_lambda_std=0.3,
                    obs_noise_scale=0.25,
                    process_noise_std=0.01,
                    min_variance=1e-4,
                    isolation_penalty=2.0,
                )
                estimator = BayesianHazardEstimator(config)
                estimator.initialize(market_data, now_et)
                print(f"Initialized Bayesian estimator: rho={args.rho}")
            else:
                # Load existing estimator state
                est_state, est_config = load_estimator_state(args.estimator_state)
                estimator = BayesianHazardEstimator(est_config)
                estimator.state = est_state
                # Update with new observations
                diagnostics = estimator.update(market_data, now_et)
                print(
                    f"Updated Bayesian estimator: n_updates={estimator.state.n_updates}, "
                    f"uncertainty={diagnostics.posterior_uncertainty:.4f}"
                )

        # Run optimization
        new_state, report = run_optimization(
            state,
            market_data,
            now_et,
            bayesian_estimator=estimator,
        )

        # Add uncertainty info to report if requested
        if args.bayesian and args.show_uncertainty and estimator is not None:
            lower, upper = estimator.get_confidence_intervals()
            report["model"]["ci_lower"] = [float(x) for x in lower]
            report["model"]["ci_upper"] = [float(x) for x in upper]

        # Print report
        print_report(report, show_uncertainty=args.show_uncertainty)

        # Save state (unless dry run or live mode without --state)
        if args.dry_run:
            print("\n[DRY RUN] State files not updated.")
        elif live_mode and not args.state:
            print("\n[LIVE MODE] No state file specified (use --state to save)")
        else:
            if args.state:
                save_state(new_state, args.state)
                print(f"\nState written to: {args.state}")

            if args.bayesian and estimator is not None:
                save_estimator_state(
                    estimator.state,
                    estimator.config,
                    args.estimator_state,
                )
                print(f"Estimator state written to: {args.estimator_state}")

        return 0

    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
