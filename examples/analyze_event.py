#!/usr/bin/env python3
"""Analyze a Polymarket term structure event.

Usage:
    python examples/analyze_event.py <url_or_slug> [--budget BUDGET]

Examples:
    python examples/analyze_event.py https://polymarket.com/event/us-strikes-iran-by
    python examples/analyze_event.py us-strikes-iran-by --budget 1000

"""

import argparse
import logging
import sys
from datetime import datetime

from polymarket_trader.analysis import analyze_term_structure
from polymarket_trader.data import PolymarketClient, transform_to_maturity_data
from polymarket_trader.data.transformer import EASTERN_TZ, extract_event_title

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Run the event analysis CLI."""
    parser = argparse.ArgumentParser(description="Analyze a Polymarket term structure event")
    parser.add_argument(
        "event",
        help="Polymarket event URL or slug (e.g., 'us-strikes-iran-by')",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=1000.0,
        help="Portfolio budget in USD (default: 1000)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Smoothness parameter for curve fitting (default: 0.01)",
    )
    parser.add_argument(
        "--min-theta",
        type=float,
        default=0.0,
        help="Minimum theta constraint (default: 0 = non-negative)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--use-spread",
        action="store_true",
        help="Use bid/ask prices for edge calculation instead of mid-market",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Fetch data from Polymarket
        print(f"\nFetching data from Polymarket: {args.event}")
        print("-" * 70)

        with PolymarketClient() as client:
            markets = client.fetch_event(args.event)

        if not markets:
            print("ERROR: No markets found for this event")
            return 1

        # Extract event title
        event_title = extract_event_title(markets)
        print(f"Event: {event_title}")
        print(f"Found {len(markets)} contracts")

        # Transform to MaturityData with current time
        now_et = datetime.now(EASTERN_TZ)
        market_data = transform_to_maturity_data(markets, reference_time=now_et)

        if len(market_data) < 2:
            print(f"ERROR: Need at least 2 active contracts, found {len(market_data)}")
            return 1

        print(f"Active contracts: {len(market_data)}")
        dte_range = f"{market_data[0].maturity:.2f} - {market_data[-1].maturity:.2f}"
        print(f"Time to expiry range: {dte_range} days")

        # Run analysis
        print("\n" + "=" * 70)
        print(f"ANALYSIS: {event_title}")
        print(f"Time: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Budget: ${args.budget:,.0f}")
        print(f"Smoothness alpha: {args.alpha}")
        print("=" * 70)

        result = analyze_term_structure(
            market_data=market_data,
            budget=args.budget,
            smoothness_alpha=args.alpha,
            delta_neutral=True,
            min_theta=args.min_theta,
            use_spread=args.use_spread,
        )

        # Print report
        result.print_report()

        # Print trade recommendations
        print("\n" + "=" * 70)
        print("TRADE RECOMMENDATIONS")
        print("=" * 70)

        # Find positions with significant size
        active_positions = [
            (i, p)
            for i, p in enumerate(result.portfolio.positions)
            if abs(p.yes_quantity) > 0.1 or abs(p.no_quantity) > 0.1
        ]

        if not active_positions:
            print("\nNo significant trades recommended with current parameters.")
        else:
            print("\nRecommended trades:")
            for i, pos in active_positions:
                contract = result.market_data[i]
                if pos.net_exposure > 0:
                    direction = "BUY YES"
                    qty = pos.yes_quantity
                else:
                    direction = "BUY NO"
                    qty = pos.no_quantity

                print(f"\n  {direction} {contract.contract_id.upper()}")
                print(f"    Days to expiry: {pos.maturity:.0f}")
                print(f"    Quantity: {qty:.1f} contracts")
                print(f"    Market price: {pos.market_price:.2%}")
                print(f"    Model price: {pos.model_price:.2%}")
                print(f"    Edge/contract: {pos.edge_per_contract:+.2%}")

        print("\n" + "=" * 70)
        print("DISCLAIMER: This is for educational purposes only.")
        print("Past performance does not guarantee future results.")
        print("Trade at your own risk.")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        print(f"\nERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
