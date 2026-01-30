#!/usr/bin/env python3
"""Find and merge YES/NO positions to redeem USDC.

On Polymarket, if you hold both YES and NO tokens for the same market,
you can merge them to get $1 USDC back per pair (1 YES + 1 NO = $1).

This script:
1. Fetches your current positions
2. Identifies markets where you hold both YES and NO
3. Shows mergeable pairs and potential USDC to redeem
4. Allows you to select which pairs to merge and confirm the transaction

Usage:
    # Show mergeable positions
    python examples/merge_positions.py

    # Interactive merge mode
    python examples/merge_positions.py --merge

Requirements:
    - POLYMARKET_ADDRESS in .env
    - POLYMARKET_PRIVATE_KEY in .env (required for merge execution)
    - POLYMARKET_FUNDER_ADDRESS in .env (recommended)
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv
from eth_account import Account
from web3 import Web3

from polymarket_trader.data.authenticated_client import (
    AuthenticatedPolymarketClient,
    UserPosition,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("web3").setLevel(logging.WARNING)

# Polygon network configuration
POLYGON_RPC = "https://polygon-rpc.com"
POLYGON_CHAIN_ID = 137

# Polymarket CTF Exchange contract on Polygon
# This contract handles merging YES+NO tokens back to collateral (USDC)
CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

# Minimal ABI for mergePositions function
# See: https://docs.polymarket.com/developers/CTF/merge
CTF_EXCHANGE_ABI = [
    {
        "inputs": [
            {"internalType": "contract IERC20", "name": "_collateral", "type": "address"},
            {"internalType": "contract IConditionalTokens", "name": "_ctf", "type": "address"},
            {"internalType": "bytes32", "name": "_conditionId", "type": "bytes32"},
            {"internalType": "uint256", "name": "_amount", "type": "uint256"},
        ],
        "name": "mergePositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

# USDC and CTF addresses on Polygon (mainnet)
USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC on Polygon
CTF_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"  # Conditional Tokens Framework


@dataclass
class MergeablePosition:
    """A pair of YES/NO positions that can be merged.

    Attributes:
        condition_id: The market condition ID.
        yes_position: The YES position.
        no_position: The NO position.
        mergeable_quantity: Min of YES and NO sizes (amount that can be merged).
        usdc_redeemable: USDC that would be returned ($1 per merged pair).

    """

    condition_id: str
    yes_position: UserPosition
    no_position: UserPosition
    mergeable_quantity: float
    usdc_redeemable: float


def find_mergeable_positions(positions: list[UserPosition]) -> list[MergeablePosition]:
    """Find positions where user holds both YES and NO for same market.

    Args:
        positions: List of all user positions.

    Returns:
        List of mergeable position pairs.

    """
    # Group positions by condition_id
    by_condition: dict[str, dict[str, UserPosition]] = {}

    for pos in positions:
        if pos.size < 0.01:  # Skip dust
            continue

        cid = pos.condition_id
        if cid not in by_condition:
            by_condition[cid] = {}

        side = pos.side  # 'YES' or 'NO'
        if side in by_condition[cid]:
            # Combine if multiple positions of same side (shouldn't happen normally)
            existing = by_condition[cid][side]
            by_condition[cid][side] = UserPosition(
                token_id=existing.token_id,
                condition_id=cid,
                outcome=existing.outcome,
                size=existing.size + pos.size,
                avg_price=existing.avg_price,  # Simplified
                current_price=pos.current_price,
                realized_pnl=existing.realized_pnl + pos.realized_pnl,
                cur_val_usd=existing.cur_val_usd + pos.cur_val_usd,
                initial_val_usd=existing.initial_val_usd + pos.initial_val_usd,
            )
        else:
            by_condition[cid][side] = pos

    # Find markets with both YES and NO
    mergeable = []
    for cid, sides in by_condition.items():
        if "YES" in sides and "NO" in sides:
            yes_pos = sides["YES"]
            no_pos = sides["NO"]
            merge_qty = min(yes_pos.size, no_pos.size)

            if merge_qty >= 1.0:  # Only show if at least 1 can be merged
                mergeable.append(
                    MergeablePosition(
                        condition_id=cid,
                        yes_position=yes_pos,
                        no_position=no_pos,
                        mergeable_quantity=merge_qty,
                        usdc_redeemable=merge_qty,  # $1 per pair
                    )
                )

    # Sort by redeemable amount (descending)
    mergeable.sort(key=lambda x: x.usdc_redeemable, reverse=True)

    return mergeable


def print_mergeable_positions(mergeable: list[MergeablePosition]) -> None:
    """Print table of mergeable positions.

    Args:
        mergeable: List of mergeable position pairs.

    """
    if not mergeable:
        print("\n  No mergeable positions found.")
        print("  (You need to hold both YES and NO tokens for the same market)")
        return

    print("\n" + "=" * 100)
    print("MERGEABLE POSITIONS")
    print("=" * 100)
    print(
        f"{'#':<4} {'Condition ID':<20} {'YES Size':>12} {'NO Size':>12} "
        f"{'Mergeable':>12} {'USDC Redeem':>14}"
    )
    print("-" * 100)

    total_redeemable = 0.0
    for i, m in enumerate(mergeable, 1):
        print(
            f"{i:<4} "
            f"{m.condition_id[:18]:<20} "
            f"{m.yes_position.size:>12.2f} "
            f"{m.no_position.size:>12.2f} "
            f"{m.mergeable_quantity:>12.2f} "
            f"${m.usdc_redeemable:>13.2f}"
        )
        total_redeemable += m.usdc_redeemable

    print("-" * 100)
    print(f"{'TOTAL':>50} {' ':>12} ${total_redeemable:>13.2f}")
    print("=" * 100)


def execute_merge(
    mergeable: MergeablePosition,
    quantity: float,
    private_key: str,
) -> bool:
    """Execute a merge transaction via CTF Exchange contract on Polygon.

    On Polymarket, merging burns YES+NO tokens and returns USDC:
    - CTF Exchange contract: 0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E
    - Function: mergePositions(collateral, ctf, conditionId, amount)

    Args:
        mergeable: The mergeable position pair.
        quantity: Amount to merge (will merge this many YES+NO pairs).
        private_key: Private key for signing the transaction.

    Returns:
        True if successful, False otherwise.

    """
    try:
        # Connect to Polygon
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
        if not w3.is_connected():
            logger.error("Failed to connect to Polygon RPC")
            return False

        # Get account from private key
        account = Account.from_key(private_key)
        sender_address = account.address

        logger.info(
            "Merging %d positions for condition %s...",
            int(quantity),
            mergeable.condition_id[:16],
        )

        # Create contract instance
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(CTF_EXCHANGE_ADDRESS),
            abi=CTF_EXCHANGE_ABI,
        )

        # Amount in 6 decimal units (USDC precision)
        amount_wei = int(quantity * 1_000_000)

        # Build transaction
        # conditionId must be bytes32, ensure it's properly formatted
        condition_id_bytes = bytes.fromhex(
            mergeable.condition_id[2:]
            if mergeable.condition_id.startswith("0x")
            else mergeable.condition_id
        )

        nonce = w3.eth.get_transaction_count(sender_address)
        gas_price = w3.eth.gas_price

        # Build the function call
        tx = contract.functions.mergePositions(
            Web3.to_checksum_address(USDC_ADDRESS),
            Web3.to_checksum_address(CTF_ADDRESS),
            condition_id_bytes,
            amount_wei,
        ).build_transaction(
            {
                "from": sender_address,
                "nonce": nonce,
                "gasPrice": gas_price,
                "chainId": POLYGON_CHAIN_ID,
            }
        )

        # Estimate gas
        try:
            gas_estimate = w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas_estimate * 1.2)  # 20% buffer
        except Exception as e:
            logger.warning("Gas estimation failed: %s. Using default gas limit.", e)
            tx["gas"] = 300_000  # Default gas limit

        # Sign transaction
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)

        # Send transaction
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info("Transaction sent: %s", tx_hash.hex())

        # Wait for receipt
        print(f"  Waiting for transaction confirmation...")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

        if receipt["status"] == 1:
            logger.info(
                "Merge successful! Tx: %s, Gas used: %d",
                tx_hash.hex(),
                receipt["gasUsed"],
            )
            return True
        else:
            logger.error("Transaction failed! Tx: %s", tx_hash.hex())
            return False

    except Exception as e:
        logger.error("Merge failed: %s", e, exc_info=True)
        return False


def interactive_merge(
    mergeable: list[MergeablePosition],
    private_key: str,
) -> None:
    """Interactive merge mode - let user select which positions to merge.

    Args:
        mergeable: List of mergeable position pairs.
        private_key: Private key for signing transactions.

    """
    if not mergeable:
        print("\n  No mergeable positions to process.")
        return

    print("\n" + "-" * 60)
    print("INTERACTIVE MERGE MODE")
    print("-" * 60)

    for i, m in enumerate(mergeable, 1):
        print(f"\n[{i}/{len(mergeable)}] Condition: {m.condition_id[:20]}...")
        print(f"  YES: {m.yes_position.size:.2f} | NO: {m.no_position.size:.2f}")
        print(f"  Max mergeable: {m.mergeable_quantity:.2f} → ${m.usdc_redeemable:.2f} USDC")

        while True:
            try:
                user_input = input(
                    f"\n  Enter quantity to merge (0 to skip, max {m.mergeable_quantity:.0f}): "
                ).strip()

                if user_input.lower() in ("q", "quit", "exit"):
                    print("\nExiting merge mode.")
                    return

                if user_input == "" or user_input == "0":
                    print("  Skipped.")
                    break

                quantity = float(user_input)

                if quantity < 0:
                    print("  Error: Quantity must be non-negative.")
                    continue

                if quantity > m.mergeable_quantity:
                    print(f"  Error: Max quantity is {m.mergeable_quantity:.2f}")
                    continue

                if quantity < 1:
                    print("  Error: Minimum merge quantity is 1.")
                    continue

                # Confirm
                confirm = input(
                    f"  Confirm merge {quantity:.0f} pairs for ${quantity:.2f} USDC? (y/n): "
                ).strip().lower()

                if confirm in ("y", "yes"):
                    print("  Executing merge on Polygon...")
                    success = execute_merge(m, quantity, private_key)
                    if success:
                        print(f"  ✓ Successfully merged {quantity:.0f} pairs!")
                    else:
                        print("  ✗ Merge failed. See logs for details.")
                else:
                    print("  Cancelled.")

                break

            except ValueError:
                print("  Error: Please enter a valid number.")

    print("\n" + "-" * 60)
    print("Merge session complete.")


def main() -> int:
    """Run the merge positions CLI."""
    parser = argparse.ArgumentParser(
        description="Find and merge YES/NO positions to redeem USDC"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable interactive merge mode (otherwise just display)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    try:
        # Initialize client
        client = AuthenticatedPolymarketClient.from_env()
        print(f"Connected: {client.address[:6]}...{client.address[-4:]}")

        # Fetch positions
        print("\nFetching positions...")
        positions = client.fetch_positions()
        print(f"Found {len(positions)} positions.")

        # Find mergeable pairs
        mergeable = find_mergeable_positions(positions)

        # Display
        print_mergeable_positions(mergeable)

        # Interactive merge if requested
        if args.merge:
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
            if not private_key:
                print(
                    "\nError: Private key required for merge execution. "
                    "Set POLYMARKET_PRIVATE_KEY in .env"
                )
                return 1

            interactive_merge(mergeable, private_key)

        return 0

    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
