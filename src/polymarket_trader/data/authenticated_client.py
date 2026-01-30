"""Authenticated client for Polymarket API with position and balance access.

This module provides authenticated access to Polymarket APIs:
- Data API: User positions (public, just needs address)
- CLOB API: Balance, orders (requires Level 2 auth with private key)

Usage:
    from polymarket_trader.data.authenticated_client import AuthenticatedPolymarketClient

    client = AuthenticatedPolymarketClient.from_env()
    positions = client.fetch_positions()
    balance = client.fetch_balance()
"""

import functools
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# API endpoints
DATA_API_BASE = "https://data-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137


def _retry_on_connection_error(
    max_retries: int, base_delay: float
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry function on connection errors with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.

    Returns:
        Decorator that wraps the function with retry logic.

    """
    # Import py-clob-client exception for CLOB API retry
    try:
        from py_clob_client.exceptions import PolyApiException as PolyExc
    except ImportError:
        PolyExc = Exception  # type: ignore[misc, assignment]  # noqa: N806

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ConnectionError,
                    httpx.RequestError,
                    ConnectionResetError,
                    PolyExc,  # Catch py-clob-client errors
                ) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = base_delay * (1.5**attempt)
                        logger.warning(
                            "Connection error on attempt %d/%d, retrying in %.1fs: %s",
                            attempt + 1,
                            max_retries + 1,
                            delay,
                            str(e)[:100],
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "Connection failed after %d attempts: %s",
                            max_retries + 1,
                            e,
                        )
            raise last_exception

        return wrapper

    return decorator


@dataclass
class UserPosition:
    """A user's position in a Polymarket market.

    Attributes:
        token_id: The token ID (YES or NO token).
        condition_id: The market condition ID.
        outcome: 'Yes' or 'No'.
        size: Number of contracts held.
        avg_price: Average entry price.
        current_price: Current market price.
        realized_pnl: Realized profit/loss.
        cur_val_usd: Current value in USD.
        initial_val_usd: Initial value in USD.

    """

    token_id: str
    condition_id: str
    outcome: str
    size: float
    avg_price: float
    current_price: float
    realized_pnl: float
    cur_val_usd: float
    initial_val_usd: float

    @classmethod
    def from_api_response(cls, data: dict) -> "UserPosition":
        """Create from data-api response."""
        return cls(
            token_id=data.get("asset", ""),
            condition_id=data.get("conditionId", data.get("market", "")),
            outcome=data.get("outcome", ""),
            size=float(data.get("size", 0)),
            avg_price=float(data.get("avgPrice", 0)),
            current_price=float(data.get("curPrice", data.get("currentPrice", 0))),
            realized_pnl=float(data.get("realizedPnl", 0)),
            cur_val_usd=float(data.get("curValUsd", 0)),
            initial_val_usd=float(data.get("initialValUsd", 0)),
        )

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        return self.cur_val_usd - self.initial_val_usd

    @property
    def side(self) -> str:
        """Return 'YES' or 'NO' based on outcome."""
        return self.outcome.upper() if self.outcome else "UNKNOWN"

    @property
    def net_position(self) -> float:
        """Net position: positive for YES, negative for NO."""
        if self.side == "YES":
            return self.size
        elif self.side == "NO":
            return -self.size
        return 0.0


class AuthenticatedPolymarketClient:
    """Authenticated client for Polymarket APIs.

    Provides access to:
    - User positions (via public data-api)
    - User balance (via authenticated CLOB API)

    Attributes:
        address: User's wallet address (EOA).
        funder_address: User's proxy/funder wallet address (holds positions).
        private_key: User's private key (for CLOB auth).
        signature_type: Signature type (0=EOA, 1=Magic, 2=Proxy).
        chain_id: Polygon chain ID (137).

    Note:
        Polymarket uses a proxy wallet system. Your positions are held by
        a proxy contract (funder_address), not your EOA directly. If you
        don't provide a funder_address, it will be derived from your orders.

    """

    def __init__(
        self,
        address: str,
        private_key: str | None = None,
        funder_address: str | None = None,
        signature_type: int = 2,
        chain_id: int = POLYGON_CHAIN_ID,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the authenticated client.

        Args:
            address: User's wallet address/EOA (required).
            private_key: User's private key (required for balance/orders).
            funder_address: User's proxy/funder wallet (holds positions).
                           If not provided, will be derived from orders.
            signature_type: Signature type for CLOB API. Default=2 (Poly EOA).
                           0=EOA/MetaMask (legacy), 1=GNOSIS SAFE, 2=Poly EOA (most common).
            chain_id: Polygon chain ID.
            timeout: Request timeout in seconds.

        """
        if not address:
            raise ValueError("Wallet address is required")

        self._address = address.lower()
        self._private_key = private_key
        self._funder_address = funder_address.lower() if funder_address else None
        self._signature_type = signature_type
        self._chain_id = chain_id
        self._timeout = timeout

        # HTTP client with HTTP/2 support (fixes SSL connection reset issues)
        self._http_client = httpx.Client(
            http2=True,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": "polymarket-trader/0.1.0",
            },
        )

        # CLOB client (lazy initialized when needed)
        self._clob_client = None
        self._api_creds = None

        logger.info(
            "AuthenticatedPolymarketClient initialized: address=%s...%s",
            self._address[:6],
            self._address[-4:],
        )

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "AuthenticatedPolymarketClient":
        """Create client from environment variables.

        Loads from .env file if present, then from environment.

        Args:
            env_file: Path to .env file. If None, searches in current directory.

        Returns:
            Configured AuthenticatedPolymarketClient.

        Raises:
            ValueError: If required environment variables are missing.

        """
        # Load .env file
        if env_file:
            load_dotenv(env_file)
        else:
            load_dotenv()

        address = os.getenv("POLYMARKET_ADDRESS")
        if not address:
            raise ValueError(
                "POLYMARKET_ADDRESS environment variable is required. "
                "Set it in .env file or export it."
            )

        private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        # Default to signature_type=2 (Poly EOA) - most common for Polymarket users
        signature_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "2"))
        chain_id = int(os.getenv("POLYMARKET_CHAIN_ID", str(POLYGON_CHAIN_ID)))

        return cls(
            address=address,
            private_key=private_key,
            funder_address=funder_address,
            signature_type=signature_type,
            chain_id=chain_id,
        )

    @property
    def address(self) -> str:
        """Get wallet address."""
        return self._address

    @property
    def has_private_key(self) -> bool:
        """Whether private key is configured."""
        return self._private_key is not None

    @property
    def funder_address(self) -> str | None:
        """Get funder/proxy wallet address."""
        return self._funder_address

    def _derive_funder_from_orders(self) -> str | None:
        """Derive funder address from user's orders.

        The funder/proxy address can be found in the maker_address field
        of any order placed by this user.

        Returns:
            Funder address if found, None otherwise.

        """
        if not self._private_key:
            logger.warning("Cannot derive funder address without private key")
            return None

        try:
            clob_client = self._get_clob_client()
            orders = clob_client.get_orders()

            if orders and len(orders) > 0:
                maker_address = orders[0].get("maker_address")
                if maker_address:
                    logger.info(
                        "Derived funder address from orders: %s...%s",
                        maker_address[:6],
                        maker_address[-4:],
                    )
                    return maker_address.lower()

            logger.warning("No orders found to derive funder address")
            return None

        except Exception as e:
            logger.error("Failed to derive funder address: %s", e)
            return None

    def get_positions_address(self) -> str:
        """Get the address to use for fetching positions.

        Returns funder_address if available, otherwise attempts to derive it,
        and falls back to EOA address.

        Returns:
            Address to query for positions.

        """
        if self._funder_address:
            return self._funder_address

        # Try to derive from orders
        derived = self._derive_funder_from_orders()
        if derived:
            self._funder_address = derived
            return derived

        # Fall back to EOA (may not have positions)
        logger.warning(
            "Using EOA address for positions (may not find positions). "
            "Set POLYMARKET_FUNDER_ADDRESS in .env for reliable position fetching."
        )
        return self._address

    @_retry_on_connection_error(max_retries=3, base_delay=2.0)
    def fetch_positions(self, market_slugs: list[str] | None = None) -> list[UserPosition]:
        """Fetch user's positions from data-api.

        Uses the funder/proxy wallet address to fetch positions.
        If not configured, attempts to derive it from orders.

        Args:
            market_slugs: Optional list of market slugs to filter by.

        Returns:
            List of UserPosition objects.

        Raises:
            requests.RequestException: If API request fails after retries.

        """
        positions_address = self.get_positions_address()
        url = f"{DATA_API_BASE}/positions"
        params = {"user": positions_address}

        try:
            response = self._http_client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.RequestError as e:
            logger.error(
                "Failed to fetch positions for %s: %s",
                positions_address[:10],
                e,
            )
            raise

        if not isinstance(data, list):
            logger.warning("Unexpected positions response format: %s", type(data))
            return []

        positions = []
        for item in data:
            try:
                position = UserPosition.from_api_response(item)
                # Filter by market slug if specified
                if market_slugs is not None:
                    # TODO: Need to map condition_id to slug
                    pass
                if position.size > 0:
                    positions.append(position)
            except Exception as e:
                logger.warning("Failed to parse position: %s - %s", item, e)

        logger.info(
            "Fetched %d positions for funder %s...%s",
            len(positions),
            positions_address[:6],
            positions_address[-4:],
        )

        return positions

    @_retry_on_connection_error(max_retries=3, base_delay=2.0)
    def fetch_balance(self) -> float:
        """Fetch user's USDC balance from CLOB API.

        Requires Level 2 authentication (private key + API credentials).

        Returns:
            USDC balance as float.

        Raises:
            ValueError: If private key is not configured.
            RuntimeError: If API authentication fails after retries.

        """
        if not self._private_key:
            raise ValueError(
                "Private key required to fetch balance. Set POLYMARKET_PRIVATE_KEY in .env file."
            )

        clob_client = self._get_clob_client()

        try:
            from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

            params = BalanceAllowanceParams(
                asset_type=AssetType.COLLATERAL,
                signature_type=self._signature_type,
            )
            result = clob_client.get_balance_allowance(params)
            logger.debug("Balance API raw response: %s", result)

            if isinstance(result, dict):
                # Balance is in smallest unit, convert to USDC (6 decimals)
                balance_raw = float(result.get("balance", 0))
                balance = balance_raw / 1_000_000  # USDC has 6 decimals

                # Also check allowance - if allowance > 0, user has approved but not deposited
                allowance_raw = float(result.get("allowance", 0))
                allowance = allowance_raw / 1_000_000

                if balance == 0 and allowance > 0:
                    logger.info(
                        "CLOB balance: $0.00 USDC (allowance: $%.2f - not deposited)",
                        allowance,
                    )
                else:
                    logger.info("Fetched balance: $%.2f USDC", balance)
                return balance
            else:
                logger.warning("Unexpected balance response: %s", result)
                return 0.0

        except Exception as e:
            # Check if it's a connection error (will be retried)
            is_connection_error = (
                "PolyApiException" in type(e).__name__
                or "Request exception" in str(e)
                or "Connection" in str(e)
            )
            if is_connection_error:
                # Don't log full trace for connection errors - retry decorator will log
                raise  # Let retry decorator handle it
            # Log full trace only for unexpected errors
            logger.error("Failed to fetch balance: %s", e, exc_info=True)
            raise RuntimeError(f"Failed to fetch balance: {e}") from e

    def _get_clob_client(self) -> Any:
        """Get or create authenticated CLOB client.

        Returns:
            Configured ClobClient with Level 2 auth.

        Raises:
            ValueError: If private key is not configured.
            RuntimeError: If authentication fails.

        """
        if self._clob_client is not None:
            return self._clob_client

        if not self._private_key:
            raise ValueError("Private key required for CLOB authentication")

        try:
            from py_clob_client.client import ClobClient

            # Create client with Level 1 auth (can create API keys)
            client = ClobClient(
                host=CLOB_API_BASE,
                key=self._private_key,
                chain_id=self._chain_id,
                signature_type=self._signature_type,
                funder=self._address if self._signature_type != 0 else None,
            )

            # Derive or create API credentials for Level 2 auth
            api_creds = client.create_or_derive_api_creds()
            if api_creds is None:
                raise RuntimeError("Failed to create/derive API credentials")

            client.set_api_creds(api_creds)
            self._clob_client = client
            self._api_creds = api_creds

            logger.info("CLOB client authenticated successfully")
            return client

        except Exception as e:
            # Check if it's a transient connection error
            error_str = str(e).lower()
            is_connection_error = (
                "connection" in error_str
                or "request exception" in error_str
                or "reset by peer" in error_str
                or "timeout" in error_str
            )
            if is_connection_error:
                # Concise log for transient errors (no stack trace)
                logger.warning("CLOB auth failed (transient): %s", e)
            else:
                # Full trace for unexpected errors
                logger.error("Failed to authenticate CLOB client: %s", e, exc_info=True)
            raise RuntimeError(f"CLOB authentication failed: {e}") from e

    def map_positions_to_contracts(
        self,
        positions: list[UserPosition],
        token_to_contract: dict[str, str],
    ) -> dict[str, float]:
        """Map positions to contract IDs.

        Args:
            positions: List of UserPosition objects.
            token_to_contract: Mapping from token_id to contract_id.

        Returns:
            Dict mapping contract_id to net position (positive=YES, negative=NO).

        """
        result: dict[str, float] = {}

        for pos in positions:
            contract_id = token_to_contract.get(pos.token_id)
            if contract_id is None:
                logger.debug(
                    "Token %s not found in mapping, skipping",
                    pos.token_id[:16] if pos.token_id else "unknown",
                )
                continue

            # Add to existing position (in case of multiple entries)
            current = result.get(contract_id, 0.0)
            result[contract_id] = current + pos.net_position

        return result

    def close(self) -> None:
        """Close HTTP sessions."""
        self._http_client.close()

    def __enter__(self) -> "AuthenticatedPolymarketClient":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Context manager exit."""
        self.close()
