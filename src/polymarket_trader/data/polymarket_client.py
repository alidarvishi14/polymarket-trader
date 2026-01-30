"""Client for fetching data from Polymarket APIs."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from polymarket_trader.data.models import (
    MarketWithOrderBook,
    OrderBook,
    OrderBookLevel,
    PolymarketMarket,
    PolymarketToken,
)

logger = logging.getLogger(__name__)

# Polymarket API endpoints
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "polymarket-trader"


class PolymarketClient:
    """Client for fetching data from Polymarket.

    Uses the public Gamma API for event/market data and CLOB API for order books.
    Caches event metadata locally to handle API failures gracefully.
    """

    def __init__(
        self,
        timeout: float = 30.0,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            timeout: Request timeout in seconds.
            cache_dir: Directory for caching event data. Defaults to ~/.cache/polymarket-trader.

        """
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "polymarket-trader/0.1.0",
            }
        )
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_event(self, url_or_slug: str) -> list[MarketWithOrderBook]:
        """Fetch all markets for an event by URL or slug.

        Args:
            url_or_slug: Either a full Polymarket URL or just the event slug.
                Example URL: https://polymarket.com/event/us-strikes-iran-by
                Example slug: us-strikes-iran-by

        Returns:
            List of MarketWithOrderBook objects for each contract in the event.

        Raises:
            ValueError: If the event cannot be found or parsed.
            requests.RequestException: If API request fails.

        """
        slug = self._extract_slug(url_or_slug)

        # Fetch event data from Gamma API (with cache fallback)
        event_data = self._fetch_event_data_with_cache(slug)

        # Parse markets from the embedded data
        markets = self._parse_embedded_markets(event_data)

        if not markets:
            raise ValueError(f"No markets found for event: {slug}")

        # Fetch order books for each market
        results = []
        for market in markets:
            yes_book = None
            no_book = None

            if market.yes_token_id:
                try:
                    yes_book = self._fetch_order_book(market.yes_token_id)
                except Exception as e:
                    logger.error(
                        "Failed to fetch YES order book for %s: %s - %s",
                        market.question[:30] if market.question else market.condition_id[:16],
                        type(e).__name__,
                        str(e)[:100],
                    )

            if market.no_token_id:
                try:
                    no_book = self._fetch_order_book(market.no_token_id)
                except Exception as e:
                    logger.error(
                        "Failed to fetch NO order book for %s: %s - %s",
                        market.question[:30] if market.question else market.condition_id[:16],
                        type(e).__name__,
                        str(e)[:100],
                    )

            results.append(
                MarketWithOrderBook(
                    market=market,
                    yes_order_book=yes_book,
                    no_order_book=no_book,
                )
            )

        return results

    def _extract_slug(self, url_or_slug: str) -> str:
        """Extract event slug from URL or return slug directly.

        Args:
            url_or_slug: URL or slug string.

        Returns:
            Event slug.

        """
        # Check if it's a URL
        if url_or_slug.startswith("http"):
            parsed = urlparse(url_or_slug)
            # URL format: /event/slug-name
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] == "event":
                return path_parts[1]
            raise ValueError(f"Cannot extract slug from URL: {url_or_slug}")

        # Already a slug
        return url_or_slug

    def _get_cache_path(self, slug: str) -> Path:
        """Get the cache file path for an event slug.

        Args:
            slug: Event slug.

        Returns:
            Path to the cache file.

        """
        return self._cache_dir / f"event_{slug}.json"

    def _load_from_cache(self, slug: str) -> dict | None:
        """Load event data from cache if available.

        Args:
            slug: Event slug.

        Returns:
            Cached event data or None if not available.

        """
        cache_path = self._get_cache_path(slug)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Failed to load cache for %s: %s", slug, e)
        return None

    def _save_to_cache(self, slug: str, data: dict) -> None:
        """Save event data to cache.

        Args:
            slug: Event slug.
            data: Event data to cache.

        """
        cache_path = self._get_cache_path(slug)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error("Failed to save cache for %s: %s", slug, e)

    def _fetch_event_data_with_cache(self, slug: str) -> dict:
        """Fetch event data from API with cache fallback.

        Tries to fetch from the API first. On success, updates the cache.
        On failure, falls back to cached data if available.

        Args:
            slug: Event slug.

        Returns:
            Event data dictionary.

        Raises:
            ValueError: If event not found and no cache available.
            requests.RequestException: If API fails and no cache available.

        """
        try:
            event_data = self._fetch_event_data(slug)
            # Success - update cache
            self._save_to_cache(slug, event_data)
            return event_data
        except Exception as e:
            # API failed - try cache
            cached_data = self._load_from_cache(slug)
            if cached_data:
                logger.warning(
                    "API request failed (%s: %s), using cached event data for %s",
                    type(e).__name__,
                    str(e)[:50],
                    slug,
                )
                return cached_data
            # No cache available - re-raise
            raise

    def _fetch_event_data(self, slug: str) -> dict:
        """Fetch event metadata from Gamma API.

        Args:
            slug: Event slug.

        Returns:
            Event data dictionary.

        Raises:
            ValueError: If event not found.
            requests.RequestException: If API request fails.

        """
        url = f"{GAMMA_API_BASE}/events"
        params = {"slug": slug}

        response = self._session.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()

        data = response.json()

        # API returns a list, find our event
        if isinstance(data, list):
            for event in data:
                if event.get("slug") == slug:
                    return event
            # If not found by exact match, return first result
            if data:
                return data[0]

        raise ValueError(f"Event not found: {slug}")

    def _parse_embedded_markets(self, event_data: dict) -> list[PolymarketMarket]:
        """Parse markets embedded in event data.

        Args:
            event_data: Event data from API containing 'markets' array.

        Returns:
            List of PolymarketMarket objects.

        """
        embedded = event_data.get("markets", [])
        if not embedded:
            return []

        markets = []
        for item in embedded:
            market = self._parse_market(item)
            if market and not market.closed:
                markets.append(market)

        # Sort by end date
        markets.sort(key=lambda m: m.end_date or datetime.max.replace(tzinfo=UTC))

        return markets

    def _parse_market(self, data: dict) -> PolymarketMarket | None:
        """Parse market data from API response.

        Args:
            data: Raw market data from API.

        Returns:
            PolymarketMarket or None if parsing fails.

        """
        import json as json_module

        try:
            # Parse end date
            end_date = None
            if data.get("endDate"):
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))

            # Parse tokens
            tokens = []
            if data.get("tokens"):
                for token_data in data["tokens"]:
                    tokens.append(
                        PolymarketToken(
                            token_id=token_data.get("token_id", ""),
                            outcome=token_data.get("outcome", ""),
                            price=float(token_data.get("price", 0)),
                        )
                    )

            # Handle outcomePrices - can be JSON array string or comma-separated
            if not tokens and data.get("outcomePrices"):
                outcome_prices_raw = data["outcomePrices"]

                # Try parsing as JSON array first (e.g., '["0.095", "0.905"]')
                if outcome_prices_raw.startswith("["):
                    try:
                        prices = json_module.loads(outcome_prices_raw)
                    except json_module.JSONDecodeError:
                        prices = outcome_prices_raw.strip("[]").replace('"', "").split(",")
                else:
                    # Comma-separated format
                    prices = outcome_prices_raw.split(",")

                if len(prices) >= 2:
                    # Get token IDs if available
                    clob_token_ids = data.get("clobTokenIds", "")
                    if clob_token_ids.startswith("["):
                        try:
                            token_ids = json_module.loads(clob_token_ids)
                        except json_module.JSONDecodeError:
                            token_ids = ["", ""]
                    elif clob_token_ids:
                        token_ids = clob_token_ids.split(",")
                    else:
                        token_ids = ["", ""]

                    tokens = [
                        PolymarketToken(
                            token_id=token_ids[0] if len(token_ids) > 0 else "",
                            outcome="Yes",
                            price=float(prices[0]),
                        ),
                        PolymarketToken(
                            token_id=token_ids[1] if len(token_ids) > 1 else "",
                            outcome="No",
                            price=float(prices[1]),
                        ),
                    ]

            return PolymarketMarket(
                condition_id=data.get("conditionId", data.get("condition_id", "")),
                question=data.get("question", ""),
                slug=data.get("slug", ""),
                end_date=end_date,
                tokens=tokens,
                volume=float(data.get("volume", 0) or 0),
                liquidity=float(data.get("liquidity", 0) or 0),
                active=data.get("active", True),
                closed=data.get("closed", False),
            )

        except Exception:
            logger.warning("Failed to parse market: %s", data, exc_info=True)
            return None

    def _fetch_order_book(self, token_id: str, max_retries: int = 3) -> OrderBook:
        """Fetch order book for a token from CLOB API.

        Args:
            token_id: Token ID to fetch order book for.
            max_retries: Number of retry attempts on connection error.

        Returns:
            OrderBook object.

        Raises:
            requests.RequestException: If API request fails after all retries.

        """
        url = f"{CLOB_API_BASE}/book"
        params = {"token_id": token_id}

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, params=params, timeout=self._timeout)
                response.raise_for_status()
                data = response.json()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                last_error = e
                if attempt < max_retries - 1:
                    import time

                    time.sleep(0.5 * (attempt + 1))  # 0.5s, 1s, 1.5s backoff
                    continue
                raise
        else:
            # All retries exhausted
            raise last_error

        bids = []
        asks = []

        # Parse bids
        for bid in data.get("bids", []):
            bids.append(
                OrderBookLevel(
                    price=float(bid.get("price", 0)),
                    size=float(bid.get("size", 0)),
                )
            )

        # Parse asks
        for ask in data.get("asks", []):
            asks.append(
                OrderBookLevel(
                    price=float(ask.get("price", 0)),
                    size=float(ask.get("size", 0)),
                )
            )

        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        return OrderBook(token_id=token_id, bids=bids, asks=asks)

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> "PolymarketClient":
        """Context manager entry."""
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: object
    ) -> None:
        """Context manager exit."""
        self.close()
