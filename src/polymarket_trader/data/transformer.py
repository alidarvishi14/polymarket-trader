"""Transform Polymarket API data to internal MaturityData format."""

import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from polymarket_trader.data.models import MarketWithOrderBook
from polymarket_trader.models.hazard import MaturityData

logger = logging.getLogger(__name__)

# Default spread to use when order book is unavailable
DEFAULT_SPREAD = 0.02

# Polymarket contracts expire at midnight Eastern Time
EASTERN_TZ = ZoneInfo("America/New_York")

# Month name to number mapping
MONTH_MAP = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def _calculate_time_to_expiry(
    expiry_date: date,
    reference_time: datetime | None = None,
) -> float:
    """Calculate time to expiry in fractional days.

    Polymarket contracts resolve at the END of the expiry day (11:59:59 PM ET).

    Args:
        expiry_date: The date the contract expires.
        reference_time: Current time (timezone-aware). If None, uses now.

    Returns:
        Time to expiry in fractional days. Can be negative if expired.

    """
    if reference_time is None:
        reference_time = datetime.now(EASTERN_TZ)
    elif reference_time.tzinfo is None:
        # Assume ET if no timezone
        reference_time = reference_time.replace(tzinfo=EASTERN_TZ)
    else:
        # Convert to ET
        reference_time = reference_time.astimezone(EASTERN_TZ)

    # Expiry is at the END of the expiry date (11:59:59 PM ET)
    # So "expires January 30" means it resolves at 23:59:59 ET on Jan 30
    expiry_datetime = datetime(
        expiry_date.year,
        expiry_date.month,
        expiry_date.day,
        23,
        59,
        59,
        tzinfo=EASTERN_TZ,
    )

    time_remaining = expiry_datetime - reference_time
    days_remaining = time_remaining.total_seconds() / (24 * 3600)

    return days_remaining


def _parse_date_from_question(question: str, reference_year: int) -> date | None:
    """Extract resolution date from market question.

    Args:
        question: Market question text (e.g., "US strikes Iran by January 28, 2026?")
        reference_year: Year to use if not specified in question.

    Returns:
        Parsed date or None if parsing fails.

    """
    # Pattern: "by January 28, 2026" or "by Jan 28" or "on January 28, 2026"
    patterns = [
        r"(?:by|on|before)\s+(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()
            day = int(match.group(2))
            year = int(match.group(3)) if match.group(3) else reference_year

            month = MONTH_MAP.get(month_str)
            if month:
                try:
                    return date(year, month, day)
                except ValueError:
                    logger.warning("Invalid date: %s %d, %d", month_str, day, year)
                    continue

    return None


def transform_to_maturity_data(
    markets: list[MarketWithOrderBook],
    reference_time: datetime | None = None,
) -> list[MaturityData]:
    """Transform Polymarket API data to MaturityData format.

    Args:
        markets: List of markets with order book data from Polymarket API.
        reference_time: Current time to calculate time-to-expiry from.
            If None, uses current time in Eastern timezone.

    Returns:
        List of MaturityData objects sorted by maturity.

    Raises:
        ValueError: If no valid markets can be transformed.

    """
    if reference_time is None:
        reference_time = datetime.now(EASTERN_TZ)
    elif reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=EASTERN_TZ)

    reference_date = reference_time.date()

    results = []

    for market_data in markets:
        market = market_data.market

        # Skip closed markets
        if market.closed:
            logger.debug("Skipping closed market: %s", market.question)
            continue

        # Try to extract resolution date from question first
        resolution_date = _parse_date_from_question(
            market.question,
            reference_year=reference_date.year,
        )

        if resolution_date is None:
            # Fall back to market end_date
            if market.end_date is None:
                logger.warning("Cannot determine date for: %s", market.question)
                continue
            if isinstance(market.end_date, datetime):
                resolution_date = market.end_date.date()
            else:
                resolution_date = market.end_date

        # Calculate precise time to expiry in fractional days
        days_to_expiry = _calculate_time_to_expiry(resolution_date, reference_time)

        # Skip expired markets (allow very short-dated but not expired)
        if days_to_expiry <= 0:
            logger.debug(
                "Skipping expired market: %s (DTE=%.2f)",
                market.question,
                days_to_expiry,
            )
            continue

        # Skip markets expiring in less than 1 hour (too risky to trade)
        if days_to_expiry < 1 / 24:  # Less than 1 hour
            logger.debug(
                "Skipping market expiring in < 1 hour: %s (DTE=%.3f)",
                market.question,
                days_to_expiry,
            )
            continue

        # Get prices from order book or fall back to market prices
        if market_data.yes_order_book and market_data.yes_order_book.mid_price is not None:
            mid_price = market_data.yes_order_book.mid_price
            bid_price = market_data.yes_order_book.best_bid or mid_price - DEFAULT_SPREAD / 2
            ask_price = market_data.yes_order_book.best_ask or mid_price + DEFAULT_SPREAD / 2
        else:
            # Fall back to market price with estimated spread
            mid_price = market.yes_price
            bid_price = max(0.001, mid_price - DEFAULT_SPREAD / 2)
            ask_price = min(0.999, mid_price + DEFAULT_SPREAD / 2)

        # Ensure prices are valid
        if not (0 < mid_price < 1):
            logger.warning(
                "Invalid mid price %.4f for %s, skipping",
                mid_price,
                market.question,
            )
            continue

        # Generate a contract ID from the question or resolution date
        contract_id = _generate_contract_id(market.question, resolution_date)

        try:
            maturity_data = MaturityData(
                maturity=float(days_to_expiry),
                market_price=mid_price,
                bid_price=bid_price,
                ask_price=ask_price,
                volume=market.volume,
                contract_id=contract_id,
                yes_token_id=market.yes_token_id or "",
                no_token_id=market.no_token_id or "",
            )
            results.append(maturity_data)

            logger.debug(
                "Transformed: %s -> DTE=%d, price=%.2f%%",
                contract_id,
                days_to_expiry,
                mid_price * 100,
            )

        except ValueError:
            logger.warning(
                "Failed to create MaturityData for %s",
                market.question,
                exc_info=True,
            )
            continue

    if not results:
        raise ValueError("No valid markets could be transformed")

    # Sort by maturity
    results.sort(key=lambda x: x.maturity)

    # Deduplicate: if multiple contracts have same maturity, keep highest volume
    deduplicated = []
    seen_maturities: dict[float, MaturityData] = {}

    for data in results:
        mat = data.maturity
        if mat not in seen_maturities:
            seen_maturities[mat] = data
        elif data.volume > seen_maturities[mat].volume:
            # Replace with higher volume contract
            seen_maturities[mat] = data

    deduplicated = sorted(seen_maturities.values(), key=lambda x: x.maturity)

    if len(deduplicated) < len(results):
        logger.info(
            "Deduplicated %d -> %d contracts (removed same-maturity duplicates)",
            len(results),
            len(deduplicated),
        )

    logger.info(
        "Transformed %d markets to MaturityData (DTE range: %.2f - %.2f days)",
        len(deduplicated),
        deduplicated[0].maturity,
        deduplicated[-1].maturity,
    )

    return deduplicated


def _generate_contract_id(question: str, end_date: date) -> str:
    """Generate a short contract ID from question and end date.

    Args:
        question: Market question text.
        end_date: Contract end date.

    Returns:
        Short contract identifier (e.g., 'jan28', 'feb06').

    """
    # Try to extract date from question first
    # Common patterns: "by January 28", "by Jan 28, 2026", etc.
    date_patterns = [
        r"(?:by\s+)?(\w+)\s+(\d{1,2})(?:,?\s+\d{4})?$",  # "January 28" or "Jan 28, 2026"
        r"(\w+)\s+(\d{1,2})(?:st|nd|rd|th)?",  # "January 28th"
    ]

    for pattern in date_patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            month_str = match.group(1).lower()[:3]
            day_str = match.group(2)
            return f"{month_str}{day_str}"

    # Fall back to end_date
    return end_date.strftime("%b%d").lower()


def extract_event_title(markets: list[MarketWithOrderBook]) -> str:
    """Extract the event title from market questions.

    Args:
        markets: List of markets.

    Returns:
        Extracted event title or "Unknown Event".

    """
    if not markets:
        return "Unknown Event"

    # Get first market question
    question = markets[0].market.question

    # Common patterns to extract title:
    # "US strikes Iran by January 28?" -> "US strikes Iran"
    # "Will X happen by Y?" -> "Will X happen"

    # Remove date suffix
    patterns = [
        r"(.+?)\s+by\s+\w+\s+\d{1,2}.*\??$",  # "... by January 28?"
        r"(.+?)\s+on\s+\w+\s+\d{1,2}.*\??$",  # "... on January 28?"
        r"(.+?)\s+before\s+\w+\s+\d{1,2}.*\??$",  # "... before January 28?"
    ]

    for pattern in patterns:
        match = re.match(pattern, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Just return the question without trailing punctuation
    return question.rstrip("?").strip()
