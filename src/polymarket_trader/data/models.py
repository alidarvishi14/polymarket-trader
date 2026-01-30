"""Data models for Polymarket API responses."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class PolymarketToken:
    """A single token (YES or NO) in a Polymarket market.

    Attributes:
        token_id: Unique identifier for the token.
        outcome: 'Yes' or 'No'.
        price: Current price (0-1).

    """

    token_id: str
    outcome: str
    price: float


@dataclass
class PolymarketMarket:
    """A single market (contract) from Polymarket.

    Attributes:
        condition_id: Unique identifier for the market condition.
        question: The market question text.
        slug: URL-friendly identifier.
        end_date: When the market resolves.
        tokens: List of tokens (YES/NO).
        volume: Total trading volume.
        liquidity: Current liquidity.
        active: Whether market is active.
        closed: Whether market is closed.
        outcome_prices: Dict mapping outcome to price.

    """

    condition_id: str
    question: str
    slug: str
    end_date: datetime | None
    tokens: list[PolymarketToken]
    volume: float
    liquidity: float
    active: bool
    closed: bool

    @property
    def yes_price(self) -> float:
        """Get the YES token price."""
        for token in self.tokens:
            if token.outcome.lower() == "yes":
                return token.price
        return 0.0

    @property
    def no_price(self) -> float:
        """Get the NO token price."""
        for token in self.tokens:
            if token.outcome.lower() == "no":
                return token.price
        return 0.0

    @property
    def yes_token_id(self) -> str | None:
        """Get the YES token ID."""
        for token in self.tokens:
            if token.outcome.lower() == "yes":
                return token.token_id
        return None

    @property
    def no_token_id(self) -> str | None:
        """Get the NO token ID."""
        for token in self.tokens:
            if token.outcome.lower() == "no":
                return token.token_id
        return None


@dataclass
class OrderBookLevel:
    """A single level in an order book.

    Attributes:
        price: Price at this level.
        size: Size available at this price.

    """

    price: float
    size: float


@dataclass
class OrderBook:
    """Order book for a token.

    Attributes:
        token_id: Token this order book is for.
        bids: List of bid levels (sorted by price descending).
        asks: List of ask levels (sorted by price ascending).

    """

    token_id: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]

    @property
    def best_bid(self) -> float | None:
        """Get best bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Get best ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        """Get mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        """Get bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class PolymarketEvent:
    """An event containing multiple markets (term structure).

    Attributes:
        slug: Event slug from URL.
        title: Event title.
        description: Event description.
        markets: List of markets in this event.

    """

    slug: str
    title: str
    description: str
    markets: list[PolymarketMarket]


@dataclass
class MarketWithOrderBook:
    """A market combined with its order book data.

    Attributes:
        market: The market data.
        yes_order_book: Order book for YES token.
        no_order_book: Order book for NO token.

    """

    market: PolymarketMarket
    yes_order_book: OrderBook | None
    no_order_book: OrderBook | None

    @property
    def best_yes_bid(self) -> float | None:
        """Best bid for YES token."""
        if self.yes_order_book:
            return self.yes_order_book.best_bid
        return None

    @property
    def best_yes_ask(self) -> float | None:
        """Best ask for YES token."""
        if self.yes_order_book:
            return self.yes_order_book.best_ask
        return None

    @property
    def yes_mid_price(self) -> float | None:
        """Mid price for YES token."""
        if self.yes_order_book:
            return self.yes_order_book.mid_price
        # Fall back to market price
        return self.market.yes_price

    @property
    def yes_spread(self) -> float | None:
        """Spread for YES token."""
        if self.yes_order_book:
            return self.yes_order_book.spread
        return None
