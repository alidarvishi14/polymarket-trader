"""Microstructure constraints for realistic trading.

This module handles the realities of trading on Polymarket:
1. Bid/ask spreads (not mid prices)
2. Order book depth limits
3. Transaction costs and fees
4. Minimum edge requirements
5. Position concentration limits
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrderBookLevel:
    """A single level in an order book.

    Attributes:
        price: Price at this level.
        quantity: Available quantity at this price.
        side: 'bid' or 'ask'.

    """

    price: float
    quantity: float
    side: str

    def __post_init__(self) -> None:
        """Validate order book level."""
        if not 0 <= self.price <= 1:
            raise ValueError(f"Price must be in [0, 1], got {self.price}")
        if self.quantity < 0:
            raise ValueError(f"Quantity must be non-negative, got {self.quantity}")
        if self.side not in ("bid", "ask"):
            raise ValueError(f"Side must be 'bid' or 'ask', got {self.side}")


@dataclass(frozen=True)
class OrderBook:
    """Order book for a contract.

    Attributes:
        contract_id: Contract identifier.
        bids: List of bid levels (sorted by price, descending).
        asks: List of ask levels (sorted by price, ascending).

    """

    contract_id: str
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
        """Get mid price (average of best bid and ask)."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        """Get bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def depth_at_price(self, price: float, side: str) -> float:
        """Get available depth at or better than a price.

        Args:
            price: Price threshold.
            side: 'bid' (for selling) or 'ask' (for buying).

        Returns:
            Total quantity available at or better than the price.

        """
        total = 0.0
        levels = self.bids if side == "bid" else self.asks

        for level in levels:
            if side == "bid" and level.price >= price or side == "ask" and level.price <= price:
                total += level.quantity

        return total


@dataclass
class TransactionCostModel:
    """Model for transaction costs on Polymarket.

    Attributes:
        maker_fee_bps: Maker fee in basis points.
        taker_fee_bps: Taker fee in basis points.
        gas_cost_usd: Estimated gas cost per transaction in USD.
        slippage_factor: Multiplier for estimated slippage (typically 0.5-2.0).

    """

    maker_fee_bps: float
    taker_fee_bps: float
    gas_cost_usd: float
    slippage_factor: float

    def __post_init__(self) -> None:
        """Validate cost model."""
        if self.maker_fee_bps < 0:
            raise ValueError(f"maker_fee_bps must be non-negative, got {self.maker_fee_bps}")
        if self.taker_fee_bps < 0:
            raise ValueError(f"taker_fee_bps must be non-negative, got {self.taker_fee_bps}")
        if self.gas_cost_usd < 0:
            raise ValueError(f"gas_cost_usd must be non-negative, got {self.gas_cost_usd}")
        if self.slippage_factor < 0:
            raise ValueError(f"slippage_factor must be non-negative, got {self.slippage_factor}")

    def estimate_cost(
        self,
        quantity: float,
        price: float,
        is_maker: bool,
        order_book: OrderBook | None = None,
    ) -> float:
        """Estimate total transaction cost.

        Args:
            quantity: Size of the trade.
            price: Expected execution price.
            is_maker: Whether this is a maker (limit) order.
            order_book: Optional order book for slippage estimation.

        Returns:
            Estimated total cost in USD.

        """
        notional = quantity * price

        # Fee cost
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee_cost = notional * fee_bps / 10000

        # Slippage cost (only for taker orders)
        slippage_cost = 0.0
        if not is_maker and order_book is not None:
            spread = order_book.spread
            if spread is not None:
                slippage_cost = notional * (spread / 2) * self.slippage_factor

        total = fee_cost + slippage_cost + self.gas_cost_usd

        logger.debug(
            "Cost estimate: notional=%.2f, fees=%.4f, slippage=%.4f, gas=%.4f, total=%.4f",
            notional,
            fee_cost,
            slippage_cost,
            self.gas_cost_usd,
            total,
        )

        return total

    def min_edge_for_profitability(
        self,
        price: float,
        is_maker: bool,
        spread: float,
    ) -> float:
        """Calculate minimum edge required for profitable trade.

        Args:
            price: Expected execution price.
            is_maker: Whether this is a maker order.
            spread: Bid-ask spread.

        Returns:
            Minimum edge required (in price terms, not percentage).

        """
        # Fee drag
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee_drag = price * fee_bps / 10000

        # Spread cost (half spread for entry, half for exit)
        spread_cost = spread

        # Need to overcome these costs twice (entry + exit)
        total_drag = 2 * fee_drag + spread_cost

        # Add buffer for model uncertainty
        buffer = 0.001  # 10 bps

        return total_drag + buffer


@dataclass
class MarketConstraints:
    """Constraints derived from market microstructure.

    Attributes:
        contract_id: Contract identifier.
        max_position_yes: Maximum YES position (from order book depth).
        max_position_no: Maximum NO position.
        effective_bid: Bid price accounting for depth.
        effective_ask: Ask price accounting for depth.
        min_edge_required: Minimum edge for profitable trade.
        liquidity_score: Relative liquidity score (0-1).

    """

    contract_id: str
    max_position_yes: float
    max_position_no: float
    effective_bid: float
    effective_ask: float
    min_edge_required: float
    liquidity_score: float


class MarketConstraintsBuilder:
    """Builder for market constraints from order book data.

    Analyzes order books to determine realistic position limits
    and minimum edge requirements.
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        max_market_impact_pct: float,
        min_liquidity_threshold: float,
    ) -> None:
        """Initialize the constraints builder.

        Args:
            cost_model: Transaction cost model.
            max_market_impact_pct: Maximum acceptable market impact (e.g., 0.02 = 2%).
            min_liquidity_threshold: Minimum depth required to consider a level liquid.

        """
        if max_market_impact_pct <= 0 or max_market_impact_pct >= 1:
            raise ValueError(
                f"max_market_impact_pct must be in (0, 1), got {max_market_impact_pct}"
            )
        if min_liquidity_threshold < 0:
            raise ValueError(
                f"min_liquidity_threshold must be non-negative, got {min_liquidity_threshold}"
            )

        self._cost_model = cost_model
        self._max_impact = max_market_impact_pct
        self._min_liquidity = min_liquidity_threshold

        logger.info(
            "MarketConstraintsBuilder initialized: max_impact=%.2f%%, min_liquidity=%.2f",
            max_market_impact_pct * 100,
            min_liquidity_threshold,
        )

    def build(self, order_book: OrderBook) -> MarketConstraints:
        """Build constraints from an order book.

        Args:
            order_book: Order book to analyze.

        Returns:
            MarketConstraints for the contract.

        Raises:
            ValueError: If order book is invalid or has no liquidity.

        """
        if order_book.best_bid is None or order_book.best_ask is None:
            raise ValueError(f"Order book for {order_book.contract_id} has no valid quotes")

        best_bid = order_book.best_bid
        best_ask = order_book.best_ask
        spread = order_book.spread

        if spread is None:
            raise ValueError(f"Cannot compute spread for {order_book.contract_id}")

        # Calculate max positions based on depth at acceptable impact
        max_bid_price = best_bid * (1 - self._max_impact)
        max_ask_price = best_ask * (1 + self._max_impact)

        max_yes = order_book.depth_at_price(max_ask_price, "ask")
        max_no = order_book.depth_at_price(max_bid_price, "bid")

        # Calculate effective prices (volume-weighted average for reasonable size)
        effective_bid = self._calculate_vwap(order_book.bids, max_no * 0.5)
        effective_ask = self._calculate_vwap(order_book.asks, max_yes * 0.5)

        # Calculate minimum edge
        min_edge = self._cost_model.min_edge_for_profitability(
            price=(best_bid + best_ask) / 2,
            is_maker=False,  # Conservative: assume taker
            spread=spread,
        )

        # Calculate liquidity score
        total_depth = sum(level.quantity for level in order_book.bids)
        total_depth += sum(level.quantity for level in order_book.asks)
        liquidity_score = min(total_depth / (2 * self._min_liquidity), 1.0)

        constraints = MarketConstraints(
            contract_id=order_book.contract_id,
            max_position_yes=max_yes,
            max_position_no=max_no,
            effective_bid=effective_bid,
            effective_ask=effective_ask,
            min_edge_required=min_edge,
            liquidity_score=liquidity_score,
        )

        logger.info(
            "Built constraints for %s: max_yes=%.2f, max_no=%.2f, min_edge=%.4f, liq_score=%.2f",
            order_book.contract_id,
            max_yes,
            max_no,
            min_edge,
            liquidity_score,
        )

        return constraints

    def build_batch(
        self,
        order_books: Sequence[OrderBook],
    ) -> list[MarketConstraints]:
        """Build constraints for multiple order books.

        Args:
            order_books: Sequence of order books.

        Returns:
            List of constraints (in same order as input).

        Raises:
            ValueError: If any order book is invalid.

        """
        return [self.build(ob) for ob in order_books]

    @staticmethod
    def _calculate_vwap(
        levels: Sequence[OrderBookLevel],
        target_quantity: float,
    ) -> float:
        """Calculate volume-weighted average price for a target quantity.

        Args:
            levels: Order book levels (sorted by price).
            target_quantity: Target quantity to fill.

        Returns:
            VWAP for the target quantity.

        """
        if not levels or target_quantity <= 0:
            return levels[0].price if levels else 0.0

        filled = 0.0
        cost = 0.0

        for level in levels:
            available = min(level.quantity, target_quantity - filled)
            cost += available * level.price
            filled += available

            if filled >= target_quantity:
                break

        return cost / filled if filled > 0 else levels[0].price


def filter_by_liquidity(
    constraints: Sequence[MarketConstraints],
    min_score: float,
) -> list[MarketConstraints]:
    """Filter constraints by minimum liquidity score.

    Args:
        constraints: Sequence of market constraints.
        min_score: Minimum liquidity score to include (0-1).

    Returns:
        Filtered list of constraints meeting the threshold.

    """
    filtered = [c for c in constraints if c.liquidity_score >= min_score]
    logger.info(
        "Liquidity filter: %d/%d contracts passed (min_score=%.2f)",
        len(filtered),
        len(constraints),
        min_score,
    )
    return filtered


def compute_concentration_limits(
    constraints: Sequence[MarketConstraints],
    total_budget: float,
    max_concentration_pct: float,
) -> NDArray[np.float64]:
    """Compute per-contract position limits based on concentration.

    Args:
        constraints: Market constraints for each contract.
        total_budget: Total portfolio budget.
        max_concentration_pct: Maximum percentage of budget per contract.

    Returns:
        Array of maximum position values per contract.

    """
    max_per_contract = total_budget * max_concentration_pct

    limits = []
    for c in constraints:
        # Take minimum of concentration limit and liquidity limit
        yes_limit = min(max_per_contract / c.effective_ask, c.max_position_yes)
        limits.append(yes_limit)

    return np.array(limits, dtype=np.float64)
