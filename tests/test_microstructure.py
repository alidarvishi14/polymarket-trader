"""Tests for the microstructure constraints module."""

import pytest

from polymarket_trader.microstructure.constraints import (
    MarketConstraints,
    MarketConstraintsBuilder,
    OrderBook,
    OrderBookLevel,
    TransactionCostModel,
    compute_concentration_limits,
    filter_by_liquidity,
)


class TestOrderBookLevel:
    """Tests for OrderBookLevel dataclass."""

    def test_valid_level(self) -> None:
        """Test creation of valid order book level."""
        level = OrderBookLevel(price=0.5, quantity=100.0, side="bid")
        assert level.price == 0.5
        assert level.quantity == 100.0
        assert level.side == "bid"

    def test_invalid_price_raises(self) -> None:
        """Test that price outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Price must be in"):
            OrderBookLevel(price=1.5, quantity=100.0, side="bid")

    def test_negative_quantity_raises(self) -> None:
        """Test that negative quantity raises ValueError."""
        with pytest.raises(ValueError, match="Quantity must be non-negative"):
            OrderBookLevel(price=0.5, quantity=-10.0, side="bid")

    def test_invalid_side_raises(self) -> None:
        """Test that invalid side raises ValueError."""
        with pytest.raises(ValueError, match="Side must be"):
            OrderBookLevel(price=0.5, quantity=100.0, side="buy")


class TestOrderBook:
    """Tests for OrderBook dataclass."""

    @pytest.fixture
    def sample_book(self) -> OrderBook:
        """Create a sample order book."""
        bids = [
            OrderBookLevel(price=0.49, quantity=100.0, side="bid"),
            OrderBookLevel(price=0.48, quantity=150.0, side="bid"),
            OrderBookLevel(price=0.47, quantity=200.0, side="bid"),
        ]
        asks = [
            OrderBookLevel(price=0.51, quantity=100.0, side="ask"),
            OrderBookLevel(price=0.52, quantity=150.0, side="ask"),
            OrderBookLevel(price=0.53, quantity=200.0, side="ask"),
        ]
        return OrderBook(contract_id="test", bids=bids, asks=asks)

    def test_best_bid(self, sample_book: OrderBook) -> None:
        """Test best bid price."""
        assert sample_book.best_bid == 0.49

    def test_best_ask(self, sample_book: OrderBook) -> None:
        """Test best ask price."""
        assert sample_book.best_ask == 0.51

    def test_mid_price(self, sample_book: OrderBook) -> None:
        """Test mid price calculation."""
        assert sample_book.mid_price == pytest.approx(0.50)

    def test_spread(self, sample_book: OrderBook) -> None:
        """Test spread calculation."""
        assert sample_book.spread == pytest.approx(0.02)

    def test_depth_at_price_bid(self, sample_book: OrderBook) -> None:
        """Test depth calculation at bid price."""
        # All bids at or above 0.47
        depth = sample_book.depth_at_price(0.47, "bid")
        assert depth == 450.0  # 100 + 150 + 200

    def test_depth_at_price_ask(self, sample_book: OrderBook) -> None:
        """Test depth calculation at ask price."""
        # All asks at or below 0.52
        depth = sample_book.depth_at_price(0.52, "ask")
        assert depth == 250.0  # 100 + 150


class TestTransactionCostModel:
    """Tests for TransactionCostModel dataclass."""

    @pytest.fixture
    def cost_model(self) -> TransactionCostModel:
        """Create a sample cost model."""
        return TransactionCostModel(
            maker_fee_bps=0.0,
            taker_fee_bps=10.0,
            gas_cost_usd=0.10,
            slippage_factor=1.0,
        )

    def test_estimate_cost_basic(self, cost_model: TransactionCostModel) -> None:
        """Test basic cost estimation."""
        cost = cost_model.estimate_cost(
            quantity=100.0,
            price=0.5,
            is_maker=False,
        )
        # 100 * 0.5 * 10 / 10000 + 0.10 = 0.05 + 0.10 = 0.15
        assert cost == pytest.approx(0.15)

    def test_estimate_cost_with_slippage(
        self,
        cost_model: TransactionCostModel,
    ) -> None:
        """Test cost estimation with order book slippage."""
        book = OrderBook(
            contract_id="test",
            bids=[OrderBookLevel(price=0.49, quantity=100.0, side="bid")],
            asks=[OrderBookLevel(price=0.51, quantity=100.0, side="ask")],
        )

        cost = cost_model.estimate_cost(
            quantity=100.0,
            price=0.5,
            is_maker=False,
            order_book=book,
        )

        # Should include slippage estimate based on spread
        assert cost > 0.15  # More than fee + gas

    def test_min_edge_for_profitability(self, cost_model: TransactionCostModel) -> None:
        """Test minimum edge calculation."""
        min_edge = cost_model.min_edge_for_profitability(
            price=0.5,
            is_maker=False,
            spread=0.02,
        )

        # Should cover fees (both ways) + spread + buffer
        assert min_edge > 0.02  # At least spread


class TestMarketConstraintsBuilder:
    """Tests for MarketConstraintsBuilder class."""

    @pytest.fixture
    def cost_model(self) -> TransactionCostModel:
        """Create a cost model."""
        return TransactionCostModel(
            maker_fee_bps=0.0,
            taker_fee_bps=10.0,
            gas_cost_usd=0.10,
            slippage_factor=1.0,
        )

    @pytest.fixture
    def builder(self, cost_model: TransactionCostModel) -> MarketConstraintsBuilder:
        """Create a constraints builder."""
        return MarketConstraintsBuilder(
            cost_model=cost_model,
            max_market_impact_pct=0.02,
            min_liquidity_threshold=100.0,
        )

    @pytest.fixture
    def sample_book(self) -> OrderBook:
        """Create a sample order book."""
        bids = [
            OrderBookLevel(price=0.49, quantity=100.0, side="bid"),
            OrderBookLevel(price=0.48, quantity=150.0, side="bid"),
        ]
        asks = [
            OrderBookLevel(price=0.51, quantity=100.0, side="ask"),
            OrderBookLevel(price=0.52, quantity=150.0, side="ask"),
        ]
        return OrderBook(contract_id="test", bids=bids, asks=asks)

    def test_build_returns_constraints(
        self,
        builder: MarketConstraintsBuilder,
        sample_book: OrderBook,
    ) -> None:
        """Test that build returns MarketConstraints."""
        constraints = builder.build(sample_book)
        assert isinstance(constraints, MarketConstraints)
        assert constraints.contract_id == "test"

    def test_build_max_positions_computed(
        self,
        builder: MarketConstraintsBuilder,
        sample_book: OrderBook,
    ) -> None:
        """Test that max positions are computed based on depth."""
        constraints = builder.build(sample_book)

        # Max positions should be based on depth within impact threshold
        assert constraints.max_position_yes > 0
        assert constraints.max_position_no > 0

    def test_build_effective_prices(
        self,
        builder: MarketConstraintsBuilder,
        sample_book: OrderBook,
    ) -> None:
        """Test that effective prices are computed."""
        constraints = builder.build(sample_book)

        # Effective bid should be <= best bid
        assert constraints.effective_bid <= 0.49
        # Effective ask should be >= best ask
        assert constraints.effective_ask >= 0.51

    def test_build_liquidity_score(
        self,
        builder: MarketConstraintsBuilder,
        sample_book: OrderBook,
    ) -> None:
        """Test that liquidity score is computed."""
        constraints = builder.build(sample_book)

        # Score should be in [0, 1]
        assert 0 <= constraints.liquidity_score <= 1

    def test_build_empty_book_raises(
        self,
        builder: MarketConstraintsBuilder,
    ) -> None:
        """Test that empty order book raises ValueError."""
        empty_book = OrderBook(contract_id="empty", bids=[], asks=[])
        with pytest.raises(ValueError, match="no valid quotes"):
            builder.build(empty_book)

    def test_build_batch(
        self,
        builder: MarketConstraintsBuilder,
        sample_book: OrderBook,
    ) -> None:
        """Test batch constraint building."""
        book2 = OrderBook(
            contract_id="test2",
            bids=[OrderBookLevel(price=0.50, quantity=200.0, side="bid")],
            asks=[OrderBookLevel(price=0.52, quantity=200.0, side="ask")],
        )

        constraints = builder.build_batch([sample_book, book2])

        assert len(constraints) == 2
        assert constraints[0].contract_id == "test"
        assert constraints[1].contract_id == "test2"


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def sample_constraints(self) -> list[MarketConstraints]:
        """Create sample constraints."""
        return [
            MarketConstraints(
                contract_id="c1",
                max_position_yes=100.0,
                max_position_no=100.0,
                effective_bid=0.48,
                effective_ask=0.52,
                min_edge_required=0.02,
                liquidity_score=0.9,
            ),
            MarketConstraints(
                contract_id="c2",
                max_position_yes=50.0,
                max_position_no=50.0,
                effective_bid=0.30,
                effective_ask=0.32,
                min_edge_required=0.03,
                liquidity_score=0.5,
            ),
            MarketConstraints(
                contract_id="c3",
                max_position_yes=20.0,
                max_position_no=20.0,
                effective_bid=0.70,
                effective_ask=0.72,
                min_edge_required=0.04,
                liquidity_score=0.2,
            ),
        ]

    def test_filter_by_liquidity(
        self,
        sample_constraints: list[MarketConstraints],
    ) -> None:
        """Test filtering by liquidity score."""
        filtered = filter_by_liquidity(sample_constraints, min_score=0.4)

        assert len(filtered) == 2
        assert all(c.liquidity_score >= 0.4 for c in filtered)

    def test_filter_by_liquidity_all_pass(
        self,
        sample_constraints: list[MarketConstraints],
    ) -> None:
        """Test filtering when all pass threshold."""
        filtered = filter_by_liquidity(sample_constraints, min_score=0.1)
        assert len(filtered) == 3

    def test_filter_by_liquidity_none_pass(
        self,
        sample_constraints: list[MarketConstraints],
    ) -> None:
        """Test filtering when none pass threshold."""
        filtered = filter_by_liquidity(sample_constraints, min_score=0.95)
        assert len(filtered) == 0

    def test_compute_concentration_limits(
        self,
        sample_constraints: list[MarketConstraints],
    ) -> None:
        """Test concentration limit computation."""
        limits = compute_concentration_limits(
            constraints=sample_constraints,
            total_budget=1000.0,
            max_concentration_pct=0.25,
        )

        assert len(limits) == 3
        # Each limit should be <= min(concentration limit, liquidity limit)
        for i, c in enumerate(sample_constraints):
            max_conc = 1000.0 * 0.25 / c.effective_ask
            assert limits[i] <= min(max_conc, c.max_position_yes) + 1e-6
