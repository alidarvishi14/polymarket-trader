"""Tests for the hazard rate model."""

import numpy as np
import pytest

from polymarket_trader.models.hazard import HazardRateModel, MaturityData


class TestMaturityData:
    """Tests for MaturityData dataclass."""

    def test_valid_maturity_data(self) -> None:
        """Test creation of valid maturity data."""
        data = MaturityData(
            maturity=30.0,
            market_price=0.5,
            bid_price=0.49,
            ask_price=0.51,
            volume=1000.0,
            contract_id="test-contract",
        )
        assert data.maturity == 30.0
        assert data.market_price == 0.5
        assert data.bid_price == 0.49
        assert data.ask_price == 0.51
        assert data.volume == 1000.0
        assert data.contract_id == "test-contract"

    def test_invalid_maturity_raises(self) -> None:
        """Test that non-positive maturity raises ValueError."""
        with pytest.raises(ValueError, match="Maturity must be positive"):
            MaturityData(
                maturity=0.0,
                market_price=0.5,
                bid_price=0.49,
                ask_price=0.51,
                volume=1000.0,
                contract_id="test",
            )

    def test_invalid_price_raises(self) -> None:
        """Test that price outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="Market price must be in"):
            MaturityData(
                maturity=30.0,
                market_price=1.5,
                bid_price=0.49,
                ask_price=0.51,
                volume=1000.0,
                contract_id="test",
            )

    def test_invalid_bid_ask_raises(self) -> None:
        """Test that bid > ask raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bid/ask"):
            MaturityData(
                maturity=30.0,
                market_price=0.5,
                bid_price=0.55,
                ask_price=0.51,
                volume=1000.0,
                contract_id="test",
            )

    def test_negative_volume_raises(self) -> None:
        """Test that negative volume raises ValueError."""
        with pytest.raises(ValueError, match="Volume must be non-negative"):
            MaturityData(
                maturity=30.0,
                market_price=0.5,
                bid_price=0.49,
                ask_price=0.51,
                volume=-100.0,
                contract_id="test",
            )


class TestHazardRateModel:
    """Tests for HazardRateModel class."""

    @pytest.fixture
    def simple_model(self) -> HazardRateModel:
        """Create a simple hazard rate model for testing."""
        maturities = np.array([30.0, 60.0, 90.0, 120.0])
        # Cumulative hazards: increasing, which gives increasing prices
        cumulative_hazards = np.array([0.1, 0.25, 0.45, 0.7])
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    def test_model_initialization(self, simple_model: HazardRateModel) -> None:
        """Test model initializes correctly."""
        assert len(simple_model.maturities) == 4
        assert len(simple_model.cumulative_hazards) == 4
        assert len(simple_model.theoretical_prices) == 4
        assert len(simple_model.survival_probabilities) == 4

    def test_survival_probability_relationship(self, simple_model: HazardRateModel) -> None:
        """Test S(T) = exp(-H(T))."""
        expected_survival = np.exp(-simple_model.cumulative_hazards)
        np.testing.assert_allclose(simple_model.survival_probabilities, expected_survival)

    def test_price_survival_relationship(self, simple_model: HazardRateModel) -> None:
        """Test P(T) = 1 - S(T)."""
        expected_prices = 1.0 - simple_model.survival_probabilities
        np.testing.assert_allclose(simple_model.theoretical_prices, expected_prices)

    def test_prices_are_increasing(self, simple_model: HazardRateModel) -> None:
        """Test that theoretical prices are monotonically increasing."""
        prices = simple_model.theoretical_prices
        assert all(prices[i] <= prices[i + 1] for i in range(len(prices) - 1))

    def test_prices_in_valid_range(self, simple_model: HazardRateModel) -> None:
        """Test prices are in [0, 1]."""
        prices = simple_model.theoretical_prices
        assert all(0 <= p <= 1 for p in prices)

    def test_non_increasing_maturities_raises(self) -> None:
        """Test that non-increasing maturities raise ValueError."""
        with pytest.raises(ValueError, match="Maturities must be strictly increasing"):
            HazardRateModel(
                maturities=np.array([30.0, 60.0, 50.0, 120.0]),
                cumulative_hazards=np.array([0.1, 0.2, 0.3, 0.4]),
            )

    def test_decreasing_hazards_raises(self) -> None:
        """Test that decreasing cumulative hazards raise ValueError."""
        with pytest.raises(ValueError, match="non-decreasing"):
            HazardRateModel(
                maturities=np.array([30.0, 60.0, 90.0, 120.0]),
                cumulative_hazards=np.array([0.1, 0.3, 0.2, 0.4]),  # Decreases at index 2
            )

    def test_negative_hazards_raises(self) -> None:
        """Test that negative cumulative hazards raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            HazardRateModel(
                maturities=np.array([30.0, 60.0, 90.0]),
                cumulative_hazards=np.array([-0.1, 0.2, 0.3]),
            )

    def test_length_mismatch_raises(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            HazardRateModel(
                maturities=np.array([30.0, 60.0, 90.0]),
                cumulative_hazards=np.array([0.1, 0.2]),
            )

    def test_minimum_maturities_required(self) -> None:
        """Test that at least 2 maturities are required."""
        with pytest.raises(ValueError, match="at least 2"):
            HazardRateModel(
                maturities=np.array([30.0]),
                cumulative_hazards=np.array([0.1]),
            )

    def test_bucket_hazards(self, simple_model: HazardRateModel) -> None:
        """Test bucket hazard calculation."""
        bucket_hazards = simple_model.bucket_hazards
        # First bucket: H_1 / T_1
        expected_first = 0.1 / 30.0
        assert bucket_hazards[0] == pytest.approx(expected_first)

        # Second bucket: (H_2 - H_1) / (T_2 - T_1)
        expected_second = (0.25 - 0.1) / (60.0 - 30.0)
        assert bucket_hazards[1] == pytest.approx(expected_second)

    def test_delta_to_hazard(self, simple_model: HazardRateModel) -> None:
        """Test delta calculation (sensitivity to hazard)."""
        deltas = simple_model.delta_to_hazard()
        # Delta = S(T) = 1 - P(T)
        expected = simple_model.survival_probabilities
        np.testing.assert_allclose(deltas, expected)

    def test_theta_proxy(self, simple_model: HazardRateModel) -> None:
        """Test theta proxy calculation."""
        thetas = simple_model.theta_proxy()
        # Theta = -λ * S(T)
        expected = -simple_model.bucket_hazards * simple_model.survival_probabilities
        np.testing.assert_allclose(thetas, expected)

    def test_mispricings(self, simple_model: HazardRateModel) -> None:
        """Test mispricing calculation."""
        # Create market prices with known mispricings
        theo = simple_model.theoretical_prices
        market_prices = theo + np.array([0.01, -0.02, 0.015, -0.005])

        mispricings = simple_model.mispricings(market_prices)
        expected = theo - market_prices
        np.testing.assert_allclose(mispricings, expected)

    def test_mispricings_wrong_length_raises(self, simple_model: HazardRateModel) -> None:
        """Test that wrong length market prices raise ValueError."""
        with pytest.raises(ValueError, match="doesn't match"):
            simple_model.mispricings(np.array([0.1, 0.2, 0.3]))

    def test_price_at_maturity_interpolation(self, simple_model: HazardRateModel) -> None:
        """Test price interpolation at arbitrary maturities."""
        # Price at known maturity should match
        price_at_60 = simple_model.price_at_maturity(60.0)
        assert price_at_60 == pytest.approx(simple_model.theoretical_prices[1])

        # Price at intermediate maturity should be interpolated
        price_at_45 = simple_model.price_at_maturity(45.0)
        assert simple_model.theoretical_prices[0] < price_at_45 < simple_model.theoretical_prices[1]

    def test_price_at_maturity_extrapolation_raises(self, simple_model: HazardRateModel) -> None:
        """Test that extrapolation beyond max maturity raises ValueError."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            simple_model.price_at_maturity(150.0)

    def test_price_at_negative_maturity_raises(self, simple_model: HazardRateModel) -> None:
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            simple_model.price_at_maturity(-10.0)

    def test_unconditional_window_probability(self, simple_model: HazardRateModel) -> None:
        """Test unconditional window probability."""
        prices = simple_model.theoretical_prices

        # First window: P(τ <= T_1) = P_1
        prob_0 = simple_model.unconditional_window_probability(0)
        assert prob_0 == pytest.approx(prices[0])

        # Second window: P(T_1 < τ <= T_2) = P_2 - P_1
        prob_1 = simple_model.unconditional_window_probability(1)
        assert prob_1 == pytest.approx(prices[1] - prices[0])

    def test_from_prices_factory(self) -> None:
        """Test creating model from prices."""
        maturities = np.array([30.0, 60.0, 90.0])
        prices = np.array([0.2, 0.4, 0.6])

        model = HazardRateModel.from_prices(maturities=maturities, prices=prices)

        # Verify prices match
        np.testing.assert_allclose(model.theoretical_prices, prices, rtol=1e-6)

    def test_from_prices_invalid_prices_raises(self) -> None:
        """Test that prices >= 1 raise ValueError in from_prices."""
        with pytest.raises(ValueError, match="must be < 1"):
            HazardRateModel.from_prices(
                maturities=np.array([30.0, 60.0]),
                prices=np.array([0.5, 1.0]),
            )
