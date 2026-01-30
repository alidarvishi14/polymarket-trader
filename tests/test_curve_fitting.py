"""Tests for the hazard curve fitting module."""

import numpy as np
import pytest

from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.optimization.curve_fitting import (
    CurveFitResult,
    HazardCurveFitter,
    LossType,
)


class TestHazardCurveFitter:
    """Tests for HazardCurveFitter class."""

    @pytest.fixture
    def sample_maturity_data(self) -> list[MaturityData]:
        """Create sample maturity data for testing."""
        return [
            MaturityData(
                maturity=30.0,
                market_price=0.15,
                bid_price=0.14,
                ask_price=0.16,
                volume=1000.0,
                contract_id="contract-30",
            ),
            MaturityData(
                maturity=60.0,
                market_price=0.30,
                bid_price=0.29,
                ask_price=0.31,
                volume=800.0,
                contract_id="contract-60",
            ),
            MaturityData(
                maturity=90.0,
                market_price=0.45,
                bid_price=0.44,
                ask_price=0.46,
                volume=600.0,
                contract_id="contract-90",
            ),
            MaturityData(
                maturity=120.0,
                market_price=0.55,
                bid_price=0.54,
                ask_price=0.56,
                volume=400.0,
                contract_id="contract-120",
            ),
        ]

    @pytest.fixture
    def fitter(self) -> HazardCurveFitter:
        """Create a default curve fitter."""
        return HazardCurveFitter(
            smoothness_alpha=0.1,
            loss_type=LossType.QUADRATIC,
        )

    def test_fitter_initialization(self) -> None:
        """Test fitter initializes correctly."""
        fitter = HazardCurveFitter(
            smoothness_alpha=0.5,
            loss_type=LossType.QUADRATIC,
        )
        assert fitter._smoothness_alpha == 0.5
        assert fitter._loss_type == LossType.QUADRATIC

    def test_negative_alpha_raises(self) -> None:
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            HazardCurveFitter(
                smoothness_alpha=-0.1,
                loss_type=LossType.QUADRATIC,
            )

    def test_huber_requires_delta(self) -> None:
        """Test that HUBER loss requires delta parameter."""
        with pytest.raises(ValueError, match="huber_delta required"):
            HazardCurveFitter(
                smoothness_alpha=0.1,
                loss_type=LossType.HUBER,
            )

    def test_invalid_huber_delta_raises(self) -> None:
        """Test that non-positive huber_delta raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            HazardCurveFitter(
                smoothness_alpha=0.1,
                loss_type=LossType.HUBER,
                huber_delta=-0.1,
            )

    def test_fit_returns_result(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fit returns a CurveFitResult."""
        result = fitter.fit(sample_maturity_data)
        assert isinstance(result, CurveFitResult)
        assert result.model is not None
        assert result.objective_value >= 0
        assert result.solver_status in ["optimal", "optimal_inaccurate"]

    def test_fit_produces_monotonic_hazards(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fitted hazards are monotonically increasing."""
        result = fitter.fit(sample_maturity_data)
        hazards = result.model.cumulative_hazards

        for i in range(len(hazards) - 1):
            assert hazards[i] <= hazards[i + 1], f"Hazards not monotonic at index {i}"

    def test_fit_produces_non_negative_hazards(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fitted hazards are non-negative."""
        result = fitter.fit(sample_maturity_data)
        hazards = result.model.cumulative_hazards

        assert all(h >= 0 for h in hazards), "Hazards contain negative values"

    def test_fit_produces_valid_prices(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fitted prices are in [0, 1]."""
        result = fitter.fit(sample_maturity_data)
        prices = result.model.theoretical_prices

        assert all(0 <= p <= 1 for p in prices), "Prices outside [0, 1]"

    def test_fit_prices_close_to_market(
        self,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fitted prices are reasonably close to market prices."""
        # Use low smoothness for tight fit
        fitter = HazardCurveFitter(
            smoothness_alpha=0.001,
            loss_type=LossType.QUADRATIC,
        )
        result = fitter.fit(sample_maturity_data)

        sorted_data = sorted(sample_maturity_data, key=lambda x: x.maturity)
        market_prices = np.array([d.market_price for d in sorted_data])
        fitted_prices = result.model.theoretical_prices

        # Fitted prices should be within 5% of market prices
        max_diff = np.max(np.abs(fitted_prices - market_prices))
        assert max_diff < 0.05, f"Max price difference {max_diff} exceeds 0.05"

    def test_fit_with_custom_weights(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test fitting with custom weights."""
        weights = np.array([1.0, 2.0, 1.0, 0.5])
        result = fitter.fit(sample_maturity_data, weights=weights)

        assert result.model is not None
        assert result.solver_status in ["optimal", "optimal_inaccurate"]

    def test_fit_wrong_weights_length_raises(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that wrong weights length raises ValueError."""
        wrong_weights = np.array([1.0, 2.0])  # Wrong length
        with pytest.raises(ValueError, match="Weights length"):
            fitter.fit(sample_maturity_data, weights=wrong_weights)

    def test_fit_insufficient_data_raises(self, fitter: HazardCurveFitter) -> None:
        """Test that fitting with < 2 maturities raises ValueError."""
        single_data = [
            MaturityData(
                maturity=30.0,
                market_price=0.15,
                bid_price=0.14,
                ask_price=0.16,
                volume=1000.0,
                contract_id="contract-30",
            ),
        ]
        with pytest.raises(ValueError, match="at least 2"):
            fitter.fit(single_data)

    def test_fit_with_huber_loss(self, sample_maturity_data: list[MaturityData]) -> None:
        """Test fitting with Huber loss function."""
        fitter = HazardCurveFitter(
            smoothness_alpha=0.1,
            loss_type=LossType.HUBER,
            huber_delta=0.1,
        )
        result = fitter.fit(sample_maturity_data)

        assert result.model is not None
        assert result.solver_status in ["optimal", "optimal_inaccurate"]

    def test_fit_unsorted_data(self, fitter: HazardCurveFitter) -> None:
        """Test that unsorted data is handled correctly."""
        # Create data in non-chronological order
        unsorted_data = [
            MaturityData(
                maturity=90.0,
                market_price=0.45,
                bid_price=0.44,
                ask_price=0.46,
                volume=600.0,
                contract_id="contract-90",
            ),
            MaturityData(
                maturity=30.0,
                market_price=0.15,
                bid_price=0.14,
                ask_price=0.16,
                volume=1000.0,
                contract_id="contract-30",
            ),
            MaturityData(
                maturity=60.0,
                market_price=0.30,
                bid_price=0.29,
                ask_price=0.31,
                volume=800.0,
                contract_id="contract-60",
            ),
        ]

        result = fitter.fit(unsorted_data)

        # Verify maturities are sorted in result
        maturities = result.model.maturities
        assert list(maturities) == sorted(maturities)

    def test_fit_with_non_monotonic_prices_logs_warning(
        self,
        fitter: HazardCurveFitter,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that non-monotonic market prices log a warning."""
        # Create data with non-monotonic prices (arbitrage opportunity)
        data = [
            MaturityData(
                maturity=30.0,
                market_price=0.30,  # Higher than next
                bid_price=0.29,
                ask_price=0.31,
                volume=1000.0,
                contract_id="contract-30",
            ),
            MaturityData(
                maturity=60.0,
                market_price=0.25,  # Lower than previous (arbitrage!)
                bid_price=0.24,
                ask_price=0.26,
                volume=800.0,
                contract_id="contract-60",
            ),
            MaturityData(
                maturity=90.0,
                market_price=0.45,
                bid_price=0.44,
                ask_price=0.46,
                volume=600.0,
                contract_id="contract-90",
            ),
        ]

        fitter.fit(data)

        # Check that warning was logged
        assert any("not monotonic" in record.message for record in caplog.records)

    def test_stress_test_alpha(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test stress testing across multiple alpha values."""
        alpha_values = [0.01, 0.1, 1.0]
        results = fitter.stress_test_alpha(sample_maturity_data, alpha_values)

        assert len(results) == len(alpha_values)
        for alpha in alpha_values:
            assert alpha in results
            assert results[alpha].model is not None

    def test_higher_alpha_produces_smoother_curve(
        self,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that higher alpha produces smoother (less volatile) curves."""
        fitter_low = HazardCurveFitter(smoothness_alpha=0.001, loss_type=LossType.QUADRATIC)
        fitter_high = HazardCurveFitter(smoothness_alpha=1.0, loss_type=LossType.QUADRATIC)

        result_low = fitter_low.fit(sample_maturity_data)
        result_high = fitter_high.fit(sample_maturity_data)

        # Calculate "curvature" as sum of squared second differences in hazard
        def curvature(hazards: np.ndarray) -> float:
            if len(hazards) < 3:
                return 0.0
            second_diff = hazards[2:] - 2 * hazards[1:-1] + hazards[:-2]
            return float(np.sum(second_diff**2))

        curv_low = curvature(result_low.model.cumulative_hazards)
        curv_high = curvature(result_high.model.cumulative_hazards)

        # Higher alpha should produce lower curvature (smoother)
        assert curv_high <= curv_low, "Higher alpha should produce smoother curve"

    def test_fit_residuals_computed(
        self,
        fitter: HazardCurveFitter,
        sample_maturity_data: list[MaturityData],
    ) -> None:
        """Test that fit residuals are computed."""
        result = fitter.fit(sample_maturity_data)

        assert result.fit_residuals is not None
        assert len(result.fit_residuals) == len(sample_maturity_data)
