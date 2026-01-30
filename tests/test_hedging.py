"""Tests for the hedging engine module."""

import numpy as np
import pytest

from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.portfolio import PortfolioPosition
from polymarket_trader.risk.hedging import (
    DriftReport,
    HedgingConfig,
    HedgingEngine,
    RiskExposures,
)


class TestHedgingConfig:
    """Tests for HedgingConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creation of valid config."""
        config = HedgingConfig(
            target_delta=0.0,
            target_theta=0.0,
            delta_threshold=0.1,
            theta_threshold=0.05,
            factor_threshold=0.1,
        )
        assert config.target_delta == 0.0
        assert config.delta_threshold == 0.1

    def test_invalid_thresholds_raise(self) -> None:
        """Test that non-positive thresholds raise ValueError."""
        with pytest.raises(ValueError, match="delta_threshold must be positive"):
            HedgingConfig(
                target_delta=0.0,
                target_theta=0.0,
                delta_threshold=0.0,
                theta_threshold=0.05,
                factor_threshold=0.1,
            )


class TestHedgingEngine:
    """Tests for HedgingEngine class."""

    @pytest.fixture
    def sample_model(self) -> HazardRateModel:
        """Create a sample hazard rate model."""
        maturities = np.array([30.0, 60.0, 90.0, 120.0])
        cumulative_hazards = np.array([0.1, 0.25, 0.45, 0.7])
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    @pytest.fixture
    def sample_config(self) -> HedgingConfig:
        """Create a sample hedging config."""
        return HedgingConfig(
            target_delta=0.0,
            target_theta=0.0,
            delta_threshold=0.1,
            theta_threshold=0.05,
            factor_threshold=0.1,
        )

    @pytest.fixture
    def hedging_engine(
        self,
        sample_model: HazardRateModel,
        sample_config: HedgingConfig,
    ) -> HedgingEngine:
        """Create a hedging engine."""
        return HedgingEngine(model=sample_model, config=sample_config)

    @pytest.fixture
    def sample_positions(self) -> list[PortfolioPosition]:
        """Create sample portfolio positions."""
        return [
            PortfolioPosition(
                maturity=30.0,
                contract_id="c30",
                yes_quantity=10.0,
                no_quantity=0.0,
                net_exposure=10.0,
                cost_basis=1.0,
                expected_value=1.0,
                edge=0.0,
            ),
            PortfolioPosition(
                maturity=60.0,
                contract_id="c60",
                yes_quantity=0.0,
                no_quantity=5.0,
                net_exposure=-5.0,
                cost_basis=1.0,
                expected_value=1.0,
                edge=0.0,
            ),
            PortfolioPosition(
                maturity=90.0,
                contract_id="c90",
                yes_quantity=8.0,
                no_quantity=0.0,
                net_exposure=8.0,
                cost_basis=1.0,
                expected_value=1.0,
                edge=0.0,
            ),
            PortfolioPosition(
                maturity=120.0,
                contract_id="c120",
                yes_quantity=0.0,
                no_quantity=7.0,
                net_exposure=-7.0,
                cost_basis=1.0,
                expected_value=1.0,
                edge=0.0,
            ),
        ]

    def test_compute_exposures_returns_result(
        self,
        hedging_engine: HedgingEngine,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test that compute_exposures returns RiskExposures."""
        exposures = hedging_engine.compute_exposures(sample_positions)
        assert isinstance(exposures, RiskExposures)

    def test_compute_exposures_delta_calculation(
        self,
        hedging_engine: HedgingEngine,
        sample_model: HazardRateModel,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test that delta is computed correctly."""
        exposures = hedging_engine.compute_exposures(sample_positions)

        # Manual calculation
        survival = sample_model.survival_probabilities
        net_exposures = [10.0, -5.0, 8.0, -7.0]
        expected_delta = sum(z * s for z, s in zip(net_exposures, survival, strict=True))

        assert abs(exposures.delta - expected_delta) < 1e-6

    def test_compute_exposures_per_maturity(
        self,
        hedging_engine: HedgingEngine,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test per-maturity delta decomposition."""
        exposures = hedging_engine.compute_exposures(sample_positions)

        assert len(exposures.per_maturity_delta) == 4
        assert abs(sum(exposures.per_maturity_delta) - exposures.delta) < 1e-6

    def test_check_drift_no_rebalance(
        self,
        hedging_engine: HedgingEngine,
    ) -> None:
        """Test drift check when within thresholds."""
        # Create exposures within thresholds
        exposures = RiskExposures(
            delta=0.05,  # Below threshold of 0.1
            theta=0.02,  # Below threshold of 0.05
            front_end=0.05,
            belly=0.05,
            back_end=0.05,
            per_maturity_delta=np.array([0.01, 0.01, 0.01, 0.02]),
            per_maturity_theta=np.array([0.005, 0.005, 0.005, 0.005]),
        )

        drift = hedging_engine.check_drift(exposures)

        assert isinstance(drift, DriftReport)
        assert not drift.requires_rebalance
        assert len(drift.triggering_factors) == 0

    def test_check_drift_delta_triggers(
        self,
        hedging_engine: HedgingEngine,
    ) -> None:
        """Test drift check when delta exceeds threshold."""
        exposures = RiskExposures(
            delta=0.15,  # Above threshold of 0.1
            theta=0.02,
            front_end=0.05,
            belly=0.05,
            back_end=0.05,
            per_maturity_delta=np.array([0.04, 0.04, 0.04, 0.03]),
            per_maturity_theta=np.array([0.005, 0.005, 0.005, 0.005]),
        )

        drift = hedging_engine.check_drift(exposures)

        assert drift.requires_rebalance
        assert "delta" in drift.triggering_factors

    def test_compute_hedge_adjustment(
        self,
        hedging_engine: HedgingEngine,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test hedge adjustment computation."""
        exposures = hedging_engine.compute_exposures(sample_positions)
        adjustments = hedging_engine.compute_hedge_adjustment(exposures)

        assert len(adjustments) == 4
        # Adjustments should try to reduce exposures toward targets

    def test_compute_hedge_adjustment_specific_maturities(
        self,
        hedging_engine: HedgingEngine,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test hedge adjustment with specific target maturities."""
        exposures = hedging_engine.compute_exposures(sample_positions)
        adjustments = hedging_engine.compute_hedge_adjustment(
            exposures,
            target_maturities=[0, 3],  # Only first and last
        )

        assert len(adjustments) == 4
        # Middle maturities should have zero adjustments
        assert adjustments[1] == 0.0
        assert adjustments[2] == 0.0
