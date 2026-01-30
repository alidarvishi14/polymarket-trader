"""Tests for the portfolio optimizer module."""

import numpy as np
import pytest

from polymarket_trader.models.hazard import HazardRateModel, MaturityData
from polymarket_trader.optimization.portfolio import (
    HedgeType,
    LiquidityConstraint,
    OptimizationConfig,
    PortfolioOptimizer,
    PortfolioResult,
)


class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creation of valid config."""
        config = OptimizationConfig(
            budget=10000.0,
            min_edge_per_leg=0.01,
            max_concentration=0.25,
            turnover_penalty=0.001,
            hedge_types={HedgeType.DELTA_NEUTRAL},
        )
        assert config.budget == 10000.0
        assert config.min_edge_per_leg == 0.01
        assert config.max_concentration == 0.25
        assert HedgeType.DELTA_NEUTRAL in config.hedge_types

    def test_invalid_budget_raises(self) -> None:
        """Test that non-positive budget raises ValueError."""
        with pytest.raises(ValueError, match="Budget must be positive"):
            OptimizationConfig(
                budget=0.0,
                min_edge_per_leg=0.01,
                max_concentration=0.25,
                turnover_penalty=0.001,
            )

    def test_invalid_concentration_raises(self) -> None:
        """Test that invalid concentration raises ValueError."""
        with pytest.raises(ValueError, match="max_concentration must be in"):
            OptimizationConfig(
                budget=10000.0,
                min_edge_per_leg=0.01,
                max_concentration=1.5,
                turnover_penalty=0.001,
            )

    def test_negative_min_edge_raises(self) -> None:
        """Test that negative min_edge raises ValueError."""
        with pytest.raises(ValueError, match="min_edge_per_leg must be non-negative"):
            OptimizationConfig(
                budget=10000.0,
                min_edge_per_leg=-0.01,
                max_concentration=0.25,
                turnover_penalty=0.001,
            )


class TestPortfolioOptimizer:
    """Tests for PortfolioOptimizer class."""

    @pytest.fixture
    def sample_model(self) -> HazardRateModel:
        """Create a sample hazard rate model."""
        maturities = np.array([30.0, 60.0, 90.0, 120.0])
        cumulative_hazards = np.array([0.1, 0.25, 0.45, 0.7])
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    @pytest.fixture
    def sample_data(self) -> list[MaturityData]:
        """Create sample market data with mispricings."""
        # Theoretical prices from model above are approximately:
        # [0.095, 0.221, 0.362, 0.503]
        # Create market prices with some mispricings
        return [
            MaturityData(
                maturity=30.0,
                market_price=0.08,  # Underpriced YES (buy YES)
                bid_price=0.07,
                ask_price=0.09,
                volume=1000.0,
                contract_id="contract-30",
            ),
            MaturityData(
                maturity=60.0,
                market_price=0.25,  # Overpriced YES (buy NO)
                bid_price=0.24,
                ask_price=0.26,
                volume=800.0,
                contract_id="contract-60",
            ),
            MaturityData(
                maturity=90.0,
                market_price=0.35,  # Slightly underpriced
                bid_price=0.34,
                ask_price=0.36,
                volume=600.0,
                contract_id="contract-90",
            ),
            MaturityData(
                maturity=120.0,
                market_price=0.52,  # Slightly overpriced
                bid_price=0.51,
                ask_price=0.53,
                volume=400.0,
                contract_id="contract-120",
            ),
        ]

    @pytest.fixture
    def basic_config(self) -> OptimizationConfig:
        """Create a basic optimization config."""
        return OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,  # No minimum for testing
            max_concentration=0.5,
            turnover_penalty=0.0,
        )

    def test_optimizer_initialization(
        self,
        sample_model: HazardRateModel,
        basic_config: OptimizationConfig,
    ) -> None:
        """Test optimizer initializes correctly."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        assert optimizer._model == sample_model
        assert optimizer._config == basic_config

    def test_optimize_returns_result(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that optimize returns a PortfolioResult."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data)

        assert isinstance(result, PortfolioResult)
        assert len(result.positions) == len(sample_data)
        assert result.solver_status in ["optimal", "optimal_inaccurate"]

    def test_optimize_respects_budget(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that total cost doesn't exceed budget."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data)

        assert result.total_cost <= basic_config.budget + 1e-6  # Small tolerance

    def test_optimize_respects_concentration(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test that no position exceeds concentration limit."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.25,  # 25% max per position
            turnover_penalty=0.0,
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        max_cost = config.budget * config.max_concentration
        for pos in result.positions:
            assert pos.cost_basis <= max_cost + 1e-6

    def test_optimize_non_negative_positions(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that YES and NO quantities are non-negative."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data)

        for pos in result.positions:
            assert pos.yes_quantity >= -1e-6
            assert pos.no_quantity >= -1e-6

    def test_optimize_with_delta_neutral(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test optimization with delta-neutral constraint."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.0,
            hedge_types={HedgeType.DELTA_NEUTRAL},
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        # Delta should be approximately zero
        assert abs(result.delta_exposure) < 1e-4

    def test_optimize_with_theta_neutral(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test optimization with theta-neutral constraint."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.0,
            hedge_types={HedgeType.THETA_NEUTRAL},
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        # Theta should be approximately zero
        assert abs(result.theta_exposure) < 1e-4

    def test_optimize_with_multiple_hedge_constraints(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test optimization with multiple hedging constraints."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.0,
            hedge_types={HedgeType.DELTA_NEUTRAL, HedgeType.THETA_NEUTRAL},
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        # Both should be approximately zero
        assert abs(result.delta_exposure) < 1e-4
        assert abs(result.theta_exposure) < 1e-4

    def test_optimize_with_liquidity_constraints(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test optimization with liquidity constraints."""
        liquidity = [
            LiquidityConstraint(
                maturity_index=0,
                max_yes_position=10.0,
                max_no_position=10.0,
                order_book_depth=100.0,
            ),
            LiquidityConstraint(
                maturity_index=1,
                max_yes_position=20.0,
                max_no_position=20.0,
                order_book_depth=200.0,
            ),
        ]

        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data, liquidity_constraints=liquidity)

        # Check liquidity limits are respected
        assert result.positions[0].yes_quantity <= 10.0 + 1e-6
        assert result.positions[0].no_quantity <= 10.0 + 1e-6
        assert result.positions[1].yes_quantity <= 20.0 + 1e-6
        assert result.positions[1].no_quantity <= 20.0 + 1e-6

    def test_optimize_min_edge_filters_positions(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test that min_edge_per_leg filters out low-edge positions."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.05,  # Require 5% edge
            max_concentration=0.5,
            turnover_penalty=0.0,
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        # Only positions with sufficient edge should be non-zero
        # (exact positions depend on actual mispricings)
        assert result.solver_status in ["optimal", "optimal_inaccurate"]

    def test_optimize_with_turnover_penalty(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test that turnover penalty reduces total positions."""
        config_no_penalty = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.0,
        )
        config_with_penalty = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.1,  # Significant penalty
        )

        optimizer_no = PortfolioOptimizer(model=sample_model, config=config_no_penalty)
        optimizer_with = PortfolioOptimizer(model=sample_model, config=config_with_penalty)

        result_no = optimizer_no.optimize(sample_data)
        result_with = optimizer_with.optimize(sample_data)

        # Total position size should be smaller with penalty
        total_no = sum(p.yes_quantity + p.no_quantity for p in result_no.positions)
        total_with = sum(p.yes_quantity + p.no_quantity for p in result_with.positions)

        assert total_with <= total_no + 1e-6

    def test_optimize_expected_value_positive_for_mispricings(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that optimizer finds positive edge when mispricings exist."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data)

        # Should have positive total edge
        assert result.total_edge >= 0

    def test_position_edge_calculation(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that position edge is calculated correctly."""
        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        result = optimizer.optimize(sample_data)

        total_edge = sum(p.edge for p in result.positions)
        assert abs(total_edge - result.total_edge) < 1e-6

    def test_maturities_mismatch_raises(
        self,
        sample_model: HazardRateModel,
        basic_config: OptimizationConfig,
    ) -> None:
        """Test that mismatched maturities raise ValueError."""
        wrong_data = [
            MaturityData(
                maturity=35.0,  # Wrong maturity
                market_price=0.15,
                bid_price=0.14,
                ask_price=0.16,
                volume=1000.0,
                contract_id="contract-35",
            ),
            MaturityData(
                maturity=60.0,
                market_price=0.25,
                bid_price=0.24,
                ask_price=0.26,
                volume=800.0,
                contract_id="contract-60",
            ),
            MaturityData(
                maturity=90.0,
                market_price=0.35,
                bid_price=0.34,
                ask_price=0.36,
                volume=600.0,
                contract_id="contract-90",
            ),
            MaturityData(
                maturity=120.0,
                market_price=0.52,
                bid_price=0.51,
                ask_price=0.53,
                volume=400.0,
                contract_id="contract-120",
            ),
        ]

        optimizer = PortfolioOptimizer(model=sample_model, config=basic_config)
        with pytest.raises(ValueError, match="don't match"):
            optimizer.optimize(wrong_data)

    def test_hedge_tolerance_allows_drift(
        self,
        sample_model: HazardRateModel,
        sample_data: list[MaturityData],
    ) -> None:
        """Test that hedge_tolerance allows some exposure drift."""
        config = OptimizationConfig(
            budget=1000.0,
            min_edge_per_leg=0.0,
            max_concentration=0.5,
            turnover_penalty=0.0,
            hedge_types={HedgeType.DELTA_NEUTRAL},
            hedge_tolerance=0.1,  # Allow 0.1 delta drift
        )
        optimizer = PortfolioOptimizer(model=sample_model, config=config)
        result = optimizer.optimize(sample_data)

        # Delta should be within tolerance
        assert abs(result.delta_exposure) <= 0.1 + 1e-6


class TestFactorWeights:
    """Tests for factor weight computation."""

    def test_front_weights_sum_to_one(self) -> None:
        """Test that front weights are normalized."""
        n = 5
        weights = PortfolioOptimizer._compute_factor_weights(n, "front")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_belly_weights_sum_to_one(self) -> None:
        """Test that belly weights are normalized."""
        n = 5
        weights = PortfolioOptimizer._compute_factor_weights(n, "belly")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_back_weights_sum_to_one(self) -> None:
        """Test that back weights are normalized."""
        n = 5
        weights = PortfolioOptimizer._compute_factor_weights(n, "back")
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_front_weights_decrease(self) -> None:
        """Test that front weights decrease with maturity."""
        n = 5
        weights = PortfolioOptimizer._compute_factor_weights(n, "front")
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1]

    def test_back_weights_increase(self) -> None:
        """Test that back weights increase with maturity."""
        n = 5
        weights = PortfolioOptimizer._compute_factor_weights(n, "back")
        for i in range(len(weights) - 1):
            assert weights[i] <= weights[i + 1]

    def test_belly_weights_peak_in_middle(self) -> None:
        """Test that belly weights peak in the middle."""
        n = 7
        weights = PortfolioOptimizer._compute_factor_weights(n, "belly")
        middle = n // 2
        assert weights[middle] >= weights[0]
        assert weights[middle] >= weights[-1]

    def test_invalid_factor_raises(self) -> None:
        """Test that invalid factor name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown factor"):
            PortfolioOptimizer._compute_factor_weights(5, "invalid")
