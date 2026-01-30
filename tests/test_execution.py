"""Tests for execution modules (rebalancing and exit rules)."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from polymarket_trader.execution.exit_rules import (
    ExitConfig,
    ExitReason,
    ExitRuleEngine,
)
from polymarket_trader.execution.rebalancing import (
    RebalanceConfig,
    RebalanceDecision,
    RebalanceReason,
    RebalancingPolicy,
)
from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.portfolio import PortfolioPosition
from polymarket_trader.risk.hedging import HedgingConfig, HedgingEngine


class TestRebalanceConfig:
    """Tests for RebalanceConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creation of valid config."""
        config = RebalanceConfig(
            min_days_to_expiry=5.0,
            roll_buffer_days=2.0,
            max_rebalance_frequency_hours=24.0,
            drift_band_pct=0.1,
            transaction_cost_bps=10.0,
            min_edge_after_costs=0.01,
        )
        assert config.min_days_to_expiry == 5.0
        assert config.roll_buffer_days == 2.0

    def test_invalid_drift_band_raises(self) -> None:
        """Test that invalid drift_band raises ValueError."""
        with pytest.raises(ValueError, match="drift_band_pct must be in"):
            RebalanceConfig(
                min_days_to_expiry=5.0,
                roll_buffer_days=2.0,
                max_rebalance_frequency_hours=24.0,
                drift_band_pct=1.5,  # Invalid
                transaction_cost_bps=10.0,
                min_edge_after_costs=0.01,
            )


class TestRebalancingPolicy:
    """Tests for RebalancingPolicy class."""

    @pytest.fixture
    def sample_model(self) -> HazardRateModel:
        """Create a sample hazard rate model."""
        maturities = np.array([30.0, 60.0, 90.0, 120.0])
        cumulative_hazards = np.array([0.1, 0.25, 0.45, 0.7])
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    @pytest.fixture
    def hedging_engine(self, sample_model: HazardRateModel) -> HedgingEngine:
        """Create a hedging engine."""
        config = HedgingConfig(
            target_delta=0.0,
            target_theta=0.0,
            delta_threshold=0.1,
            theta_threshold=0.05,
            factor_threshold=0.1,
        )
        return HedgingEngine(model=sample_model, config=config)

    @pytest.fixture
    def rebal_config(self) -> RebalanceConfig:
        """Create a rebalancing config."""
        return RebalanceConfig(
            min_days_to_expiry=5.0,
            roll_buffer_days=2.0,
            max_rebalance_frequency_hours=24.0,
            drift_band_pct=0.1,
            transaction_cost_bps=10.0,
            min_edge_after_costs=0.01,
        )

    @pytest.fixture
    def policy(
        self,
        sample_model: HazardRateModel,
        hedging_engine: HedgingEngine,
        rebal_config: RebalanceConfig,
    ) -> RebalancingPolicy:
        """Create a rebalancing policy."""
        return RebalancingPolicy(
            model=sample_model,
            hedging_engine=hedging_engine,
            config=rebal_config,
        )

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
                yes_quantity=10.0,
                no_quantity=0.0,
                net_exposure=10.0,
                cost_basis=1.0,
                expected_value=1.0,
                edge=0.0,
            ),
        ]

    def test_evaluate_returns_decision(
        self,
        policy: RebalancingPolicy,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test that evaluate returns a RebalanceDecision."""
        decision = policy.evaluate(
            positions=sample_positions,
            current_time=datetime.now(),
            days_to_expiry=np.array([30.0, 60.0, 90.0, 120.0]),
            market_prices=np.array([0.1, 0.2, 0.4, 0.5]),
        )
        assert isinstance(decision, RebalanceDecision)

    def test_evaluate_time_to_expiry_trigger(
        self,
        policy: RebalancingPolicy,
    ) -> None:
        """Test that near-expiry positions trigger rebalance."""
        positions = [
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
        ]

        decision = policy.evaluate(
            positions=positions,
            current_time=datetime.now(),
            days_to_expiry=np.array([3.0, 60.0, 90.0, 120.0]),  # First is near expiry
            market_prices=np.array([0.1, 0.2, 0.4, 0.5]),
        )

        assert decision.should_rebalance
        assert RebalanceReason.TIME_TO_EXPIRY in decision.reasons

    def test_respects_rebalance_frequency(
        self,
        policy: RebalancingPolicy,
        sample_positions: list[PortfolioPosition],
    ) -> None:
        """Test that policy respects minimum rebalance frequency."""
        current_time = datetime.now()

        # Record a recent rebalance
        policy.record_rebalance(current_time - timedelta(hours=12))

        # Try to evaluate again (should skip due to frequency limit)
        decision = policy.evaluate(
            positions=sample_positions,
            current_time=current_time,
            days_to_expiry=np.array([30.0, 60.0, 90.0, 120.0]),
            market_prices=np.array([0.1, 0.2, 0.4, 0.5]),
        )

        assert not decision.should_rebalance

    def test_roll_instructions_generated(
        self,
        policy: RebalancingPolicy,
    ) -> None:
        """Test that roll instructions are generated for expiring positions."""
        positions = [
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
        ]

        decision = policy.evaluate(
            positions=positions,
            current_time=datetime.now(),
            days_to_expiry=np.array([5.0, 60.0, 90.0, 120.0]),  # Within buffer
            market_prices=np.array([0.1, 0.2, 0.4, 0.5]),
        )

        assert len(decision.roll_instructions) > 0
        assert decision.roll_instructions[0].from_maturity == 30.0
        assert decision.roll_instructions[0].to_maturity == 60.0


class TestExitConfig:
    """Tests for ExitConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creation of valid config."""
        config = ExitConfig(
            convergence_threshold=0.01,
            pnl_capture_target=0.7,
            min_days_to_expiry=5.0,
            stop_loss_pct=0.1,
            model_stability_threshold=0.8,
        )
        assert config.convergence_threshold == 0.01
        assert config.pnl_capture_target == 0.7

    def test_invalid_pnl_target_raises(self) -> None:
        """Test that invalid pnl_capture_target raises ValueError."""
        with pytest.raises(ValueError, match="pnl_capture_target must be in"):
            ExitConfig(
                convergence_threshold=0.01,
                pnl_capture_target=1.5,  # Invalid
                min_days_to_expiry=5.0,
                stop_loss_pct=0.1,
                model_stability_threshold=0.8,
            )


class TestExitRuleEngine:
    """Tests for ExitRuleEngine class."""

    @pytest.fixture
    def exit_config(self) -> ExitConfig:
        """Create an exit config."""
        return ExitConfig(
            convergence_threshold=0.01,
            pnl_capture_target=0.7,
            min_days_to_expiry=5.0,
            stop_loss_pct=0.2,
            model_stability_threshold=0.8,
        )

    @pytest.fixture
    def exit_engine(self, exit_config: ExitConfig) -> ExitRuleEngine:
        """Create an exit rule engine."""
        return ExitRuleEngine(config=exit_config)

    @pytest.fixture
    def sample_model(self) -> HazardRateModel:
        """Create a sample hazard rate model."""
        maturities = np.array([30.0, 60.0, 90.0])
        cumulative_hazards = np.array([0.1, 0.25, 0.45])
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    @pytest.fixture
    def sample_position(self) -> PortfolioPosition:
        """Create a sample position."""
        return PortfolioPosition(
            maturity=30.0,
            contract_id="c30",
            yes_quantity=10.0,
            no_quantity=0.0,
            net_exposure=10.0,
            cost_basis=1.0,
            expected_value=1.1,
            edge=0.1,
        )

    def test_register_position(
        self,
        exit_engine: ExitRuleEngine,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test position registration."""
        exit_engine.register_position(
            position=sample_position,
            entry_price=0.1,
            entry_theoretical=0.15,
        )
        assert sample_position.contract_id in exit_engine._position_states

    def test_unregister_position(
        self,
        exit_engine: ExitRuleEngine,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test position unregistration."""
        exit_engine.register_position(
            position=sample_position,
            entry_price=0.1,
            entry_theoretical=0.15,
        )
        exit_engine.unregister_position(sample_position.contract_id)
        assert sample_position.contract_id not in exit_engine._position_states

    def test_evaluate_convergence_exit(
        self,
        exit_engine: ExitRuleEngine,
        sample_model: HazardRateModel,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test convergence exit rule."""
        exit_engine.register_position(
            position=sample_position,
            entry_price=0.08,
            entry_theoretical=0.10,
        )

        # Market prices close to theoretical (converged)
        market_prices = sample_model.theoretical_prices + 0.005  # Small mispricing

        evaluation = exit_engine.evaluate(
            model=sample_model,
            current_positions=[sample_position],
            market_prices=market_prices,
            days_to_expiry=np.array([30.0, 60.0, 90.0]),
        )

        assert len(evaluation.signals) > 0
        assert evaluation.signals[0].reason == ExitReason.CONVERGENCE

    def test_evaluate_time_exit(
        self,
        exit_engine: ExitRuleEngine,
        sample_model: HazardRateModel,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test time-to-expiry exit rule."""
        exit_engine.register_position(
            position=sample_position,
            entry_price=0.08,
            entry_theoretical=0.10,
        )

        # Position is close to expiry
        market_prices = np.array([0.08, 0.20, 0.40])

        evaluation = exit_engine.evaluate(
            model=sample_model,
            current_positions=[sample_position],
            market_prices=market_prices,
            days_to_expiry=np.array([3.0, 60.0, 90.0]),  # First is near expiry
        )

        assert len(evaluation.signals) > 0
        assert evaluation.signals[0].reason == ExitReason.TIME_TO_EXPIRY

    def test_check_model_stability_unstable(
        self,
        exit_engine: ExitRuleEngine,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test model stability check for unstable position."""
        # Edge varies wildly with alpha
        model_results = {
            0.01: 0.05,
            0.1: 0.15,
            1.0: -0.05,
        }

        signal = exit_engine.check_model_stability(sample_position, model_results)

        assert signal is not None
        assert signal.reason == ExitReason.MODEL_CHANGE

    def test_check_model_stability_stable(
        self,
        exit_engine: ExitRuleEngine,
        sample_position: PortfolioPosition,
    ) -> None:
        """Test model stability check for stable position."""
        # Edge consistent across alphas
        model_results = {
            0.01: 0.10,
            0.1: 0.11,
            1.0: 0.09,
        }

        signal = exit_engine.check_model_stability(sample_position, model_results)

        assert signal is None  # No exit signal for stable position
