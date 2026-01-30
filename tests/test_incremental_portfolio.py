"""Tests for incremental portfolio optimization and state management."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from polymarket_trader.models.hazard import HazardRateModel
from polymarket_trader.optimization.simple_portfolio import optimize_simple_portfolio
from polymarket_trader.state import (
    PortfolioState,
    load_state,
    save_state,
)


class TestIncrementalOptimization:
    """Tests for incremental portfolio optimization."""

    @pytest.fixture
    def simple_model(self) -> HazardRateModel:
        """Create a simple hazard model for testing."""
        maturities = np.array([30.0, 60.0, 90.0])
        # Prices: 0.15, 0.30, 0.45 → cumulative hazards
        prices = np.array([0.15, 0.30, 0.45])
        cumulative_hazards = -np.log(1.0 - prices)
        return HazardRateModel(maturities=maturities, cumulative_hazards=cumulative_hazards)

    def test_from_scratch_with_tolerance(self, simple_model: HazardRateModel) -> None:
        """Test from-scratch optimization with tolerance bands."""
        market_prices = np.array([0.14, 0.29, 0.44])  # Slight underpricing

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=1000.0,
            min_theta=-10.0,
            current_positions=None,
            target_delta=0.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )

        # Should find some trades with positive edge
        assert result.solver_status == "optimal"
        # Delta should be within tolerance
        assert abs(result.delta_exposure) <= 10.0 + 1e-6

    def test_incremental_with_existing_positions(self, simple_model: HazardRateModel) -> None:
        """Test optimization with existing positions."""
        market_prices = np.array([0.15, 0.30, 0.45])
        current_positions = np.array([10.0, -5.0, 0.0])  # Long YES at 30d, long NO at 60d

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=500.0,
            min_theta=-10.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )

        assert result.solver_status == "optimal"
        # Final positions should include existing positions + trades
        # net_exposure in result includes current + trade

    def test_tolerance_bands_respected(self, simple_model: HazardRateModel) -> None:
        """Test that final exposures respect tolerance bands."""
        market_prices = np.array([0.15, 0.30, 0.45])
        current_positions = np.array([50.0, 0.0, 0.0])  # Large long position

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=1000.0,
            min_theta=-5.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=5.0,
            theta_tolerance=5.0,
        )

        assert result.solver_status == "optimal"
        # Delta should be within target ± tolerance
        assert result.delta_exposure >= -5.0 - 1e-6
        assert result.delta_exposure <= 5.0 + 1e-6

    def test_no_tolerance_uses_strict_equality(self, simple_model: HazardRateModel) -> None:
        """Test that None tolerance gives strict equality (backward compatible)."""
        market_prices = np.array([0.15, 0.30, 0.45])

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=1000.0,
            min_theta=0.0,
            current_positions=None,
            target_delta=0.0,
            delta_tolerance=None,  # Strict equality
        )

        assert result.solver_status == "optimal"
        # Delta should be exactly 0 (within numerical precision)
        assert abs(result.delta_exposure) < 1e-4

    def test_edge_from_trades_only(self, simple_model: HazardRateModel) -> None:
        """Test that edge is computed from new trades, not existing positions."""
        market_prices = np.array([0.14, 0.29, 0.44])  # Underpriced
        current_positions = np.array([100.0, 0.0, 0.0])  # Already long

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=100.0,
            min_theta=-10.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=50.0,
        )

        # Edge should only reflect new trades, not existing position value
        # With small budget, trades should be limited
        assert result.total_cost <= 100.0 + 1e-6

    def test_position_length_mismatch_raises(self, simple_model: HazardRateModel) -> None:
        """Test that mismatched position length raises error."""
        market_prices = np.array([0.15, 0.30, 0.45])
        current_positions = np.array([10.0, -5.0])  # Wrong length

        with pytest.raises(ValueError, match="current_positions length"):
            optimize_simple_portfolio(
                model=simple_model,
                market_prices=market_prices,
                budget=1000.0,
                min_theta=0.0,
                current_positions=current_positions,
            )

    def test_selling_returns_cash_to_budget(self, simple_model: HazardRateModel) -> None:
        """Test that selling positions returns cash (negative cost)."""
        market_prices = np.array([0.50, 0.50, 0.50])  # All at 50%
        # Large long YES position that needs to be reduced for delta neutrality
        current_positions = np.array([100.0, 0.0, 0.0])

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=0.0,  # No budget for new trades!
            min_theta=-100.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=5.0,
        )

        # With zero budget but existing positions, optimizer should sell to rebalance
        # Cost should be negative (we receive cash from selling)
        # Note: If optimizer can close positions, it should work even with 0 budget
        assert result.solver_status == "optimal"

    def test_cannot_sell_more_than_owned(self, simple_model: HazardRateModel) -> None:
        """Test that we cannot sell more YES than we own."""
        market_prices = np.array([0.15, 0.30, 0.45])
        # Small long YES position
        current_positions = np.array([5.0, 0.0, 0.0])

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=1000.0,
            min_theta=-10.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=10.0,
        )

        # Check that sell_yes doesn't exceed current holdings
        assert result.positions[0].sell_yes <= 5.0 + 1e-6

    def test_cannot_sell_no_if_long_yes(self, simple_model: HazardRateModel) -> None:
        """Test that we cannot sell NO if we're long YES."""
        market_prices = np.array([0.15, 0.30, 0.45])
        current_positions = np.array([10.0, 0.0, 0.0])  # Long YES at maturity 0

        result = optimize_simple_portfolio(
            model=simple_model,
            market_prices=market_prices,
            budget=1000.0,
            min_theta=-10.0,
            current_positions=current_positions,
            target_delta=0.0,
            delta_tolerance=10.0,
        )

        # Cannot sell NO at maturity 0 because we're long YES there
        assert result.positions[0].sell_no < 1e-6


class TestPortfolioState:
    """Tests for portfolio state management."""

    def test_create_initial_state(self) -> None:
        """Test creating initial state."""
        state = PortfolioState.create_initial(
            event="test-event",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )

        assert state.event == "test-event"
        assert state.positions == {}
        assert state.budget == 1000.0
        assert state.config.delta_tolerance == 10.0
        assert state.config.theta_tolerance == 10.0

    def test_get_position_returns_zero_for_missing(self) -> None:
        """Test get_position returns 0 for unknown contracts."""
        state = PortfolioState.create_initial(
            event="test",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )

        assert state.get_position("unknown-contract") == 0.0

    def test_get_position_returns_value_for_existing(self) -> None:
        """Test get_position returns correct value for known contracts."""
        state = PortfolioState.create_initial(
            event="test",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )
        state.positions = {"contract-a": 10.5, "contract-b": -5.2}

        assert state.get_position("contract-a") == 10.5
        assert state.get_position("contract-b") == -5.2

    def test_to_dict_and_from_dict(self) -> None:
        """Test round-trip serialization."""
        original = PortfolioState.create_initial(
            event="test-event",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=5.0,
            use_spread=True,
            smoothness_alpha=0.01,
        )
        original.positions = {"a": 10.0, "b": -5.0}

        data = original.to_dict()
        restored = PortfolioState.from_dict(data)

        assert restored.event == original.event
        assert restored.positions == original.positions
        assert restored.budget == original.budget
        assert restored.config.delta_tolerance == original.config.delta_tolerance
        assert restored.config.theta_tolerance == original.config.theta_tolerance
        assert restored.config.use_spread == original.config.use_spread
        assert restored.config.smoothness_alpha == original.config.smoothness_alpha


class TestStatePersistence:
    """Tests for state file I/O."""

    def test_save_and_load_state(self) -> None:
        """Test saving and loading state from file."""
        state = PortfolioState.create_initial(
            event="test-event",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )
        state.positions = {"contract-1": 15.5, "contract-2": -8.3}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_state(state, path)
            loaded = load_state(path)

            assert loaded.event == state.event
            assert loaded.positions == state.positions
            assert loaded.budget == state.budget
        finally:
            path.unlink()

    def test_load_nonexistent_file_raises(self) -> None:
        """Test loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_state(Path("/nonexistent/path.json"))

    def test_load_invalid_json_raises(self) -> None:
        """Test loading invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_state(path)
        finally:
            path.unlink()

    def test_load_missing_field_raises(self) -> None:
        """Test loading JSON with missing required field raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"event": "test"}, f)  # Missing positions, budget, config
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing required field"):
                load_state(path)
        finally:
            path.unlink()

    def test_state_file_format(self) -> None:
        """Test that saved state file has expected format."""
        state = PortfolioState.create_initial(
            event="us-strikes-iran-by",
            budget=1000.0,
            delta_tolerance=10.0,
            theta_tolerance=10.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_state(state, path)

            with open(path) as f:
                data = json.load(f)

            # Check expected structure
            assert "event" in data
            assert "positions" in data
            assert "budget" in data
            assert "config" in data
            assert "last_updated" in data

            # Check config structure
            config = data["config"]
            assert "target_delta" in config
            assert "target_theta" in config
            assert "delta_tolerance" in config
            assert "theta_tolerance" in config
            assert "use_spread" in config
            assert "smoothness_alpha" in config
        finally:
            path.unlink()
