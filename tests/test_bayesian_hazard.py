"""Tests for Bayesian hazard rate estimator."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from polymarket_trader.models.bayesian_hazard import (
    BayesianHazardConfig,
    BayesianHazardEstimator,
    BayesianHazardState,
)
from polymarket_trader.models.hazard import MaturityData
from polymarket_trader.state.estimator_state import (
    load_estimator_state,
    save_estimator_state,
)

EASTERN_TZ = ZoneInfo("America/New_York")


def make_market_data(
    maturities: list[float],
    prices: list[float],
    spreads: list[float] | None = None,
) -> list[MaturityData]:
    """Create MaturityData list for testing."""
    if spreads is None:
        spreads = [0.02] * len(maturities)

    return [
        MaturityData(
            maturity=m,
            market_price=p,
            bid_price=p - s / 2,
            ask_price=p + s / 2,
            volume=1000.0,
            contract_id=f"contract_{i}",
        )
        for i, (m, p, s) in enumerate(zip(maturities, prices, spreads, strict=True))
    ]


class TestBayesianHazardConfig:
    """Tests for BayesianHazardConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = BayesianHazardConfig.default()

        assert config.rho == 0.85
        assert config.log_lambda_std == 0.3
        assert config.obs_noise_scale == 0.25
        assert config.process_noise_std == 0.01
        assert config.min_variance == 1e-4
        assert config.isolation_penalty == 2.0

    def test_config_validation_rho(self) -> None:
        """Test rho must be in (0, 1)."""
        with pytest.raises(ValueError, match="rho must be in"):
            BayesianHazardConfig(
                rho=1.5,
                log_lambda_std=0.3,
                obs_noise_scale=0.25,
                process_noise_std=0.01,
                min_variance=1e-4,
                isolation_penalty=2.0,
            )

        with pytest.raises(ValueError, match="rho must be in"):
            BayesianHazardConfig(
                rho=0.0,
                log_lambda_std=0.3,
                obs_noise_scale=0.25,
                process_noise_std=0.01,
                min_variance=1e-4,
                isolation_penalty=2.0,
            )

    def test_config_validation_std(self) -> None:
        """Test log_lambda_std must be positive."""
        with pytest.raises(ValueError, match="log_lambda_std must be positive"):
            BayesianHazardConfig(
                rho=0.85,
                log_lambda_std=-0.1,
                obs_noise_scale=0.25,
                process_noise_std=0.01,
                min_variance=1e-4,
                isolation_penalty=2.0,
            )

    def test_config_validation_isolation_penalty(self) -> None:
        """Test isolation_penalty must be non-negative."""
        with pytest.raises(ValueError, match="isolation_penalty must be non-negative"):
            BayesianHazardConfig(
                rho=0.85,
                log_lambda_std=0.3,
                obs_noise_scale=0.25,
                process_noise_std=0.01,
                min_variance=1e-4,
                isolation_penalty=-1.0,
            )

    def test_config_to_dict_from_dict(self) -> None:
        """Test config serialization roundtrip."""
        config = BayesianHazardConfig.default()
        data = config.to_dict()
        restored = BayesianHazardConfig.from_dict(data)

        assert restored.rho == config.rho
        assert restored.log_lambda_std == config.log_lambda_std
        assert restored.obs_noise_scale == config.obs_noise_scale


class TestBayesianHazardState:
    """Tests for BayesianHazardState."""

    def test_state_validation(self) -> None:
        """Test state validates dimensions."""
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.eye(3)
        maturities = np.array([1.0, 2.0, 3.0])
        bucket_widths = np.array([1.0, 1.0, 1.0])
        contract_ids = ["a", "b", "c"]
        timestamp = datetime.now(EASTERN_TZ)

        # Valid state
        state = BayesianHazardState(
            mu=mu,
            sigma=sigma,
            maturities=maturities,
            bucket_widths=bucket_widths,
            contract_ids=contract_ids,
            timestamp=timestamp,
            n_updates=0,
        )
        assert state.n_buckets == 3

    def test_state_validation_sigma_shape(self) -> None:
        """Test sigma shape must match mu length."""
        with pytest.raises(ValueError, match="sigma shape"):
            BayesianHazardState(
                mu=np.array([1.0, 2.0, 3.0]),
                sigma=np.eye(2),  # Wrong shape
                maturities=np.array([1.0, 2.0, 3.0]),
                bucket_widths=np.array([1.0, 1.0, 1.0]),
                contract_ids=["a", "b", "c"],
                timestamp=datetime.now(EASTERN_TZ),
                n_updates=0,
            )

    def test_state_to_dict_from_dict(self) -> None:
        """Test state serialization roundtrip."""
        state = BayesianHazardState(
            mu=np.array([1.0, 2.0, 3.0]),
            sigma=np.eye(3) * 0.1,
            maturities=np.array([1.0, 2.0, 3.0]),
            bucket_widths=np.array([1.0, 1.0, 1.0]),
            contract_ids=["a", "b", "c"],
            timestamp=datetime.now(EASTERN_TZ),
            n_updates=42,
        )

        data = state.to_dict()
        restored = BayesianHazardState.from_dict(data)

        np.testing.assert_array_almost_equal(restored.mu, state.mu)
        np.testing.assert_array_almost_equal(restored.sigma, state.sigma)
        assert restored.contract_ids == state.contract_ids
        assert restored.n_updates == 42


class TestCorrelationMatrix:
    """Tests for correlation matrix construction."""

    def test_exponential_decay_structure(self) -> None:
        """Test correlation matrix has exponential decay."""
        config = BayesianHazardConfig(
            rho=0.8,
            log_lambda_std=1.0,  # σ² = 1 for easier testing
            obs_noise_scale=0.25,
            process_noise_std=0.01,
            min_variance=1e-4,
            isolation_penalty=2.0,
        )
        estimator = BayesianHazardEstimator(config)

        sigma = estimator._build_correlation_matrix(4)

        # Check diagonal is σ² = 1
        np.testing.assert_array_almost_equal(np.diag(sigma), [1.0, 1.0, 1.0, 1.0])

        # Check off-diagonal follows ρ^|i-j|
        assert abs(sigma[0, 1] - 0.8) < 1e-10  # ρ^1
        assert abs(sigma[0, 2] - 0.64) < 1e-10  # ρ^2
        assert abs(sigma[0, 3] - 0.512) < 1e-10  # ρ^3

    def test_positive_definiteness(self) -> None:
        """Test correlation matrix is positive definite."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        for n in [2, 5, 10, 20]:
            sigma = estimator._build_correlation_matrix(n)
            eigenvalues = np.linalg.eigvalsh(sigma)
            assert np.all(eigenvalues > 0), f"Not positive definite for n={n}"

    def test_symmetry(self) -> None:
        """Test correlation matrix is symmetric."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        sigma = estimator._build_correlation_matrix(5)
        np.testing.assert_array_almost_equal(sigma, sigma.T)


class TestJacobian:
    """Tests for Jacobian computation."""

    def test_lower_triangular(self) -> None:
        """Test Jacobian is lower triangular."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.1, 0.3, 0.5, 0.7],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        jacobian = estimator._compute_jacobian()

        # Check upper triangle is zero (excluding diagonal)
        upper_triangle = np.triu(jacobian, k=1)
        np.testing.assert_array_almost_equal(upper_triangle, np.zeros_like(upper_triangle))

    def test_jacobian_positive_entries(self) -> None:
        """Test Jacobian has positive lower-triangular entries."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        jacobian = estimator._compute_jacobian()

        # Lower triangle should be positive (S_i × λ_j × Δt_j > 0)
        for i in range(3):
            for j in range(i + 1):
                assert jacobian[i, j] > 0, f"J[{i},{j}] should be positive"

    def test_jacobian_shape(self) -> None:
        """Test Jacobian has correct shape."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0, 30.0],
            prices=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        jacobian = estimator._compute_jacobian()

        assert jacobian.shape == (5, 5)


class TestBayesianHazardEstimator:
    """Tests for main estimator class."""

    def test_initialize_sets_state(self) -> None:
        """Test initialization creates valid state."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        assert not estimator.is_initialized

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        assert estimator.is_initialized
        assert estimator.state.n_buckets == 3
        assert estimator.state.n_updates == 0

    def test_initialize_requires_two_contracts(self) -> None:
        """Test initialization fails with fewer than 2 contracts."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0],
            prices=[0.1],
        )

        with pytest.raises(ValueError, match="at least 2 contracts"):
            estimator.initialize(market_data, datetime.now(EASTERN_TZ))

    def test_update_reduces_uncertainty(self) -> None:
        """Test that updates generally reduce uncertainty."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
            spreads=[0.01, 0.01, 0.01],  # Tight spreads = informative
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        # Update with same data (consistent observation)
        diagnostics = estimator.update(market_data, datetime.now(EASTERN_TZ))

        # Uncertainty should decrease (we're getting confirming information)
        assert diagnostics.posterior_uncertainty <= diagnostics.prior_uncertainty

    def test_update_moves_mean_toward_observation(self) -> None:
        """Test that updates move mean toward observed prices."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        # Initialize with some prices
        initial_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(initial_data, datetime.now(EASTERN_TZ))

        initial_model_prices = estimator.get_model_prices()

        # Update with higher prices
        higher_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.15, 0.35, 0.55],  # All prices higher
            spreads=[0.01, 0.01, 0.01],  # Tight spreads
        )
        estimator.update(higher_data, datetime.now(EASTERN_TZ))

        new_model_prices = estimator.get_model_prices()

        # Model prices should have increased
        assert np.all(new_model_prices > initial_model_prices)

    def test_get_model_returns_valid_hazard_model(self) -> None:
        """Test get_model returns usable HazardRateModel."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        model = estimator.get_model()

        # Model should have correct maturities
        np.testing.assert_array_almost_equal(model.maturities, [1.0, 5.0, 10.0])

        # Prices should be in valid range
        assert np.all(model.theoretical_prices >= 0)
        assert np.all(model.theoretical_prices <= 1)

        # Prices should be monotonically increasing
        assert np.all(np.diff(model.theoretical_prices) >= 0)

    def test_confidence_intervals_contain_model_price(self) -> None:
        """Test confidence intervals contain the model price."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        model_prices = estimator.get_model_prices()
        lower, upper = estimator.get_confidence_intervals(confidence=0.95)

        # Model prices should be within confidence intervals
        assert np.all(lower <= model_prices)
        assert np.all(model_prices <= upper)


class TestIsolatedMoves:
    """Tests for the key behavior: isolated moves create larger mispricings."""

    def test_isolated_move_vs_correlated_move(self) -> None:
        """Test isolated move creates larger mispricing than correlated move."""
        config = BayesianHazardConfig(
            rho=0.9,  # High correlation
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.001,  # Low process noise
            min_variance=1e-4,
            isolation_penalty=3.0,  # Moderate penalty
        )

        # Initialize two estimators with same data
        initial_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.10, 0.25, 0.40, 0.55],
        )

        estimator_isolated = BayesianHazardEstimator(config)
        estimator_correlated = BayesianHazardEstimator(config)

        now = datetime.now(EASTERN_TZ)
        estimator_isolated.initialize(initial_data, now)
        estimator_correlated.initialize(initial_data, now)

        # Scenario 1: ONE contract moves (isolated)
        isolated_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.10, 0.25, 0.50, 0.55],  # Only 10.0 moves up by 10%
            spreads=[0.01, 0.01, 0.01, 0.01],
        )

        # Scenario 2: ALL contracts move together (correlated)
        correlated_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.15, 0.30, 0.45, 0.60],  # All move up proportionally
            spreads=[0.01, 0.01, 0.01, 0.01],
        )

        estimator_isolated.update(isolated_data, now)
        estimator_correlated.update(correlated_data, now)

        # Get mispricings (how much model disagrees with market)
        market_isolated = np.array([0.10, 0.25, 0.50, 0.55])
        market_correlated = np.array([0.15, 0.30, 0.45, 0.60])

        mispricing_isolated = estimator_isolated.get_mispricings(market_isolated)
        mispricing_correlated = estimator_correlated.get_mispricings(market_correlated)

        # The isolated move should create a larger absolute mispricing
        # for the contract that moved (index 2)
        isolated_mispricing_at_moved = abs(mispricing_isolated[2])

        # For correlated case, average mispricing should be smaller
        avg_correlated_mispricing = np.mean(np.abs(mispricing_correlated))

        # The isolated move's mispricing for the specific contract should be
        # larger than typical mispricings in the correlated case
        # This is the key behavior we want!
        assert isolated_mispricing_at_moved > avg_correlated_mispricing

    def test_isolation_penalty_increases_noise_for_outliers(self) -> None:
        """Test that isolation penalty increases observation noise for isolated moves."""
        # High penalty config
        config_high = BayesianHazardConfig(
            rho=0.9,
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.001,
            min_variance=1e-4,
            isolation_penalty=5.0,  # High penalty
        )

        # No penalty config
        config_zero = BayesianHazardConfig(
            rho=0.9,
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.001,
            min_variance=1e-4,
            isolation_penalty=0.0,  # No penalty
        )

        initial_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.10, 0.25, 0.40, 0.55],
        )

        estimator_high = BayesianHazardEstimator(config_high)
        estimator_zero = BayesianHazardEstimator(config_zero)

        now = datetime.now(EASTERN_TZ)
        estimator_high.initialize(initial_data, now)
        estimator_zero.initialize(initial_data, now)

        # Isolated move: only contract 2 moves up
        isolated_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.10, 0.25, 0.50, 0.55],  # Only 10.0 moves up
            spreads=[0.01, 0.01, 0.01, 0.01],
        )

        estimator_high.update(isolated_data, now)
        estimator_zero.update(isolated_data, now)

        # With high penalty, the model should be more skeptical of the isolated move
        # So the mispricing at the moved contract should be larger
        market = np.array([0.10, 0.25, 0.50, 0.55])

        mispricing_high = estimator_high.get_mispricings(market)
        mispricing_zero = estimator_zero.get_mispricings(market)

        # With high penalty, model resists the isolated move more
        # → larger mispricing (model disagrees more with the isolated price)
        assert abs(mispricing_high[2]) > abs(mispricing_zero[2])

    def test_coherent_move_is_trusted(self) -> None:
        """Test that coherent moves (all contracts together) are trusted."""
        config = BayesianHazardConfig(
            rho=0.9,
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.001,
            min_variance=1e-4,
            isolation_penalty=5.0,  # High penalty
        )

        initial_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.10, 0.25, 0.40, 0.55],
        )

        estimator = BayesianHazardEstimator(config)
        now = datetime.now(EASTERN_TZ)
        estimator.initialize(initial_data, now)

        # All contracts move up together (coherent signal)
        coherent_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.15, 0.30, 0.45, 0.60],  # All +5%
            spreads=[0.01, 0.01, 0.01, 0.01],
        )

        estimator.update(coherent_data, now)

        # Model should trust the coherent move and have small mispricings
        market = np.array([0.15, 0.30, 0.45, 0.60])
        mispricings = estimator.get_mispricings(market)

        # Average mispricing should be small (model followed the market)
        avg_mispricing = np.mean(np.abs(mispricings))
        assert avg_mispricing < 0.03  # Less than 3%


class TestStatePersistence:
    """Tests for state save/load."""

    def test_save_load_roundtrip(self) -> None:
        """Test state survives save/load roundtrip."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.1, 0.3, 0.5],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        # Do some updates
        for _ in range(3):
            estimator.update(market_data, datetime.now(EASTERN_TZ))

        original_state = estimator.state
        original_config = estimator.config

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            save_estimator_state(original_state, original_config, path)
            loaded_state, loaded_config = load_estimator_state(path)

            np.testing.assert_array_almost_equal(loaded_state.mu, original_state.mu)
            np.testing.assert_array_almost_equal(loaded_state.sigma, original_state.sigma)
            assert loaded_state.contract_ids == original_state.contract_ids
            assert loaded_state.n_updates == original_state.n_updates
            assert loaded_config.rho == original_config.rho
        finally:
            path.unlink()

    def test_load_missing_file_raises(self) -> None:
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_estimator_state(Path("/nonexistent/path.json"))

    def test_load_invalid_json_raises(self) -> None:
        """Test loading invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            f.write("not valid json {{{")
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_estimator_state(path)
        finally:
            path.unlink()

    def test_load_missing_field_raises(self) -> None:
        """Test loading file with missing field raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"version": 1, "config": {}}, f)  # Missing "state"
            path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Missing 'state' field"):
                load_estimator_state(path)
        finally:
            path.unlink()


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_many_updates_stays_positive_definite(self) -> None:
        """Test covariance stays positive definite after many updates."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0, 20.0],
            prices=[0.1, 0.3, 0.5, 0.7],
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        # Run many updates
        for i in range(100):
            # Slightly perturb prices
            perturbed_prices = [
                0.1 + 0.01 * np.sin(i),
                0.3 + 0.01 * np.cos(i),
                0.5 + 0.01 * np.sin(i * 2),
                0.7 + 0.01 * np.cos(i * 2),
            ]
            perturbed_data = make_market_data(
                maturities=[1.0, 5.0, 10.0, 20.0],
                prices=perturbed_prices,
            )
            estimator.update(perturbed_data, datetime.now(EASTERN_TZ))

        # Check covariance is still positive definite
        eigenvalues = np.linalg.eigvalsh(estimator.state.sigma)
        assert np.all(eigenvalues > 0), "Covariance became non-positive-definite"

        # Check no NaN or Inf
        assert not np.any(np.isnan(estimator.state.mu))
        assert not np.any(np.isnan(estimator.state.sigma))
        assert not np.any(np.isinf(estimator.state.mu))
        assert not np.any(np.isinf(estimator.state.sigma))

    def test_extreme_prices_dont_cause_overflow(self) -> None:
        """Test extreme prices don't cause numerical issues."""
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        # Very extreme (but valid) prices
        market_data = make_market_data(
            maturities=[1.0, 5.0, 10.0],
            prices=[0.01, 0.5, 0.99],  # Near boundaries
        )
        estimator.initialize(market_data, datetime.now(EASTERN_TZ))

        # Should not raise and should produce valid outputs
        model_prices = estimator.get_model_prices()
        assert np.all(np.isfinite(model_prices))
        assert np.all(model_prices >= 0)
        assert np.all(model_prices <= 1)
