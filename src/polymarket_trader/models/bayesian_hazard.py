"""Bayesian estimation of hazard rates using Extended Kalman Filter.

This module implements a Bayesian approach to hazard rate estimation where:
- log(λ) ~ N(μ, Σ) - log hazard rates are multivariate normal
- Σ has exponential decay correlation structure (adjacent buckets correlated)
- Updates via Extended Kalman Filter when new prices observed

The correlation structure makes isolated price moves suspicious - if one contract
moves but neighbors don't, the model resists updating fully, creating larger
mispricings (trading opportunities).

Key insight: Instead of fitting a curve that perfectly matches market prices,
we maintain beliefs about hazard rates and update them gradually. This provides:
- Resistance to noise/temporary imbalances
- Natural mean reversion behavior
- Uncertainty quantification for model prices
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Self

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

from polymarket_trader.models.hazard import HazardRateModel, MaturityData
from polymarket_trader.optimization.curve_fitting import HazardCurveFitter, LossType

logger = logging.getLogger(__name__)


@dataclass
class BayesianHazardConfig:
    """Configuration for Bayesian hazard rate estimator.

    Attributes:
        rho: Correlation decay between adjacent buckets (0 < rho < 1).
            Higher values = smoother hazard curve assumption.
            0.85 means adjacent buckets are 85% correlated.
        log_lambda_std: Prior standard deviation of log(λ).
            0.3 means λ varies by ~35% (exp(0.3) ≈ 1.35).
        obs_noise_scale: Multiplier for spread-based observation noise.
            R_ii = scale × (ask - bid)². Higher = trust prior more.
        process_noise_std: Std dev of process noise per update.
            How much log(λ) can drift between observations.
            TODO: Could make time-dependent (scale by dt between updates).
        min_variance: Floor for diagonal of Σ to prevent collapse.
            Ensures we never become "too certain".
        isolation_penalty: Penalty for isolated/inconsistent innovations (κ >= 0).
            When one contract moves but correlated neighbors don't, increase
            its observation noise by factor (1 + κ × normalized_deviation²).
            0.0 = no penalty (trust all observations equally).
            2.0 = moderate penalty (recommended).
            5.0 = strong penalty (very skeptical of isolated moves).

    """

    rho: float
    log_lambda_std: float
    obs_noise_scale: float
    process_noise_std: float
    min_variance: float
    isolation_penalty: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.rho < 1:
            raise ValueError(f"rho must be in (0, 1), got {self.rho}")
        if self.log_lambda_std <= 0:
            raise ValueError(f"log_lambda_std must be positive, got {self.log_lambda_std}")
        if self.obs_noise_scale <= 0:
            raise ValueError(f"obs_noise_scale must be positive, got {self.obs_noise_scale}")
        if self.process_noise_std < 0:
            raise ValueError(
                f"process_noise_std must be non-negative, got {self.process_noise_std}"
            )
        if self.min_variance <= 0:
            raise ValueError(f"min_variance must be positive, got {self.min_variance}")
        if self.isolation_penalty < 0:
            raise ValueError(
                f"isolation_penalty must be non-negative, got {self.isolation_penalty}"
            )

    @classmethod
    def default(cls) -> Self:
        """Create default configuration with reasonable starting values."""
        return cls(
            rho=0.85,
            log_lambda_std=0.3,
            obs_noise_scale=0.25,
            process_noise_std=0.01,
            min_variance=1e-4,
            isolation_penalty=2.0,
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "rho": self.rho,
            "log_lambda_std": self.log_lambda_std,
            "obs_noise_scale": self.obs_noise_scale,
            "process_noise_std": self.process_noise_std,
            "min_variance": self.min_variance,
            "isolation_penalty": self.isolation_penalty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from dictionary."""
        return cls(
            rho=data["rho"],
            log_lambda_std=data["log_lambda_std"],
            obs_noise_scale=data["obs_noise_scale"],
            process_noise_std=data["process_noise_std"],
            min_variance=data["min_variance"],
            isolation_penalty=data.get("isolation_penalty", 2.0),
        )


@dataclass
class BayesianHazardState:
    """State of the Bayesian hazard estimator.

    This is what gets persisted between runs.

    Attributes:
        mu: Mean of log(λ) for each bucket, shape (n_buckets,).
        sigma: Covariance matrix of log(λ), shape (n_buckets, n_buckets).
        maturities: Bucket endpoints (DTEs), shape (n_buckets,).
        bucket_widths: Time width of each bucket Δt, shape (n_buckets,).
        contract_ids: Contract identifiers for each maturity.
        timestamp: When this state was last updated.
        n_updates: Number of EKF updates applied since initialization.

    """

    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    maturities: NDArray[np.float64]
    bucket_widths: NDArray[np.float64]
    contract_ids: list[str]
    timestamp: datetime
    n_updates: int

    def __post_init__(self) -> None:
        """Validate state consistency."""
        n = len(self.mu)
        if self.sigma.shape != (n, n):
            raise ValueError(f"sigma shape {self.sigma.shape} doesn't match mu length {n}")
        if len(self.maturities) != n:
            raise ValueError(
                f"maturities length {len(self.maturities)} doesn't match mu length {n}"
            )
        if len(self.bucket_widths) != n:
            raise ValueError(
                f"bucket_widths length {len(self.bucket_widths)} doesn't match mu length {n}"
            )
        if len(self.contract_ids) != n:
            raise ValueError(
                f"contract_ids length {len(self.contract_ids)} doesn't match mu length {n}"
            )

    @property
    def n_buckets(self) -> int:
        """Number of hazard rate buckets."""
        return len(self.mu)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "maturities": self.maturities.tolist(),
            "bucket_widths": self.bucket_widths.tolist(),
            "contract_ids": self.contract_ids,
            "timestamp": self.timestamp.isoformat(),
            "n_updates": self.n_updates,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from dictionary."""
        return cls(
            mu=np.array(data["mu"], dtype=np.float64),
            sigma=np.array(data["sigma"], dtype=np.float64),
            maturities=np.array(data["maturities"], dtype=np.float64),
            bucket_widths=np.array(data["bucket_widths"], dtype=np.float64),
            contract_ids=data["contract_ids"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            n_updates=data["n_updates"],
        )


@dataclass
class UpdateDiagnostics:
    """Diagnostics from an EKF update step.

    Useful for monitoring filter health and debugging.

    Attributes:
        innovation: Difference between observed and predicted prices.
        kalman_gain_diag: Diagonal of Kalman gain matrix (how much we update).
        prior_uncertainty: Trace of Σ before update.
        posterior_uncertainty: Trace of Σ after update.
        observation_noise_diag: Diagonal of R matrix.

    """

    innovation: NDArray[np.float64]
    kalman_gain_diag: NDArray[np.float64]
    prior_uncertainty: float
    posterior_uncertainty: float
    observation_noise_diag: NDArray[np.float64]


class BayesianHazardEstimator:
    """Bayesian estimation of hazard rates using Extended Kalman Filter.

    Models log(λ) as multivariate normal with correlated buckets.
    Updates beliefs when new market prices are observed.

    The correlation structure makes isolated price moves suspicious -
    if one contract moves but neighbors don't, the model resists updating,
    creating larger mispricings (trading opportunities).

    Example usage:
        config = BayesianHazardConfig.default()
        estimator = BayesianHazardEstimator(config)

        # Cold start
        estimator.initialize(market_data, reference_time)

        # Subsequent updates
        diagnostics = estimator.update(new_market_data, new_time)

        # Get model for portfolio optimization
        model = estimator.get_model()
        mispricings = estimator.get_mispricings(market_prices)

        # Get uncertainty
        lower, upper = estimator.get_confidence_intervals()
    """

    def __init__(self, config: BayesianHazardConfig) -> None:
        """Initialize estimator with hyperparameters.

        Args:
            config: Hyperparameters for the Bayesian model.

        """
        self._config = config
        self._state: BayesianHazardState | None = None

    @property
    def config(self) -> BayesianHazardConfig:
        """Get configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Whether the estimator has been initialized with data."""
        return self._state is not None

    @property
    def state(self) -> BayesianHazardState:
        """Current state. Raises if not initialized."""
        if self._state is None:
            raise RuntimeError("Estimator not initialized. Call initialize() first.")
        return self._state

    @state.setter
    def state(self, value: BayesianHazardState) -> None:
        """Set state (used when loading from persistence)."""
        self._state = value

    def initialize(
        self,
        market_data: list[MaturityData],
        reference_time: datetime,
    ) -> None:
        """Initialize state from market data (cold start).

        Uses existing QP curve fitter to get initial μ estimate,
        then constructs Σ with exponential decay correlation.

        Args:
            market_data: Current market observations.
            reference_time: Current time for timestamp.

        Raises:
            ValueError: If market_data has fewer than 2 contracts.

        """
        if len(market_data) < 2:
            raise ValueError(f"Need at least 2 contracts to initialize, got {len(market_data)}")

        # Sort by maturity
        sorted_data = sorted(market_data, key=lambda x: x.maturity)
        n = len(sorted_data)

        # Extract arrays
        maturities = np.array([d.maturity for d in sorted_data], dtype=np.float64)
        contract_ids = [d.contract_id for d in sorted_data]

        # Compute bucket widths: Δt_i = T_i - T_{i-1} (with T_0 = 0)
        bucket_widths = np.diff(maturities, prepend=0.0)

        # Use QP fitter to get initial hazard estimates
        # This leverages existing, tested code for a good starting point
        fitter = HazardCurveFitter(
            smoothness_alpha=0.01,
            loss_type=LossType.QUADRATIC,
        )
        fit_result = fitter.fit(sorted_data)

        # Extract bucket hazards and convert to log space
        bucket_hazards = fit_result.model.bucket_hazards
        # Clamp to avoid log(0)
        bucket_hazards = np.clip(bucket_hazards, 1e-6, None)
        mu = np.log(bucket_hazards)

        # Build covariance matrix with exponential decay correlation
        sigma = self._build_correlation_matrix(n)

        # Create state
        self._state = BayesianHazardState(
            mu=mu,
            sigma=sigma,
            maturities=maturities,
            bucket_widths=bucket_widths,
            contract_ids=contract_ids,
            timestamp=reference_time,
            n_updates=0,
        )

        logger.info(
            "BayesianHazardEstimator initialized: n_buckets=%d, "
            "mu_range=[%.3f, %.3f], trace(Σ)=%.4f",
            n,
            mu.min(),
            mu.max(),
            np.trace(sigma),
        )

    def update(
        self,
        market_data: list[MaturityData],
        reference_time: datetime,
    ) -> UpdateDiagnostics:
        """Update posterior with new observations (EKF step).

        Args:
            market_data: New market observations.
            reference_time: Current time.

        Returns:
            Diagnostics with innovation, Kalman gain, uncertainty changes.

        Raises:
            RuntimeError: If estimator not initialized.
            ValueError: If market_data doesn't match state maturities.

        """
        if self._state is None:
            raise RuntimeError("Estimator not initialized. Call initialize() first.")

        # Sort market data by maturity
        sorted_data = sorted(market_data, key=lambda x: x.maturity)

        # Match maturities - for now, require exact match
        # TODO: Implement proper maturity matching for time evolution
        new_maturities = np.array([d.maturity for d in sorted_data])
        if len(new_maturities) != self._state.n_buckets:
            # Handle maturity changes
            self._state = self._match_maturities(sorted_data, reference_time)

        # Extract observation data
        market_prices = np.array([d.market_price for d in sorted_data])
        bid_prices = np.array([d.bid_price for d in sorted_data])
        ask_prices = np.array([d.ask_price for d in sorted_data])

        # Update bucket widths (DTEs may have changed slightly)
        new_bucket_widths = np.diff(new_maturities, prepend=0.0)
        self._state.bucket_widths = new_bucket_widths
        self._state.maturities = new_maturities

        # --- EKF Update ---

        # 1. Predict step: add process noise (scaled by time since last update)
        dt_seconds = (reference_time - self._state.timestamp).total_seconds()
        dt_minutes = max(dt_seconds / 60.0, 0.0)  # Convert to minutes, floor at 0
        # Scale process noise linearly with time elapsed
        # Use small floor (0.01 min = 0.6s) to avoid numerical issues, not 1.0
        time_scaled_process_var = self._config.process_noise_std**2 * max(dt_minutes, 0.01)
        prior_sigma = self._state.sigma + np.eye(self._state.n_buckets) * time_scaled_process_var
        prior_uncertainty = np.trace(prior_sigma)

        # 2. Compute model predictions at current μ
        predicted_prices = self._compute_prices(self._state.mu)

        # 3. Compute innovation (needed for adaptive noise)
        innovation = market_prices - predicted_prices

        # 4. Compute Jacobian
        jacobian = self._compute_jacobian()

        # 5. Compute observation noise with isolation penalty
        # Base noise from bid-ask spreads
        base_obs_noise = self._compute_observation_noise(bid_prices, ask_prices, market_prices)
        # Adaptive noise: penalize isolated/inconsistent innovations
        obs_noise = self._compute_adaptive_observation_noise(innovation, base_obs_noise)

        # 6. Kalman update equations
        # Innovation covariance: S = J Σ J' + R
        innovation_cov = jacobian @ prior_sigma @ jacobian.T + np.diag(obs_noise)

        # Add regularization to prevent singularity when DTE is very small
        # This can happen when a contract is close to expiry (Δt → 0)
        min_innovation_var = 1e-6
        innovation_cov = innovation_cov + np.eye(self._state.n_buckets) * min_innovation_var

        # Kalman gain: K = Σ J' S^{-1}
        # Use solve for numerical stability instead of explicit inverse
        try:
            kalman_gain = scipy.linalg.solve(
                innovation_cov.T,
                (prior_sigma @ jacobian.T).T,
                assume_a="pos",
            ).T
        except np.linalg.LinAlgError:
            # If still singular, use pseudoinverse as fallback
            logger.warning("Innovation covariance singular, using pseudoinverse fallback")
            innovation_cov_inv = np.linalg.pinv(innovation_cov)
            kalman_gain = prior_sigma @ jacobian.T @ innovation_cov_inv

        # Update mean: μ = μ + K y
        new_mu = self._state.mu + kalman_gain @ innovation

        # Update covariance using Joseph form for numerical stability:
        # Σ = (I - KJ) Σ (I - KJ)' + K R K'
        i_kj = np.eye(self._state.n_buckets) - kalman_gain @ jacobian
        new_sigma = i_kj @ prior_sigma @ i_kj.T + kalman_gain @ np.diag(obs_noise) @ kalman_gain.T

        # 6. Ensure numerical stability
        new_sigma = self._ensure_positive_definite(new_sigma)

        posterior_uncertainty = np.trace(new_sigma)

        # 7. Update state
        self._state.mu = new_mu
        self._state.sigma = new_sigma
        self._state.timestamp = reference_time
        self._state.n_updates += 1

        logger.info(
            "BayesianHazardEstimator updated: n_updates=%d, "
            "innovation_norm=%.4f, uncertainty %.4f → %.4f",
            self._state.n_updates,
            np.linalg.norm(innovation),
            prior_uncertainty,
            posterior_uncertainty,
        )

        return UpdateDiagnostics(
            innovation=innovation,
            kalman_gain_diag=np.diag(kalman_gain),
            prior_uncertainty=prior_uncertainty,
            posterior_uncertainty=posterior_uncertainty,
            observation_noise_diag=obs_noise,
        )

    def get_model(self) -> HazardRateModel:
        """Get HazardRateModel from current posterior mean.

        Returns:
            HazardRateModel that can be used with existing portfolio optimizer.

        """
        state = self.state  # Raises if not initialized

        # Compute hazard rates from log-space
        bucket_hazards = np.exp(state.mu)

        # Compute cumulative hazards
        cumulative_hazards = np.cumsum(bucket_hazards * state.bucket_widths)

        return HazardRateModel(
            maturities=state.maturities.copy(),
            cumulative_hazards=cumulative_hazards,
        )

    def get_model_prices(self) -> NDArray[np.float64]:
        """Get model prices from posterior mean.

        Returns:
            Array of model prices for each maturity.

        """
        return self._compute_prices(self.state.mu)

    def get_mispricings(
        self,
        market_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Get mispricings (model - market).

        Positive = model thinks price should be higher = YES underpriced.
        Negative = model thinks price should be lower = NO underpriced.

        Args:
            market_prices: Current market prices.

        Returns:
            Array of mispricings.

        """
        return self.get_model_prices() - market_prices

    def get_price_uncertainty(self) -> NDArray[np.float64]:
        """Get uncertainty (std dev) of model prices.

        Propagates posterior uncertainty through the observation model
        using first-order approximation (delta method).

        Returns:
            Array of standard deviations for each price.

        Note:
            TODO: Could use Monte Carlo for more accurate uncertainty
            when nonlinearity is severe.

        """
        state = self.state
        jacobian = self._compute_jacobian()

        # Var(P) ≈ J Σ J' (delta method)
        price_cov = jacobian @ state.sigma @ jacobian.T

        # Return standard deviations
        return np.sqrt(np.diag(price_cov))

    def get_confidence_intervals(
        self,
        confidence: float = 0.95,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get confidence intervals for model prices.

        Args:
            confidence: Confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Tuple of (lower_bounds, upper_bounds) arrays.

        """
        from scipy import stats

        model_prices = self.get_model_prices()
        price_std = self.get_price_uncertainty()

        # Normal quantile for confidence level
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = np.clip(model_prices - z * price_std, 0.0, 1.0)
        upper = np.clip(model_prices + z * price_std, 0.0, 1.0)

        return lower, upper

    # --- Private methods ---

    def _build_correlation_matrix(self, n: int) -> NDArray[np.float64]:
        """Build exponential decay covariance matrix.

        Σ_ij = σ² × ρ^|i-j|

        where σ² = log_lambda_std² and ρ = correlation decay.

        Args:
            n: Number of buckets.

        Returns:
            Covariance matrix of shape (n, n).

        Note:
            TODO: Could support other kernels (squared_exp, matern)
            via config.correlation_kernel parameter.

        """
        rho = self._config.rho
        sigma_sq = self._config.log_lambda_std**2

        # Build correlation matrix: C_ij = ρ^|i-j|
        indices = np.arange(n)
        correlation = rho ** np.abs(indices[:, None] - indices[None, :])

        # Scale by variance
        return sigma_sq * correlation

    def _compute_prices(self, log_lambda: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute prices from log hazard rates.

        Args:
            log_lambda: Log hazard rates for each bucket.

        Returns:
            Prices (cumulative probabilities) for each maturity.

        """
        state = self.state
        bucket_hazards = np.exp(log_lambda)
        cumulative_hazards = np.cumsum(bucket_hazards * state.bucket_widths)
        survival_probs = np.exp(-cumulative_hazards)
        return 1.0 - survival_probs

    def _compute_jacobian(self) -> NDArray[np.float64]:
        """Compute Jacobian ∂P/∂log(λ).

        J[i,j] = S_i × λ_j × Δt_j    if j ≤ i
               = 0                    if j > i

        where:
            S_i = survival probability at maturity i
            λ_j = hazard rate for bucket j
            Δt_j = bucket width for bucket j

        Returns:
            Lower-triangular Jacobian matrix of shape (n, n).

        """
        state = self.state
        n = state.n_buckets

        # Compute current values
        bucket_hazards = np.exp(state.mu)  # λ_j
        cumulative_hazards = np.cumsum(bucket_hazards * state.bucket_widths)
        survival_probs = np.exp(-cumulative_hazards)  # S_i

        # Build Jacobian (lower triangular)
        # J[i,j] = S_i × λ_j × Δt_j for j ≤ i
        jacobian = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1):
                jacobian[i, j] = survival_probs[i] * bucket_hazards[j] * state.bucket_widths[j]

        return jacobian

    def _compute_observation_noise(
        self,
        bid_prices: NDArray[np.float64],
        ask_prices: NDArray[np.float64],
        market_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute base observation noise variance from relative bid-ask spreads.

        R_ii = scale × (spread_i / mid_i)²

        Uses relative spread (spread / mid price) so that:
        - 2c spread at 5c price (40% relative) → high noise
        - 2c spread at 50c price (4% relative) → low noise

        Wider relative spread = less certain observation = larger R = trust prior more.

        Args:
            bid_prices: Bid prices for each maturity.
            ask_prices: Ask prices for each maturity.
            market_prices: Mid-market prices for each maturity.

        Returns:
            Diagonal of base observation noise covariance R.

        """
        spreads = ask_prices - bid_prices
        # Use relative spread (spread / mid) to scale by price level
        # Clamp mid price to avoid division by zero for very low prices
        mid_prices_safe = np.maximum(market_prices, 0.01)
        relative_spreads = spreads / mid_prices_safe
        # Ensure minimum noise to avoid numerical issues
        min_noise = 1e-6
        return np.maximum(
            self._config.obs_noise_scale * relative_spreads**2,
            min_noise,
        )

    def _compute_adaptive_observation_noise(
        self,
        innovation: NDArray[np.float64],
        base_noise: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Scale observation noise based on innovation consistency with neighbors.

        Uses the prior correlation structure (ρ) to compute what each contract's
        innovation "should be" based on its correlated neighbors. Isolated moves
        (deviations from expected) get higher observation noise, making the filter
        more skeptical.

        This implements: "If you move like your neighbors, I trust you.
        If you're the odd one out, I'm suspicious."

        Args:
            innovation: Observed minus predicted prices (y = market - model).
            base_noise: Base observation noise from bid-ask spreads.

        Returns:
            Adaptive observation noise with isolated moves penalized.

        """
        n = len(innovation)
        kappa = self._config.isolation_penalty

        # If no penalty, return base noise unchanged
        if kappa == 0.0:
            return base_noise

        rho = self._config.rho

        # Compute expected innovation for each contract based on correlated neighbors
        # expected[i] = Σ_j (ρ^|i-j| × y[j]) / Σ_j (ρ^|i-j|)  for j ≠ i
        expected_innovation = np.zeros_like(innovation)

        for i in range(n):
            # Correlation weights: ρ^|i-j| for neighbors, 0 for self
            weights = np.array([rho ** abs(i - j) if j != i else 0.0 for j in range(n)])
            weight_sum = weights.sum()

            if weight_sum > 0:
                weights /= weight_sum
                expected_innovation[i] = np.dot(weights, innovation)
            else:
                # Edge case: only one contract, no neighbors
                expected_innovation[i] = 0.0

        # Deviation from expected (how "isolated" is this move?)
        deviation = innovation - expected_innovation

        # Normalize by typical innovation scale
        innovation_std = np.std(innovation)
        if innovation_std < 1e-8:
            # All innovations are the same - no isolated moves
            return base_noise

        normalized_deviation = deviation / innovation_std

        # Penalty multiplier: 1 + κ × deviation²
        # Large deviation → large multiplier → high noise → skeptical
        multiplier = 1.0 + kappa * normalized_deviation**2

        logger.debug(
            "Adaptive observation noise: max_deviation=%.4f, max_multiplier=%.2f",
            np.max(np.abs(normalized_deviation)),
            np.max(multiplier),
        )

        return base_noise * multiplier

    def _ensure_positive_definite(
        self,
        sigma: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Ensure covariance matrix is positive definite.

        1. Symmetrize: Σ = (Σ + Σ') / 2
        2. Clamp small/negative eigenvalues
        3. Ensure minimum variance on diagonal

        Args:
            sigma: Covariance matrix to fix.

        Returns:
            Positive definite covariance matrix.

        """
        # Symmetrize
        sigma = (sigma + sigma.T) / 2

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)

        # Clamp small eigenvalues
        eigenvalues = np.maximum(eigenvalues, self._config.min_variance)

        # Reconstruct
        sigma = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Ensure minimum diagonal variance
        np.fill_diagonal(
            sigma,
            np.maximum(np.diag(sigma), self._config.min_variance),
        )

        return sigma

    def _match_maturities(
        self,
        new_market_data: list[MaturityData],
        reference_time: datetime,
    ) -> BayesianHazardState:
        """Match new maturities to existing state.

        Handles:
        - Expired contracts (in state but not in new data)
        - New contracts (in new data but not in state)
        - Shifted maturities (DTE decreased due to time passing)

        Args:
            new_market_data: New market observations.
            reference_time: Current time.

        Returns:
            Updated state with matched maturities.

        Note:
            TODO: Could use interpolation for smoother transitions
            when maturities shift significantly.

        """
        old_state = self.state
        new_n = len(new_market_data)

        new_maturities = np.array([d.maturity for d in new_market_data])
        new_contract_ids = [d.contract_id for d in new_market_data]
        new_bucket_widths = np.diff(new_maturities, prepend=0.0)

        # Try to match by contract_id
        old_id_to_idx = {cid: i for i, cid in enumerate(old_state.contract_ids)}

        new_mu = np.zeros(new_n, dtype=np.float64)
        new_sigma = self._build_correlation_matrix(new_n)

        matched_count = 0
        for new_idx, cid in enumerate(new_contract_ids):
            if cid in old_id_to_idx:
                old_idx = old_id_to_idx[cid]
                new_mu[new_idx] = old_state.mu[old_idx]
                # Keep the correlation structure, but transfer diagonal uncertainty
                new_sigma[new_idx, new_idx] = min(
                    old_state.sigma[old_idx, old_idx],
                    new_sigma[new_idx, new_idx],
                )
                matched_count += 1
            else:
                # New contract - interpolate from neighbors or use prior
                # Find nearest neighbors in old state
                if new_idx > 0 and new_contract_ids[new_idx - 1] in old_id_to_idx:
                    # Use previous bucket's value
                    prev_old_idx = old_id_to_idx[new_contract_ids[new_idx - 1]]
                    new_mu[new_idx] = old_state.mu[prev_old_idx]
                elif new_idx < new_n - 1 and new_contract_ids[new_idx + 1] in old_id_to_idx:
                    # Use next bucket's value
                    next_old_idx = old_id_to_idx[new_contract_ids[new_idx + 1]]
                    new_mu[new_idx] = old_state.mu[next_old_idx]
                else:
                    # No neighbors - use mean of old state
                    new_mu[new_idx] = np.mean(old_state.mu)

        logger.info(
            "Maturity matching: %d/%d contracts matched, %d new",
            matched_count,
            new_n,
            new_n - matched_count,
        )

        return BayesianHazardState(
            mu=new_mu,
            sigma=new_sigma,
            maturities=new_maturities,
            bucket_widths=new_bucket_widths,
            contract_ids=new_contract_ids,
            timestamp=reference_time,
            n_updates=old_state.n_updates,
        )
