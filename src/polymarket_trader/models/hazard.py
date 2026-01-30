"""Hazard rate model for binary event contracts.

This module implements the survival/hazard-rate framework where:
- τ = random event time
- S(T) = P(τ > T) = survival probability
- P(T) = 1 - S(T) = cumulative probability of event by time T
- H(T) = cumulative hazard = -log(S(T))
- λ(t) = instantaneous hazard rate = dH(t)/dt
"""

import logging
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MaturityData:
    """Data for a single maturity in the term structure.

    Attributes:
        maturity: Time to maturity T_i (in days or any consistent unit).
        market_price: Market price P_i^mkt (bid/ask mid or specified side).
        bid_price: Bid price for YES contract.
        ask_price: Ask price for YES contract.
        volume: Trading volume (for weighting in curve fit).
        contract_id: Unique identifier for the contract.
        yes_token_id: Token ID for YES contract (for position mapping).
        no_token_id: Token ID for NO contract (for position mapping).

    """

    maturity: float
    market_price: float
    bid_price: float
    ask_price: float
    volume: float
    contract_id: str
    yes_token_id: str = ""
    no_token_id: str = ""

    def __post_init__(self) -> None:
        """Validate maturity data."""
        if self.maturity <= 0:
            raise ValueError(f"Maturity must be positive, got {self.maturity}")
        if not 0 <= self.market_price <= 1:
            raise ValueError(f"Market price must be in [0, 1], got {self.market_price}")
        if not 0 <= self.bid_price <= self.ask_price <= 1:
            raise ValueError(f"Invalid bid/ask: bid={self.bid_price}, ask={self.ask_price}")
        if self.volume < 0:
            raise ValueError(f"Volume must be non-negative, got {self.volume}")


class HazardRateModel:
    """Hazard rate model for binary event term structures.

    This class represents the fitted hazard curve and provides methods to:
    - Compute theoretical prices from cumulative hazards
    - Calculate bucket hazard rates between maturities
    - Compute forward probabilities
    - Calculate sensitivities (Greeks) in hazard space
    """

    def __init__(
        self,
        maturities: NDArray[np.float64],
        cumulative_hazards: NDArray[np.float64],
    ) -> None:
        """Initialize the hazard rate model.

        Args:
            maturities: Array of maturities T_1 < T_2 < ... < T_n.
            cumulative_hazards: Array of cumulative hazards H_1, H_2, ..., H_n.

        Raises:
            ValueError: If maturities are not strictly increasing or hazards
                are not non-decreasing.

        """
        self._maturities = np.asarray(maturities, dtype=np.float64)
        self._cumulative_hazards = np.asarray(cumulative_hazards, dtype=np.float64)

        self._validate_inputs()
        self._compute_derived_quantities()

        logger.info(
            "HazardRateModel initialized with %d maturities, cumulative hazard range [%.4f, %.4f]",
            len(self._maturities),
            self._cumulative_hazards[0],
            self._cumulative_hazards[-1],
        )

    def _validate_inputs(self) -> None:
        """Validate model inputs satisfy no-arbitrage conditions."""
        if len(self._maturities) != len(self._cumulative_hazards):
            raise ValueError(
                f"Maturities and hazards must have same length: "
                f"{len(self._maturities)} vs {len(self._cumulative_hazards)}"
            )

        if len(self._maturities) < 2:
            raise ValueError(f"Need at least 2 maturities, got {len(self._maturities)}")

        # Check maturities are strictly increasing
        maturity_diffs = np.diff(self._maturities)
        if np.any(maturity_diffs <= 0):
            raise ValueError("Maturities must be strictly increasing")

        # Check cumulative hazards are non-negative
        if np.any(self._cumulative_hazards < 0):
            raise ValueError("Cumulative hazards must be non-negative")

        # Check cumulative hazards are non-decreasing (no-arbitrage)
        hazard_diffs = np.diff(self._cumulative_hazards)
        if np.any(hazard_diffs < -1e-10):  # Small tolerance for numerical errors
            raise ValueError("Cumulative hazards must be non-decreasing (no-arbitrage violation)")

    def _compute_derived_quantities(self) -> None:
        """Compute survival, prices, and bucket hazards from cumulative hazard."""
        # Survival: S(T) = exp(-H(T))
        self._survival_probs = np.exp(-self._cumulative_hazards)

        # Theoretical prices: P(T) = 1 - S(T)
        self._theoretical_prices = 1.0 - self._survival_probs

        # Bucket hazard rates: λ_i = (H_i - H_{i-1}) / (T_i - T_{i-1})
        delta_h = np.diff(self._cumulative_hazards)
        delta_t = np.diff(self._maturities)

        # Prepend the first bucket from T=0 to T_1
        delta_h_full = np.concatenate([[self._cumulative_hazards[0]], delta_h])
        delta_t_full = np.concatenate([[self._maturities[0]], delta_t])

        self._bucket_hazards = delta_h_full / delta_t_full

        # Forward probabilities: q_i = 1 - exp(-λ_i * ΔT_i)
        self._forward_probs = 1.0 - np.exp(-self._bucket_hazards * delta_t_full)

    @property
    def maturities(self) -> NDArray[np.float64]:
        """Get the maturities array."""
        return self._maturities.copy()

    @property
    def cumulative_hazards(self) -> NDArray[np.float64]:
        """Get the cumulative hazards array."""
        return self._cumulative_hazards.copy()

    @property
    def survival_probabilities(self) -> NDArray[np.float64]:
        """Get survival probabilities S(T_i) = P(τ > T_i)."""
        return self._survival_probs.copy()

    @property
    def theoretical_prices(self) -> NDArray[np.float64]:
        """Get theoretical YES prices P(T_i) = P(τ ≤ T_i)."""
        return self._theoretical_prices.copy()

    @property
    def bucket_hazards(self) -> NDArray[np.float64]:
        """Get bucket hazard rates λ_i for each interval."""
        return self._bucket_hazards.copy()

    @property
    def forward_probabilities(self) -> NDArray[np.float64]:
        """Get forward (conditional) probabilities for each bucket."""
        return self._forward_probs.copy()

    def price_at_maturity(self, maturity: float) -> float:
        """Interpolate theoretical price at arbitrary maturity.

        Uses linear interpolation in cumulative hazard space.

        Args:
            maturity: Time to maturity to price.

        Returns:
            Theoretical YES price at the given maturity.

        Raises:
            ValueError: If maturity is outside the fitted range.

        """
        if maturity < 0:
            raise ValueError(f"Maturity must be non-negative, got {maturity}")

        if maturity < self._maturities[0]:
            # Extrapolate from first bucket hazard
            h_interp = self._bucket_hazards[0] * maturity
        elif maturity > self._maturities[-1]:
            raise ValueError(
                f"Maturity {maturity} exceeds maximum fitted maturity {self._maturities[-1]}"
            )
        else:
            # Linear interpolation in cumulative hazard space
            h_interp = float(np.interp(maturity, self._maturities, self._cumulative_hazards))

        return 1.0 - np.exp(-h_interp)

    def delta_to_hazard(self) -> NDArray[np.float64]:
        """Compute delta (sensitivity) of prices to cumulative hazard.

        ∂P_i/∂H_i = (1 - P_i) = S_i

        Returns:
            Array of deltas for each maturity.

        """
        return self._survival_probs.copy()

    def theta_proxy(self) -> NDArray[np.float64]:
        """Compute theta proxy (time decay sensitivity).

        Θ_i ≈ -λ_i * (1 - P_i)

        Returns:
            Array of theta proxies for each maturity.

        """
        return -self._bucket_hazards * self._survival_probs

    def unconditional_window_probability(self, i: int) -> float:
        """Compute unconditional probability of event in window [T_{i-1}, T_i].

        P(T_{i-1} < τ ≤ T_i) = P_i - P_{i-1}

        Args:
            i: Index of the maturity (1-based, where 1 is first bucket).

        Returns:
            Unconditional probability of event in the window.

        Raises:
            ValueError: If index is out of range.

        """
        if i < 0 or i >= len(self._maturities):
            raise ValueError(f"Index {i} out of range [0, {len(self._maturities) - 1}]")

        if i == 0:
            return self._theoretical_prices[0]
        return self._theoretical_prices[i] - self._theoretical_prices[i - 1]

    def mispricings(
        self,
        market_prices: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute mispricings relative to market prices.

        Mispricing_i = P_i^theo - P_i^mkt

        Positive mispricing means YES is underpriced (buy YES).
        Negative mispricing means NO is underpriced (buy NO).

        Args:
            market_prices: Array of market prices for each maturity.

        Returns:
            Array of mispricings for each maturity.

        Raises:
            ValueError: If market_prices length doesn't match maturities.

        """
        market_prices = np.asarray(market_prices, dtype=np.float64)
        if len(market_prices) != len(self._maturities):
            raise ValueError(
                f"Market prices length {len(market_prices)} doesn't match "
                f"maturities length {len(self._maturities)}"
            )

        return self._theoretical_prices - market_prices

    @classmethod
    def from_prices(
        cls,
        maturities: NDArray[np.float64],
        prices: NDArray[np.float64],
    ) -> Self:
        """Create model directly from prices (for testing or manual construction).

        Converts prices to cumulative hazards: H_i = -log(1 - P_i)

        Args:
            maturities: Array of maturities.
            prices: Array of prices in [0, 1).

        Returns:
            HazardRateModel instance.

        Raises:
            ValueError: If any price is >= 1 (would give infinite hazard).

        """
        prices = np.asarray(prices, dtype=np.float64)
        if np.any(prices >= 1.0):
            raise ValueError("Prices must be < 1 to compute finite cumulative hazard")
        if np.any(prices < 0.0):
            raise ValueError("Prices must be >= 0")

        # H = -log(1 - P) = -log(S)
        cumulative_hazards = -np.log(1.0 - prices)

        return cls(maturities=maturities, cumulative_hazards=cumulative_hazards)
