"""Single observation builder and fixed observation normaliser.

Observation (K2, OQ-1): ``o_t = (demand(t), PTF(t - k) for k in lags)`` with
the lags in ascending order. Periods are addressed by ``periodNumber`` (B1).
A lag is missing when ``t - k < 1`` or no cleared PTF exists for ``t - k``.
A missing lag is never replaced by 0. The same builder and the same normaliser
serve acting and learning (C2, C4).

Demand is standardised with fixed constants computed once from all 8808 valid
DEMAND values of ``abmem/data/marketData.xlsx`` (OQ-2). Price features are
mapped affinely so that the market bid bounds land on -1 and +1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final, Protocol, Self

import numpy as np

DEFAULT_LAGS: Final[tuple[int, ...]] = (1, 24)
ALLOWED_LAG_SETS: Final[frozenset[tuple[int, ...]]] = frozenset({(1, 24), (1, 24, 168)})

# Mean and population standard deviation (ddof = 0) of the 8808 non-missing
# DEMAND values of abmem/data/marketData.xlsx (OQ-2). Checked against the file
# by tests/agent/test_observation.py.
DEMAND_SCALING_MEAN: Final[float] = 38997.47348206176
DEMAND_SCALING_STD: Final[float] = 6494.932135899229


class DemandSource(Protocol):
    """Structural type for a demand series addressed by period number."""

    def at(self, period_number: int) -> Decimal:
        """Return the demand of period ``period_number``.

        Args:
            period_number: 1-based period number.

        Returns:
            Demand in MWh.
        """
        ...


def _validate_lags(lags: Sequence[int]) -> tuple[int, ...]:
    """Return ``lags`` as a tuple if it is an allowed lag set.

    Args:
        lags: Candidate lag set.

    Returns:
        The lag set as a tuple.

    Raises:
        ValueError: If the lag set is not in ``ALLOWED_LAG_SETS``.
    """
    lag_tuple = tuple(lags)
    if lag_tuple not in ALLOWED_LAG_SETS:
        allowed = sorted(ALLOWED_LAG_SETS)
        raise ValueError(f"lags {lag_tuple} not allowed, expected one of {allowed}")
    return lag_tuple


@dataclass(frozen=True)
class Observation:
    """Observation of period ``t`` before it is cleared.

    Attributes:
        t: Period number the observation belongs to.
        lags: Lag set used to build it.
        raw: float64 array of shape ``(1 + len(lags),)`` when complete, else None.
        missing_lags: Lags whose PTF is not available, ascending.
    """

    t: int
    lags: tuple[int, ...]
    raw: np.ndarray | None
    missing_lags: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Whether every configured lag is available.

        Returns:
            True if ``missing_lags`` is empty.
        """
        return len(self.missing_lags) == 0


def build_observation(
    t: int,
    demand: DemandSource,
    ptf_history: Mapping[int, Decimal | None],
    lags: Sequence[int] = DEFAULT_LAGS,
) -> Observation:
    """Build ``o_t = (demand(t), PTF(t - k) for k in lags)``.

    Only ``demand(t)`` and PTF values of periods strictly before ``t`` are
    read. A lag ``k`` is missing if ``t - k < 1``, if ``t - k`` is not a key
    of ``ptf_history`` or if its value is None (period not cleared).

    Args:
        t: Period number (>= 1).
        demand: Demand source addressed by period number.
        ptf_history: Cleared PTF by period number.
        lags: Lag set, must be in ``ALLOWED_LAG_SETS``.

    Returns:
        The observation. ``raw`` is None if any lag is missing.

    Raises:
        ValueError: If the lag set is not allowed, ``t < 1``, or a demand or
            PTF value is non-finite.
    """
    lag_tuple = _validate_lags(lags)
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 1:
        raise ValueError(f"t must be an integer >= 1, got {t!r}")
    t = int(t)

    prices: list[float] = []
    missing: list[int] = []
    for k in lag_tuple:
        period = t - k
        value = ptf_history.get(period) if period >= 1 else None
        if value is None:
            missing.append(k)
            continue
        price = float(value)
        if not math.isfinite(price):
            raise ValueError(f"PTF of period {period} is not finite: {value}")
        prices.append(price)

    if missing:
        return Observation(t=t, lags=lag_tuple, raw=None, missing_lags=tuple(missing))

    demand_value = float(demand.at(t))
    if not math.isfinite(demand_value):
        raise ValueError(f"demand of period {t} is not finite: {demand_value}")
    raw = np.array([demand_value, *prices], dtype=np.float64)
    raw.setflags(write=False)
    return Observation(t=t, lags=lag_tuple, raw=raw, missing_lags=())


@dataclass(frozen=True, eq=False)
class ObservationNormalizer:
    """Fixed per-feature affine normaliser ``(raw - mean) / scale``.

    Attributes:
        mean: float64 array of shape ``(d,)``.
        scale: float64 array of shape ``(d,)``, strictly positive.
    """

    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        """Validate and freeze the parameters.

        Raises:
            ValueError: If shapes differ or are not 1-D and non-empty, a value
                is non-finite, or a scale is not strictly positive.
        """
        mean = np.array(self.mean, dtype=np.float64)
        scale = np.array(self.scale, dtype=np.float64)
        if mean.ndim != 1 or mean.size == 0 or mean.shape != scale.shape:
            raise ValueError(
                f"mean and scale must be non-empty 1-D arrays of equal shape, "
                f"got {mean.shape} and {scale.shape}"
            )
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(scale))):
            raise ValueError("mean and scale must be finite")
        if not np.all(scale > 0):
            raise ValueError(f"scale must be strictly > 0, got {scale}")
        mean.setflags(write=False)
        scale.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def for_market(
        cls, lags: Sequence[int], price_low: float, price_high: float
    ) -> Self:
        """Create the normaliser for a market's bid bounds.

        Demand uses ``(DEMAND_SCALING_MEAN, DEMAND_SCALING_STD)``. Each price
        feature uses ``mean = (low + high) / 2`` and ``scale = (high - low) / 2``.

        Args:
            lags: Lag set, must be in ``ALLOWED_LAG_SETS``.
            price_low: Lower bid bound of the market.
            price_high: Upper bid bound of the market.

        Returns:
            The normaliser for ``1 + len(lags)`` features.

        Raises:
            ValueError: If the lag set is not allowed, a bound is non-finite,
                or ``price_high <= price_low``.
        """
        lag_tuple = _validate_lags(lags)
        low = float(price_low)
        high = float(price_high)
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"price bounds must be finite, got [{low}, {high}]")
        if high <= low:
            raise ValueError(f"price_high must be > price_low, got [{low}, {high}]")
        n_prices = len(lag_tuple)
        mean = np.array(
            [DEMAND_SCALING_MEAN] + [(low + high) / 2.0] * n_prices, dtype=np.float64
        )
        scale = np.array(
            [DEMAND_SCALING_STD] + [(high - low) / 2.0] * n_prices, dtype=np.float64
        )
        return cls(mean=mean, scale=scale)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Normalise one observation or a batch.

        Args:
            raw: Array of shape ``(d,)`` or ``(n, d)``.

        Returns:
            float32 array ``(raw - mean) / scale`` of the same shape.

        Raises:
            ValueError: If the shape does not match or ``raw`` contains NaN or
                Inf.
        """
        x = np.asarray(raw, dtype=np.float64)
        d = self.mean.shape[0]
        if x.ndim not in (1, 2) or x.shape[-1] != d:
            raise ValueError(f"raw must have shape ({d},) or (n, {d}), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("raw contains NaN or Inf")
        return ((x - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        """Serialise the parameters for the run log.

        Returns:
            ``{"mean": [...], "scale": [...]}`` as Python floats.
        """
        return {
            "mean": [float(v) for v in self.mean],
            "scale": [float(v) for v in self.scale],
        }
