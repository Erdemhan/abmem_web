"""Synthetic hourly demand of a scenario (PLAN 2.8, spec demand_generation).

A :class:`DemandProfile` describes the demand of a scenario as two segments,
day and night, each with a truncated marginal distribution (Normal, Uniform
or Log-normal on ``[min, max]``), plus an AR(1) correlation of consecutive
hours and a seed of its own.

Method (Gaussian copula with an AR(1) latent process). Period ``t`` (1-based)
has hour ``h = (t - 1) mod 24`` and belongs to the day segment when
``day_start_hour <= h < day_end_hour``, otherwise to the night segment.
The latent process is::

    z_1 = eps_1
    z_t = phi * z_{t-1} + sqrt(1 - phi^2) * eps_t,   eps_t ~ N(0, 1)

with the innovations drawn from ``numpy.random.default_rng(seed)`` in period
order. Then ``u_t = Phi(z_t)`` (clipped to ``[1e-12, 1 - 1e-12]``) and
``d_t = F^-1(u_t)`` with ``F`` the truncated marginal of the segment. The
demand is rounded to 0.001 MWh (the clearing quantum) and clipped to the
segment bounds.

No step depends on the total number of periods, so the series is never
stored: ``generate_demand(p, n)[:m] == generate_demand(p, m)`` bit for bit
for every ``m <= n``. Only numpy and ``statistics.NormalDist`` are used.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Final

import numpy as np
from django.core.exceptions import ValidationError

__all__ = [
    "PROFILE_VERSION",
    "Distribution",
    "Segment",
    "DailyShape",
    "Seasons",
    "DemandProfile",
    "day_weights",
    "generate_demand",
    "night_mask",
    "normalize_profile",
]

PROFILE_VERSION: Final[int] = 1
SEED_UPPER_BOUND: Final[int] = 2**31
U_EPSILON: Final[float] = 1e-12
DEMAND_DECIMALS: Final[int] = 3
HOURS_PER_DAY: Final[int] = 24
MAX_RAMP_HOURS: Final[int] = 12
MAX_SEASON_TRANSITION_DAYS: Final[int] = 90
MAX_SUMMER_END_PERIOD: Final[int] = 1_000_000

_STANDARD_NORMAL: Final[NormalDist] = NormalDist()

PROFILE_KEYS: Final[tuple[str, ...]] = (
    "version",
    "seed",
    "day_start_hour",
    "day_end_hour",
    "autocorrelation",
    "day",
    "night",
    "morning_ramp_hours",
    "evening_ramp_hours",
    "seasons",
)
SEASON_KEYS: Final[tuple[str, ...]] = ("summer_end_period", "transition_days", "winter")
SEGMENT_KEYS: Final[tuple[str, ...]] = ("distribution", "mean", "std", "min", "max")
SEGMENT_NAMES: Final[tuple[str, ...]] = ("day", "night")


class Distribution(str, enum.Enum):
    """Marginal distribution of a demand segment."""

    NORMAL = "NORMAL"
    UNIFORM = "UNIFORM"
    LOGNORMAL = "LOGNORMAL"


def _uses_moments(distribution: Distribution) -> bool:
    """Return True when the distribution is parameterised by mean and std.

    Args:
        distribution: The distribution.

    Returns:
        True for NORMAL and LOGNORMAL.
    """
    return distribution in (Distribution.NORMAL, Distribution.LOGNORMAL)


@dataclass(frozen=True)
class Segment:
    """Marginal distribution of the demand of the day or the night hours.

    Attributes:
        distribution: NORMAL, UNIFORM or LOGNORMAL.
        min: Lower truncation bound in MWh, > 0.
        max: Upper truncation bound in MWh, > ``min``.
        mean: Mean of the untruncated distribution in MWh (NORMAL and
            LOGNORMAL only, None for UNIFORM).
        std: Standard deviation of the untruncated distribution in MWh
            (NORMAL and LOGNORMAL only, None for UNIFORM).
    """

    distribution: Distribution
    min: float
    max: float
    mean: float | None = None
    std: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form of the segment.

        Returns:
            ``{"distribution", "mean", "std", "min", "max"}``; ``mean`` and
            ``std`` are left out for UNIFORM.
        """
        data: dict[str, Any] = {"distribution": self.distribution.value}
        if _uses_moments(self.distribution):
            data["mean"] = self.mean
            data["std"] = self.std
        data["min"] = self.min
        data["max"] = self.max
        return data

    def errors(self, prefix: str) -> dict[str, list[str]]:
        """Return the semantic errors of the segment.

        Args:
            prefix: Error key prefix, for example ``"day"``.

        Returns:
            Mapping ``"<prefix>.<field>"`` to messages, empty when valid.
        """
        errors: dict[str, list[str]] = {}
        label = _segment_label(prefix)

        def add(field: str, message: str) -> None:
            errors.setdefault(f"{prefix}.{field}", []).append(f"{label}: {message}")

        if not self.min > 0:
            add("min", "the minimum must be greater than 0 MWh.")
        if not self.max > self.min:
            add("max", "the maximum must be greater than the minimum.")
        if _uses_moments(self.distribution):
            if self.std is None or not self.std > 0:
                add("std", "the standard deviation must be greater than 0 MWh.")
            if self.mean is None or not self.mean > 0:
                add("mean", "the mean must be greater than 0 MWh.")
            elif not self.min < self.mean < self.max:
                add(
                    "mean",
                    "the mean must lie strictly between the minimum and the maximum.",
                )
        return errors

    def quantile(self, u: float) -> float:
        """Return the inverse CDF of the truncated marginal at ``u``.

        Args:
            u: Probability in ``(0, 1)``.

        Returns:
            The quantile in ``[min, max]`` (MWh).
        """
        if self.distribution is Distribution.UNIFORM:
            value = self.min + u * (self.max - self.min)
        elif self.distribution is Distribution.NORMAL:
            assert self.mean is not None and self.std is not None
            a = (self.min - self.mean) / self.std
            b = (self.max - self.mean) / self.std
            value = self.mean + self.std * _truncated_standard_normal_quantile(a, b, u)
        else:
            assert self.mean is not None and self.std is not None
            sigma2 = math.log1p((self.std / self.mean) ** 2)
            sigma = math.sqrt(sigma2)
            mu = math.log(self.mean) - sigma2 / 2.0
            a = (math.log(self.min) - mu) / sigma
            b = (math.log(self.max) - mu) / sigma
            value = math.exp(mu + sigma * _truncated_standard_normal_quantile(a, b, u))
        return min(max(value, self.min), self.max)

    def moments(self) -> tuple[float, float]:
        """Return the mean and variance of the truncated marginal (FORMULATION EQ-02b).

        UNIFORM on ``[a, b]``: ``(a + b) / 2`` and ``(b - a)^2 / 12``.
        NORMAL with location ``m`` and scale ``s`` truncated to ``[a, b]``,
        ``alpha = (a - m) / s``, ``beta = (b - m) / s``,
        ``Z = Phi(beta) - Phi(alpha)``: mean
        ``m + s (phi(alpha) - phi(beta)) / Z`` and variance
        ``s^2 [1 + (alpha phi(alpha) - beta phi(beta)) / Z
        - ((phi(alpha) - phi(beta)) / Z)^2]``.
        LOGNORMAL ``X = exp(Y)``, ``Y ~ N(mu, sigma^2)`` truncated to
        ``[a, b]``: raw moments ``E[X^k] = exp(k mu + k^2 sigma^2 / 2)
        (Phi(beta - k sigma) - Phi(alpha - k sigma)) / Z`` with ``alpha``,
        ``beta`` taken on ``ln a``, ``ln b``.

        Returns:
            ``(mean, variance)`` in MWh and MWh^2.
        """
        if self.distribution is Distribution.UNIFORM:
            return (self.min + self.max) / 2.0, (self.max - self.min) ** 2 / 12.0
        assert self.mean is not None and self.std is not None
        cdf = _STANDARD_NORMAL.cdf
        pdf = _STANDARD_NORMAL.pdf
        if self.distribution is Distribution.NORMAL:
            alpha = (self.min - self.mean) / self.std
            beta = (self.max - self.mean) / self.std
            z = cdf(beta) - cdf(alpha)
            ratio = (pdf(alpha) - pdf(beta)) / z
            mean = self.mean + self.std * ratio
            variance = self.std**2 * (
                1.0 + (alpha * pdf(alpha) - beta * pdf(beta)) / z - ratio**2
            )
            return mean, max(variance, 0.0)
        sigma2 = math.log1p((self.std / self.mean) ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(self.mean) - sigma2 / 2.0
        alpha = (math.log(self.min) - mu) / sigma
        beta = (math.log(self.max) - mu) / sigma
        z = cdf(beta) - cdf(alpha)

        def raw_moment(k: int) -> float:
            return (
                math.exp(k * mu + k * k * sigma2 / 2.0)
                * (cdf(beta - k * sigma) - cdf(alpha - k * sigma))
                / z
            )

        mean = raw_moment(1)
        return mean, max(raw_moment(2) - mean**2, 0.0)


def _truncated_standard_normal_quantile(a: float, b: float, u: float) -> float:
    """Return the quantile of a standard normal truncated to ``[a, b]``.

    When the interval lies in the upper half (``a > 0``) the mirrored lower
    tail is used, so the CDF differences keep their precision.

    Args:
        a: Lower bound in standard units.
        b: Upper bound in standard units, ``b > a``.
        u: Probability in ``(0, 1)``.

    Returns:
        ``Phi^-1(Phi(a) + u (Phi(b) - Phi(a)))``, clipped to ``[a, b]``.
    """
    if a > 0:
        return -_truncated_standard_normal_quantile(-b, -a, 1.0 - u)
    cdf_a = _STANDARD_NORMAL.cdf(a)
    cdf_b = _STANDARD_NORMAL.cdf(b)
    p = cdf_a + u * (cdf_b - cdf_a)
    tiny = math.ulp(0.0)
    p = min(max(p, tiny), 1.0 - 2.0**-53)
    return min(max(_STANDARD_NORMAL.inv_cdf(p), a), b)


def _is_number(value: Any) -> bool:
    """Return True for a finite int or float that is not a bool.

    Args:
        value: Any value.

    Returns:
        Whether ``value`` is a usable number.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _number(raw: Any) -> float | None:
    """Convert a JSON number or numeric string to a finite float.

    Args:
        raw: Payload value.

    Returns:
        The float, or None when ``raw`` is not a finite number.
    """
    if isinstance(raw, str):
        try:
            value = float(raw.strip())
        except ValueError:
            return None
        return value if math.isfinite(value) else None
    if _is_number(raw):
        return float(raw)
    return None


def _integer(raw: Any) -> int | None:
    """Convert a JSON integer (or integral float or digit string) to int.

    Args:
        raw: Payload value.

    Returns:
        The integer, or None when ``raw`` is not an integer.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw) if raw.is_integer() else None
    if isinstance(raw, str):
        text = raw.strip()
        if text.lstrip("-").isdigit():
            return int(text)
    return None


def _segment_label(prefix: str) -> str:
    """Return the message label of a segment prefix.

    Args:
        prefix: ``"day"``, ``"night"`` or ``"seasons.winter.day"``.

    Returns:
        ``"Day"``, ``"Night"`` or ``"Winter day"``.
    """
    parts = prefix.split(".")
    if "winter" in parts[:-1]:
        return f"Winter {parts[-1]}"
    return parts[-1].capitalize()


def _parse_segment(
    raw: Any, prefix: str, errors: dict[str, list[str]]
) -> Segment | None:
    """Parse one segment mapping, collecting field-keyed errors.

    Args:
        raw: Payload value of ``day`` or ``night``.
        prefix: ``"day"`` or ``"night"``.
        errors: Error mapping, updated in place.

    Returns:
        The segment, or None when a field is missing or malformed.
    """
    label = _segment_label(prefix)

    def add(field: str, message: str) -> None:
        key = f"{prefix}.{field}" if field else prefix
        errors.setdefault(key, []).append(f"{label}: {message}")

    if not isinstance(raw, Mapping):
        add("", "the segment must be an object with distribution, min and max.")
        return None
    unknown = sorted(str(key) for key in raw if key not in SEGMENT_KEYS)
    if unknown:
        add("", f"unknown field(s) {', '.join(unknown)}.")
    raw_distribution = raw.get("distribution")
    try:
        distribution = Distribution(
            raw_distribution.strip().upper()
            if isinstance(raw_distribution, str)
            else raw_distribution
        )
    except ValueError:
        add(
            "distribution",
            "the distribution must be one of "
            + ", ".join(member.value for member in Distribution)
            + ".",
        )
        return None
    needed = (
        ("mean", "std", "min", "max") if _uses_moments(distribution) else ("min", "max")
    )
    values: dict[str, float] = {}
    ok = True
    for field in needed:
        if field not in raw or raw.get(field) in (None, ""):
            add(
                field, f"{field} is required for the {distribution.value} distribution."
            )
            ok = False
            continue
        value = _number(raw[field])
        if value is None:
            add(field, f"{field} must be a number (MWh).")
            ok = False
            continue
        values[field] = value
    if not ok:
        return None
    return Segment(
        distribution=distribution,
        min=values["min"],
        max=values["max"],
        mean=values.get("mean"),
        std=values.get("std"),
    )


@dataclass(frozen=True)
class DailyShape:
    """Day and night marginals of one season with their hour boundaries.

    Attributes:
        day_start_hour: First day hour on the 24-hour clock (0..23).
        day_end_hour: Exclusive end of the day hours (1..24).
        day: Marginal of the day hours.
        night: Marginal of the night hours.
        morning_ramp_hours: Width ``k_m`` in hours of the transition centred
            on ``day_start_hour`` (0 = abrupt switch).
        evening_ramp_hours: Width ``k_e`` in hours of the transition centred
            on ``day_end_hour`` (0 = abrupt switch).
    """

    day_start_hour: int
    day_end_hour: int
    day: Segment
    night: Segment
    morning_ramp_hours: int = 0
    evening_ramp_hours: int = 0

    @property
    def day_hours(self) -> int:
        """Number of day hours per 24 hours."""
        return self.day_end_hour - self.day_start_hour

    @property
    def night_hours(self) -> int:
        """Number of night hours per 24 hours."""
        return HOURS_PER_DAY - self.day_hours

    def day_weights(self) -> np.ndarray:
        """Return the day weight ``w_h`` of every hour of the day (EQ-02c).

        Returns:
            ``float64`` array of length 24, see :func:`day_weights`.
        """
        return day_weights(
            self.day_start_hour,
            self.day_end_hour,
            self.morning_ramp_hours,
            self.evening_ramp_hours,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form of the shape (hour keys and both segments)."""
        return {
            "day_start_hour": self.day_start_hour,
            "day_end_hour": self.day_end_hour,
            "morning_ramp_hours": self.morning_ramp_hours,
            "evening_ramp_hours": self.evening_ramp_hours,
            "day": self.day.to_dict(),
            "night": self.night.to_dict(),
        }

    def errors(self, prefix: str = "") -> dict[str, list[str]]:
        """Return the semantic errors of the shape.

        Args:
            prefix: Key prefix such as ``"seasons.winter."`` (empty for the
                top-level shape).

        Returns:
            Field-keyed messages, empty when the shape is valid.
        """
        errors: dict[str, list[str]] = {}
        if not 0 <= self.day_start_hour <= 23:
            errors[f"{prefix}day_start_hour"] = ["Day start hour must be from 0 to 23."]
        if not 1 <= self.day_end_hour <= 24:
            errors[f"{prefix}day_end_hour"] = ["Day end hour must be from 1 to 24."]
        elif f"{prefix}day_start_hour" not in errors and not 1 <= self.day_hours <= 23:
            errors[f"{prefix}day_end_hour"] = [
                "Day end hour must be after the day start hour, and day and night "
                "must each have at least one hour (day length 1 to 23 h)."
            ]
        for field, label, width in (
            ("morning_ramp_hours", "Morning transition", self.morning_ramp_hours),
            ("evening_ramp_hours", "Evening transition", self.evening_ramp_hours),
        ):
            if not 0 <= width <= MAX_RAMP_HOURS:
                errors[f"{prefix}{field}"] = [
                    f"{label} must be from 0 to {MAX_RAMP_HOURS} hours."
                ]
        if not errors:
            limit = 2 * min(self.day_hours, self.night_hours)
            if self.morning_ramp_hours + self.evening_ramp_hours > limit:
                errors[f"{prefix}evening_ramp_hours"] = [
                    "The two transitions overlap. Morning plus evening transition "
                    f"must be at most {limit} hours (twice the shorter of day and "
                    "night)."
                ]
        errors.update(self.day.errors(f"{prefix}day"))
        errors.update(self.night.errors(f"{prefix}night"))
        return errors

    @classmethod
    def parse(
        cls, data: Mapping[str, Any], prefix: str, errors: dict[str, list[str]]
    ) -> DailyShape | None:
        """Parse the shape keys of ``data``, collecting errors in ``errors``.

        Args:
            data: Mapping with the shape keys.
            prefix: Error key prefix (``""`` or ``"seasons.winter."``).
            errors: Field-keyed errors, updated in place.

        Returns:
            The shape when every key parsed, otherwise None. Semantic checks
            are left to :meth:`errors`.
        """
        ints: dict[str, int] = {}
        for field, label in (
            ("day_start_hour", "Day start hour"),
            ("day_end_hour", "Day end hour"),
        ):
            value = _integer(data.get(field))
            if value is None:
                errors[f"{prefix}{field}"] = [
                    f"{label} is required and must be an integer."
                ]
            else:
                ints[field] = value
        for field, label in (
            ("morning_ramp_hours", "Morning transition"),
            ("evening_ramp_hours", "Evening transition"),
        ):
            raw = data.get(field, 0)
            value = 0 if raw is None or raw == "" else _integer(raw)
            if value is None:
                errors[f"{prefix}{field}"] = [
                    f"{label} must be a whole number of hours."
                ]
            else:
                ints[field] = value
        segments = {
            name: _parse_segment(data.get(name), f"{prefix}{name}", errors)
            for name in SEGMENT_NAMES
        }
        if len(ints) < 4 or segments["day"] is None or segments["night"] is None:
            for name in SEGMENT_NAMES:
                if segments[name] is not None:
                    errors.update(segments[name].errors(f"{prefix}{name}"))
            return None
        return cls(
            day_start_hour=ints["day_start_hour"],
            day_end_hour=ints["day_end_hour"],
            day=segments["day"],
            night=segments["night"],
            morning_ramp_hours=ints["morning_ramp_hours"],
            evening_ramp_hours=ints["evening_ramp_hours"],
        )


@dataclass(frozen=True)
class Seasons:
    """Optional summer/winter split of a profile (EQ-02c).

    Periods ``1..summer_end_period`` are summer and later periods winter.
    The switch is centred on the end of period ``summer_end_period`` and
    spread over ``transition_days`` days (0 = abrupt switch).

    Attributes:
        summer_end_period: Last summer period, >= 1.
        transition_days: Width of the season transition in days.
        winter: Day/night shape of the winter periods.
    """

    summer_end_period: int
    transition_days: int
    winter: DailyShape

    def winter_weight(self, periods: int) -> np.ndarray:
        """Return the winter weight ``v_t`` of the periods ``1..periods``.

        Period ``t`` is represented by its centre ``t - 0.5`` hours and
        ``v_t = r(t - 0.5 - S, 24 W)`` with :func:`_ramp`, ``S`` the summer
        end period and ``W`` the transition days.

        Args:
            periods: Number of periods.

        Returns:
            ``float64`` array of length ``periods`` in ``[0, 1]``.
        """
        width = HOURS_PER_DAY * self.transition_days
        return np.array(
            [
                _ramp(t - 0.5 - self.summer_end_period, width)
                for t in range(1, periods + 1)
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form of the season split."""
        return {
            "summer_end_period": self.summer_end_period,
            "transition_days": self.transition_days,
            "winter": self.winter.to_dict(),
        }

    def errors(self, prefix: str = "seasons.") -> dict[str, list[str]]:
        """Return the semantic errors of the season split.

        Args:
            prefix: Error key prefix.

        Returns:
            Field-keyed messages, empty when valid.
        """
        errors: dict[str, list[str]] = {}
        if not 1 <= self.summer_end_period <= MAX_SUMMER_END_PERIOD:
            errors[f"{prefix}summer_end_period"] = [
                f"Summer end period must be from 1 to {MAX_SUMMER_END_PERIOD}."
            ]
        if not 0 <= self.transition_days <= MAX_SEASON_TRANSITION_DAYS:
            errors[f"{prefix}transition_days"] = [
                "Season transition must be from 0 to "
                f"{MAX_SEASON_TRANSITION_DAYS} days."
            ]
        elif (
            f"{prefix}summer_end_period" not in errors
            and HOURS_PER_DAY * self.transition_days > 2 * self.summer_end_period
        ):
            errors[f"{prefix}transition_days"] = [
                "The season transition starts before period 1. Half of the "
                "transition (in hours) must not exceed the summer end period."
            ]
        errors.update(self.winter.errors(f"{prefix}winter."))
        return errors

    @classmethod
    def parse(
        cls, data: Any, prefix: str, errors: dict[str, list[str]]
    ) -> Seasons | None:
        """Parse a ``seasons`` mapping, collecting errors in ``errors``.

        Args:
            data: The ``seasons`` value.
            prefix: Error key prefix (``"seasons."``).
            errors: Field-keyed errors, updated in place.

        Returns:
            The season split, or None when a key did not parse.
        """
        if not isinstance(data, Mapping):
            errors[prefix.rstrip(".")] = ["Seasons must be an object or null."]
            return None
        unknown = sorted(str(key) for key in data if key not in SEASON_KEYS)
        if unknown:
            errors[prefix.rstrip(".")] = [
                f"Unknown seasons field(s): {', '.join(unknown)}."
            ]
        ints: dict[str, int] = {}
        for field, label, default in (
            ("summer_end_period", "Summer end period", None),
            ("transition_days", "Season transition", 0),
        ):
            raw = data.get(field, default)
            value = (
                default
                if default is not None and (raw is None or raw == "")
                else _integer(raw)
            )
            if value is None:
                errors[f"{prefix}{field}"] = [
                    f"{label} is required and must be a whole number."
                ]
            else:
                ints[field] = value
        winter_raw = data.get("winter")
        if not isinstance(winter_raw, Mapping):
            errors[f"{prefix}winter"] = ["Winter settings are required."]
            return None
        winter = DailyShape.parse(winter_raw, f"{prefix}winter.", errors)
        if winter is None or len(ints) < 2:
            return None
        return cls(
            summer_end_period=ints["summer_end_period"],
            transition_days=ints["transition_days"],
            winter=winter,
        )


@dataclass(frozen=True)
class DemandProfile:
    """Synthetic demand profile of a scenario (schema version 1).

    Attributes:
        seed: Seed of the demand generator, in ``[0, 2**31)``; independent
            of the run seed.
        autocorrelation: AR(1) coefficient ``phi`` of the latent process, in
            ``[0, 1)``.
        summer: Day/night shape of the whole run, or of the summer periods
            when ``seasons`` is set (the top-level keys of the JSON form).
        seasons: Optional summer/winter split, None when off.
    """

    seed: int
    autocorrelation: float
    summer: DailyShape
    seasons: Seasons | None = None

    @property
    def day_start_hour(self) -> int:
        """First day hour of the (summer) shape."""
        return self.summer.day_start_hour

    @property
    def day_end_hour(self) -> int:
        """Exclusive end of the day hours of the (summer) shape."""
        return self.summer.day_end_hour

    @property
    def day(self) -> Segment:
        """Day marginal of the (summer) shape."""
        return self.summer.day

    @property
    def night(self) -> Segment:
        """Night marginal of the (summer) shape."""
        return self.summer.night

    @property
    def day_hours(self) -> int:
        """Number of day hours per 24 hours of the (summer) shape."""
        return self.summer.day_hours

    @property
    def night_hours(self) -> int:
        """Number of night hours per 24 hours of the (summer) shape."""
        return self.summer.night_hours

    def segments(self) -> tuple[Segment, ...]:
        """Return the segments in weight order (EQ-02c).

        Returns:
            ``(summer night, summer day)``, followed by
            ``(winter night, winter day)`` when seasons are on.
        """
        own = (self.summer.night, self.summer.day)
        if self.seasons is None:
            return own
        return own + (self.seasons.winter.night, self.seasons.winter.day)

    def weights(self, periods: int) -> np.ndarray:
        """Return the weight of every segment in every period (EQ-02c).

        Without seasons ``a_t = (1 - w_h, w_h)``. With seasons
        ``a_t = ((1 - v_t)(1 - w_h^S), (1 - v_t) w_h^S, v_t (1 - w_h^W),
        v_t w_h^W)``, with ``h = (t - 1) mod 24``, ``w^S``, ``w^W`` the day
        weights of the summer and winter shapes and ``v_t`` the winter
        weight. Each row sums to 1.

        Args:
            periods: Number of periods.

        Returns:
            ``float64`` array of shape ``(periods, len(segments()))``.
        """
        hours = np.arange(periods, dtype=np.int64) % HOURS_PER_DAY
        w_summer = self.summer.day_weights()[hours]
        summer = np.column_stack((1.0 - w_summer, w_summer))
        if self.seasons is None:
            return summer
        w_winter = self.seasons.winter.day_weights()[hours]
        winter = np.column_stack((1.0 - w_winter, w_winter))
        v = self.seasons.winter_weight(periods)[:, None]
        return np.hstack(((1.0 - v) * summer, v * winter))

    def night_mask(self, periods: int) -> np.ndarray:
        """Return which periods are night hours of their dominant season.

        Args:
            periods: Number of periods.

        Returns:
            Boolean array of length ``periods``, used to shade previews.
        """
        mask = night_mask(periods, self.summer.day_start_hour, self.summer.day_end_hour)
        if self.seasons is None:
            return mask
        winter = night_mask(
            periods,
            self.seasons.winter.day_start_hour,
            self.seasons.winter.day_end_hour,
        )
        return np.where(self.seasons.winter_weight(periods) >= 0.5, winter, mask)

    def scaling(self, periods: int | None = None) -> tuple[float, float]:
        """Return the observation scaling constants of the profile (FORMULATION EQ-02b).

        The demand of period ``t`` is ``X_t = sum_i a_ti Q_i(U)`` with ``U``
        uniform, ``Q_i`` the quantile functions of :meth:`segments` and
        ``a_t`` the row of :meth:`weights`. Hence ``E[X_t] = a_t . mu`` and
        ``E[X_t^2] = a_t' G a_t`` with ``G_ij = int_0^1 Q_i(u) Q_j(u) du``
        (diagonal from :meth:`Segment.moments`, off-diagonal by Gauss-Legendre
        quadrature, computed only when a period mixes segments). The
        constants are the moments of the equal-weight mixture of the rows:
        the 24 hours of a day without seasons, the periods ``1..periods``
        with seasons. Without ramps and seasons this equals
        ``w_d mu_d + w_n mu_n`` and
        ``w_d (v_d + mu_d^2) + w_n (v_n + mu_n^2) - mean^2``.

        Args:
            periods: Number of periods of the run. Required with seasons,
                ignored without.

        Returns:
            ``(mean, std)`` of the generated demand in MWh.

        Raises:
            ValueError: If seasons are on and ``periods`` is not positive.
        """
        if self.seasons is None:
            rows = self.weights(HOURS_PER_DAY)
        else:
            if periods is None or periods < 1:
                raise ValueError(
                    f"a seasonal profile needs a positive number of periods, got {periods}"
                )
            rows = self.weights(periods)
        segments = self.segments()
        moments = [segment.moments() for segment in segments]
        means = np.array([mean for mean, _ in moments], dtype=np.float64)
        gram = np.diag([var + mean**2 for mean, var in moments])
        used = rows > 0.0
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                if bool((used[:, i] & used[:, j]).any()):
                    gram[i, j] = gram[j, i] = _quantile_cross_moment(
                        segments[i], segments[j]
                    )
        mean = float(np.mean(rows @ means))
        second = float(np.mean(np.einsum("ti,ij,tj->t", rows, gram, rows)))
        return mean, math.sqrt(max(second - mean**2, 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form stored in ``Simulation.demand_profile``.

        Returns:
            The version-1 profile mapping. The (summer) shape keys are at the
            top level and ``seasons`` is null when the split is off.
        """
        return {
            "version": PROFILE_VERSION,
            "seed": self.seed,
            "autocorrelation": self.autocorrelation,
            **self.summer.to_dict(),
            "seasons": None if self.seasons is None else self.seasons.to_dict(),
        }

    def errors(self) -> dict[str, list[str]]:
        """Return the semantic errors of the profile.

        Returns:
            Field-keyed messages (``seed``, ``day_end_hour``, ``day.mean``,
            ``seasons.winter.night.std``, ...), empty when the profile is
            valid.
        """
        errors: dict[str, list[str]] = {}
        if not 0 <= self.seed < SEED_UPPER_BOUND:
            errors["seed"] = [
                f"Demand seed must be an integer from 0 to {SEED_UPPER_BOUND - 1}."
            ]
        if not 0 <= self.autocorrelation < 1:
            errors["autocorrelation"] = [
                "Autocorrelation must be at least 0 and less than 1."
            ]
        errors.update(self.summer.errors())
        if self.seasons is not None:
            errors.update(self.seasons.errors())
        return errors

    def validate(self) -> None:
        """Check the profile.

        Raises:
            ValidationError: Field-keyed errors of :meth:`errors`.
        """
        errors = self.errors()
        if errors:
            raise ValidationError(errors)

    @classmethod
    def from_dict(cls, data: Any) -> DemandProfile:
        """Parse and validate a profile mapping (``version`` is optional).

        Args:
            data: Mapping with the version-1 keys. Numbers may be JSON numbers
                or numeric strings. ``morning_ramp_hours``,
                ``evening_ramp_hours`` default to 0 and ``seasons`` to null.

        Returns:
            The validated profile.

        Raises:
            ValidationError: Field-keyed errors (``demand_profile`` for the
                whole object, ``seed``, ``day.std``,
                ``seasons.summer_end_period``, ...).
        """
        if not isinstance(data, Mapping):
            raise ValidationError(
                {"demand_profile": ["The demand profile must be an object."]}
            )
        errors: dict[str, list[str]] = {}
        unknown = sorted(str(key) for key in data if key not in PROFILE_KEYS)
        if unknown:
            errors["demand_profile"] = [
                f"Unknown demand profile field(s): {', '.join(unknown)}."
            ]
        if "version" in data and (
            isinstance(data["version"], bool) or data["version"] != PROFILE_VERSION
        ):
            errors["version"] = [
                f"Unsupported demand profile version {data['version']!r}; "
                f"expected {PROFILE_VERSION}."
            ]
        seed = _integer(data.get("seed"))
        if seed is None:
            errors["seed"] = ["Demand seed is required and must be an integer."]
        phi = _number(data.get("autocorrelation"))
        if phi is None:
            errors["autocorrelation"] = [
                "Autocorrelation is required and must be a number."
            ]
        summer = DailyShape.parse(data, "", errors)
        seasons_raw = data.get("seasons")
        seasons = (
            None
            if seasons_raw is None
            else Seasons.parse(seasons_raw, "seasons.", errors)
        )
        if (
            errors
            or seed is None
            or phi is None
            or summer is None
            or (seasons_raw is not None and seasons is None)
        ):
            # Report the semantic errors of the parts that did parse as well.
            if summer is not None:
                errors.update(summer.errors())
            if seasons is not None:
                errors.update(seasons.errors())
            raise ValidationError(errors)
        profile = cls(seed=seed, autocorrelation=phi, summer=summer, seasons=seasons)
        profile.validate()
        return profile

PAYLOAD_KEY: Final[str] = "demand_profile"


def normalize_profile(raw: Any, key: str = PAYLOAD_KEY) -> dict[str, Any] | None:
    """Validate a profile value of a request or file and return its stored form.

    Every page and the YAML import use this function, so a profile is
    accepted in exactly one way.

    Args:
        raw: None (demand file) or a version-1 profile mapping.
        key: Payload key of the profile; error keys are ``key`` for the whole
            object and ``"<key>.<field>"`` for a field
            (``demand_profile.day.std``).

    Returns:
        None, or ``DemandProfile.from_dict(raw).to_dict()``.

    Raises:
        ValidationError: Field-keyed errors as described above.
    """
    if raw is None:
        return None
    try:
        return DemandProfile.from_dict(raw).to_dict()
    except ValidationError as exc:
        raise ValidationError(
            {
                (key if name == PAYLOAD_KEY else f"{key}.{name}"): messages
                for name, messages in exc.message_dict.items()
            }
        ) from None


def night_mask(periods: int, day_start_hour: int, day_end_hour: int) -> np.ndarray:
    """Return which of periods ``1..periods`` are night hours.

    Args:
        periods: Number of periods.
        day_start_hour: First day hour.
        day_end_hour: Exclusive end of the day hours.

    Returns:
        Boolean array of length ``periods``; True for night hours.
    """
    hours = np.arange(periods, dtype=np.int64) % HOURS_PER_DAY
    return ~((hours >= day_start_hour) & (hours < day_end_hour))


def _ramp(x: float, width: int) -> float:
    """Raised-cosine step from 0 to 1 over ``[-width / 2, width / 2]``.

    Args:
        x: Signed distance in hours from the boundary (positive on the day
            side). Hour centres give half-integer values, never 0.
        width: Transition width in hours, 0 for an abrupt step.

    Returns:
        ``0`` for ``x <= -width / 2``, ``1`` for ``x >= width / 2``, else
        ``(1 + sin(pi x / width)) / 2``.
    """
    half = width / 2.0
    if x <= -half:
        return 0.0
    if x >= half:
        return 1.0
    return 0.5 * (1.0 + math.sin(math.pi * x / width))


def day_weights(
    day_start_hour: int,
    day_end_hour: int,
    morning_ramp_hours: int = 0,
    evening_ramp_hours: int = 0,
) -> np.ndarray:
    """Return the day weight ``w_h`` of the hours ``h = 0..23`` (EQ-02c).

    The hour is represented by its centre ``h + 0.5``. A day hour has
    ``w_h = min(r(a, k_m), r(c, k_e))`` with ``a`` its distance after the
    day start and ``c`` its distance before the day end. A night hour has
    ``w_h = max(r(-c', k_m), r(-a', k_e))`` with ``c'`` its distance before
    the next day start and ``a'`` its distance after the last day end, both
    on the 24-hour circle. ``r`` is :func:`_ramp`. With both widths 0 the
    weights are exactly 1 for day hours and 0 for night hours.

    Args:
        day_start_hour: First day hour.
        day_end_hour: Exclusive end of the day hours.
        morning_ramp_hours: Width ``k_m`` of the morning transition.
        evening_ramp_hours: Width ``k_e`` of the evening transition.

    Returns:
        ``float64`` array of length 24 with values in ``[0, 1]``.
    """
    weights = np.empty(HOURS_PER_DAY, dtype=np.float64)
    for hour in range(HOURS_PER_DAY):
        centre = hour + 0.5
        if day_start_hour <= hour < day_end_hour:
            weights[hour] = min(
                _ramp(centre - day_start_hour, morning_ramp_hours),
                _ramp(day_end_hour - centre, evening_ramp_hours),
            )
        else:
            before_start = (day_start_hour - centre) % HOURS_PER_DAY
            after_end = (centre - day_end_hour) % HOURS_PER_DAY
            weights[hour] = max(
                _ramp(-before_start, morning_ramp_hours),
                _ramp(-after_end, evening_ramp_hours),
            )
    return weights


_QUADRATURE_PANELS: Final[int] = 64
_QUADRATURE_NODES: Final[int] = 16


def _quantile_cross_moment(first: Segment, second: Segment) -> float:
    """Return ``int_0^1 Q_1(u) Q_2(u) du`` of two segments (EQ-02b).

    Composite Gauss-Legendre quadrature with 64 panels of 16 nodes. The
    quantile functions of the truncated marginals are bounded and smooth on
    ``(0, 1)``, so the rule converges quickly.

    Args:
        first: First segment.
        second: Second segment.

    Returns:
        The cross moment in MWh^2.
    """
    nodes, weights = np.polynomial.legendre.leggauss(_QUADRATURE_NODES)
    width = 1.0 / _QUADRATURE_PANELS
    total = 0.0
    for panel in range(_QUADRATURE_PANELS):
        left = panel * width
        for node, weight in zip(nodes, weights):
            u = left + (float(node) + 1.0) * width / 2.0
            total += float(weight) * first.quantile(u) * second.quantile(u)
    return total * width / 2.0


def generate_demand(profile: DemandProfile, periods: int) -> np.ndarray:
    """Generate the demand of periods ``1..periods`` from a profile.

    Args:
        profile: Validated profile.
        periods: Number of periods, >= 0.

    Returns:
        ``float64`` array of length ``periods`` in MWh, rounded to 0.001,
        every value inside its segment's ``[min, max]`` (a transition hour
        inside the weighted bounds of both segments) and > 0. Identical
        for identical inputs; a prefix of a longer series equals the shorter
        series.

    Raises:
        ValueError: If ``periods`` is negative.
    """
    if periods < 0:
        raise ValueError(f"periods must be non-negative, got {periods}")
    rng = np.random.default_rng(profile.seed)
    phi = float(profile.autocorrelation)
    scale = math.sqrt(1.0 - phi * phi)
    weights = profile.weights(periods)
    segments = profile.segments()
    values = np.empty(periods, dtype=np.float64)
    z = 0.0
    for index in range(periods):
        # One innovation per period, drawn in period order: the series of
        # m periods is a prefix of the series of n >= m periods.
        eps = float(rng.standard_normal())
        z = eps if index == 0 else phi * z + scale * eps
        u = min(max(_STANDARD_NORMAL.cdf(z), U_EPSILON), 1.0 - U_EPSILON)
        row = weights[index]
        active = [i for i in range(len(segments)) if row[i] > 0.0]
        if len(active) == 1:
            segment = segments[active[0]]
            demand = round(segment.quantile(u), DEMAND_DECIMALS)
            values[index] = min(max(demand, segment.min), segment.max)
        else:
            # Transition period: comonotone mix of the quantiles at the same u.
            mixed = sum(float(row[i]) * segments[i].quantile(u) for i in active)
            low = sum(float(row[i]) * segments[i].min for i in active)
            high = sum(float(row[i]) * segments[i].max for i in active)
            values[index] = min(max(round(mixed, DEMAND_DECIMALS), low), high)
    return values
