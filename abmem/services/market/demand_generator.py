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
    "DemandProfile",
    "generate_demand",
    "night_mask",
    "normalize_profile",
]

PROFILE_VERSION: Final[int] = 1
SEED_UPPER_BOUND: Final[int] = 2**31
U_EPSILON: Final[float] = 1e-12
DEMAND_DECIMALS: Final[int] = 3
HOURS_PER_DAY: Final[int] = 24

_STANDARD_NORMAL: Final[NormalDist] = NormalDist()

PROFILE_KEYS: Final[tuple[str, ...]] = (
    "version",
    "seed",
    "day_start_hour",
    "day_end_hour",
    "autocorrelation",
    "day",
    "night",
)
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
        label = prefix.capitalize()

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
                add("mean", "the mean must lie strictly between the minimum and the maximum.")
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
    label = prefix.capitalize()

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
    needed = ("mean", "std", "min", "max") if _uses_moments(distribution) else ("min", "max")
    values: dict[str, float] = {}
    ok = True
    for field in needed:
        if field not in raw or raw.get(field) in (None, ""):
            add(field, f"{field} is required for the {distribution.value} distribution.")
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
class DemandProfile:
    """Synthetic demand profile of a scenario (schema version 1).

    Attributes:
        seed: Seed of the demand generator, in ``[0, 2**31)``; independent
            of the run seed.
        day_start_hour: First day hour on the 24-hour clock (0..23).
        day_end_hour: Exclusive end of the day hours (1..24).
        autocorrelation: AR(1) coefficient ``phi`` of the latent process, in
            ``[0, 1)``.
        day: Marginal of the day hours.
        night: Marginal of the night hours.
    """

    seed: int
    day_start_hour: int
    day_end_hour: int
    autocorrelation: float
    day: Segment
    night: Segment

    @property
    def day_hours(self) -> int:
        """Number of day hours per 24 hours."""
        return self.day_end_hour - self.day_start_hour

    @property
    def night_hours(self) -> int:
        """Number of night hours per 24 hours."""
        return HOURS_PER_DAY - self.day_hours

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON form stored in ``Simulation.demand_profile``.

        Returns:
            The version-1 profile mapping.
        """
        return {
            "version": PROFILE_VERSION,
            "seed": self.seed,
            "day_start_hour": self.day_start_hour,
            "day_end_hour": self.day_end_hour,
            "autocorrelation": self.autocorrelation,
            "day": self.day.to_dict(),
            "night": self.night.to_dict(),
        }

    def errors(self) -> dict[str, list[str]]:
        """Return the semantic errors of the profile.

        Returns:
            Field-keyed messages (``seed``, ``day_end_hour``, ``day.mean``,
            ...), empty when the profile is valid.
        """
        errors: dict[str, list[str]] = {}
        if not 0 <= self.seed < SEED_UPPER_BOUND:
            errors["seed"] = [
                f"Demand seed must be an integer from 0 to {SEED_UPPER_BOUND - 1}."
            ]
        if not 0 <= self.day_start_hour <= 23:
            errors["day_start_hour"] = ["Day start hour must be from 0 to 23."]
        if not 1 <= self.day_end_hour <= 24:
            errors["day_end_hour"] = ["Day end hour must be from 1 to 24."]
        elif "day_start_hour" not in errors and not 1 <= self.day_hours <= 23:
            errors["day_end_hour"] = [
                "Day end hour must be after the day start hour, and day and night "
                "must each have at least one hour (day length 1 to 23 h)."
            ]
        if not 0 <= self.autocorrelation < 1:
            errors["autocorrelation"] = [
                "Autocorrelation must be at least 0 and less than 1."
            ]
        errors.update(self.day.errors("day"))
        errors.update(self.night.errors("night"))
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
                or numeric strings.

        Returns:
            The validated profile.

        Raises:
            ValidationError: Field-keyed errors (``demand_profile`` for the
                whole object, ``seed``, ``day.std``, ...).
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
        ints: dict[str, int] = {}
        for field, label in (
            ("seed", "Demand seed"),
            ("day_start_hour", "Day start hour"),
            ("day_end_hour", "Day end hour"),
        ):
            value = _integer(data.get(field))
            if value is None:
                errors[field] = [f"{label} is required and must be an integer."]
            else:
                ints[field] = value
        phi = _number(data.get("autocorrelation"))
        if phi is None:
            errors["autocorrelation"] = [
                "Autocorrelation is required and must be a number."
            ]
        segments = {
            name: _parse_segment(data.get(name), name, errors) for name in SEGMENT_NAMES
        }
        if errors or phi is None or segments["day"] is None or segments["night"] is None:
            # Report the semantic errors of the parts that did parse as well.
            if segments["day"] is not None:
                errors.update(segments["day"].errors("day"))
            if segments["night"] is not None:
                errors.update(segments["night"].errors("night"))
            raise ValidationError(errors)
        profile = cls(
            seed=ints["seed"],
            day_start_hour=ints["day_start_hour"],
            day_end_hour=ints["day_end_hour"],
            autocorrelation=phi,
            day=segments["day"],
            night=segments["night"],
        )
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


def generate_demand(profile: DemandProfile, periods: int) -> np.ndarray:
    """Generate the demand of periods ``1..periods`` from a profile.

    Args:
        profile: Validated profile.
        periods: Number of periods, >= 0.

    Returns:
        ``float64`` array of length ``periods`` in MWh, rounded to 0.001,
        every value inside its segment's ``[min, max]`` and > 0. Identical
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
    night = night_mask(periods, profile.day_start_hour, profile.day_end_hour)
    values = np.empty(periods, dtype=np.float64)
    z = 0.0
    for index in range(periods):
        # One innovation per period, drawn in period order: the series of
        # m periods is a prefix of the series of n >= m periods.
        eps = float(rng.standard_normal())
        z = eps if index == 0 else phi * z + scale * eps
        u = min(max(_STANDARD_NORMAL.cdf(z), U_EPSILON), 1.0 - U_EPSILON)
        segment = profile.night if night[index] else profile.day
        demand = round(segment.quantile(u), DEMAND_DECIMALS)
        values[index] = min(max(demand, segment.min), segment.max)
    return values
