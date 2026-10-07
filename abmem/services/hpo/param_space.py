"""Search space of a hyperparameter study.

The optimizer searches in the unit cube ``[0, 1]^D``. Each dimension is mapped
to a hyperparameter value by a :class:`ParamSpec`:

* ``linear``: ``value = low + u * (high - low)``.
* ``log``: linear interpolation of ``log10(value)`` between ``log10(low)`` and
  ``log10(high)`` (requires ``low > 0``).
* ``int``: the integers ``low, ..., high`` (both inclusive, integral bounds)
  each receive an interval of equal width ``1 / (high - low + 1)`` in the unit
  interval, ``value = low + floor(u * (high - low + 1))`` with ``u = 1`` mapped
  to ``high``. Every point of an interval decodes to the same integer, so the
  rounding is consistent and ``from_unit(to_unit(k)) == k``.

Each study builds its own :class:`ParamSpace` from ``HPOStudy.param_bounds``.
There is no module-level mutable dictionary of bounds (defect B10).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Scale = Literal["linear", "log", "int"]
_SCALES: tuple[str, ...] = ("linear", "log", "int")


def _clip_unit(u: float) -> float:
    """Clips a value to the closed unit interval.

    Args:
        u: Value to clip.

    Returns:
        ``u`` clipped to ``[0, 1]``.

    Raises:
        ValueError: If ``u`` is NaN or infinite.
    """
    if not math.isfinite(u):
        raise ValueError(f"unit-cube coordinate must be finite, got {u!r}")
    return min(1.0, max(0.0, float(u)))


@dataclass(frozen=True)
class ParamSpec:
    """One searched hyperparameter.

    Attributes:
        name: Parameter name (a field of the algorithm configuration).
        low: Lower bound (inclusive).
        high: Upper bound (inclusive), strictly greater than ``low``.
        scale: ``'linear'``, ``'log'`` or ``'int'``.
    """

    name: str
    low: float
    high: float
    scale: Scale = "linear"

    def __post_init__(self) -> None:
        """Validates the bounds and the scale.

        Raises:
            ValueError: If the name is empty, the scale is unknown, a bound is
                not finite, ``low >= high``, a log bound is not positive or an
                int bound is not integral.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be a non-empty string")
        if self.scale not in _SCALES:
            raise ValueError(
                f"{self.name}: unknown scale {self.scale!r}, expected one of {_SCALES}"
            )
        try:
            low = float(self.low)
            high = float(self.high)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.name}: bounds must be numbers") from exc
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"{self.name}: bounds must be finite, got [{self.low}, {self.high}]"
            )
        if not low < high:
            raise ValueError(
                f"{self.name}: low must be < high, got [{self.low}, {self.high}]"
            )
        if self.scale == "log" and low <= 0.0:
            raise ValueError(f"{self.name}: log scale requires low > 0, got {self.low}")
        if self.scale == "int" and not (low.is_integer() and high.is_integer()):
            raise ValueError(
                f"{self.name}: int scale requires integral bounds, got [{self.low}, {self.high}]"
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def to_unit(self, value: float) -> float:
        """Maps a parameter value to the unit interval.

        For ``int`` the centre of the integer's interval is returned.

        Args:
            value: Parameter value within ``[low, high]``.

        Returns:
            Coordinate in ``[0, 1]``.

        Raises:
            ValueError: If ``value`` is not finite or lies outside the bounds,
                or is not integral for the ``int`` scale.
        """
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{self.name}: value must be finite, got {value!r}")
        if v < self.low or v > self.high:
            raise ValueError(
                f"{self.name}: value {value} outside [{self.low}, {self.high}]"
            )
        if self.scale == "linear":
            return _clip_unit((v - self.low) / (self.high - self.low))
        if self.scale == "log":
            lo, hi = math.log10(self.low), math.log10(self.high)
            return _clip_unit((math.log10(v) - lo) / (hi - lo))
        if not v.is_integer():
            raise ValueError(
                f"{self.name}: int parameter requires an integral value, got {value}"
            )
        n_values = self.high - self.low + 1.0
        return _clip_unit((v - self.low + 0.5) / n_values)

    def from_unit(self, u: float) -> float | int:
        """Maps a unit-interval coordinate to a parameter value.

        Args:
            u: Coordinate. Values outside ``[0, 1]`` are clipped.

        Returns:
            ``float`` for ``linear`` and ``log``, ``int`` for ``int``. The
            result always lies within ``[low, high]``.

        Raises:
            ValueError: If ``u`` is NaN or infinite.
        """
        u = _clip_unit(u)
        if self.scale == "linear":
            value = self.low + u * (self.high - self.low)
            return float(min(self.high, max(self.low, value)))
        if self.scale == "log":
            lo, hi = math.log10(self.low), math.log10(self.high)
            value = 10.0 ** (lo + u * (hi - lo))
            return float(min(self.high, max(self.low, value)))
        low_i, high_i = int(self.low), int(self.high)
        offset = math.floor(u * (high_i - low_i + 1))
        return int(min(high_i, low_i + offset))


@dataclass(frozen=True)
class ParamSpace:
    """Ordered collection of :class:`ParamSpec`, one per unit-cube dimension.

    Attributes:
        specs: Parameter specifications. Names are unique.
    """

    specs: tuple[ParamSpec, ...]

    def __post_init__(self) -> None:
        """Validates the collection.

        Raises:
            ValueError: If there are no specs or the names are not unique.
            TypeError: If an element is not a :class:`ParamSpec`.
        """
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("parameter space must contain at least one parameter")
        for spec in specs:
            if not isinstance(spec, ParamSpec):
                raise TypeError(f"expected ParamSpec, got {type(spec).__name__}")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"parameter names must be unique, got {names}")
        object.__setattr__(self, "specs", specs)

    @property
    def dim(self) -> int:
        """Number of dimensions of the unit cube."""
        return len(self.specs)

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names in dimension order."""
        return tuple(spec.name for spec in self.specs)

    def decode(self, x: np.ndarray) -> dict[str, float | int]:
        """Decodes a unit-cube point into parameter values.

        Args:
            x: Array of shape ``(dim,)`` with coordinates in ``[0, 1]``
                (values outside are clipped).

        Returns:
            Mapping from parameter name to value, in dimension order.

        Raises:
            ValueError: If the shape does not match or a coordinate is not
                finite.
        """
        arr = np.asarray(x, dtype=float)
        if arr.shape != (self.dim,):
            raise ValueError(f"expected shape ({self.dim},), got {arr.shape}")
        return {spec.name: spec.from_unit(float(u)) for spec, u in zip(self.specs, arr)}

    def encode(self, params: Mapping[str, float]) -> np.ndarray:
        """Encodes parameter values into a unit-cube point.

        Args:
            params: Value for every parameter of the space.

        Returns:
            Array of shape ``(dim,)``.

        Raises:
            KeyError: If a parameter is missing.
            ValueError: If a value is invalid for its spec.
        """
        return np.array(
            [spec.to_unit(params[spec.name]) for spec in self.specs], dtype=float
        )

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draws a point uniformly from the unit cube.

        Args:
            rng: Random generator, the only source of randomness.

        Returns:
            Array of shape ``(dim,)`` with entries in ``[0, 1)``.
        """
        return rng.random(self.dim)

    @classmethod
    def from_bounds(cls, bounds: Mapping[str, Mapping[str, Any]]) -> ParamSpace:
        """Builds a space from the JSON shape stored in ``HPOStudy.param_bounds``.

        Args:
            bounds: ``{name: {"low": ..., "high": ..., "scale": ...}}``.
                ``scale`` defaults to ``'linear'``. For convenience a value may
                also be a ``(low, high, scale)`` sequence, the shape returned by
                ``AlgorithmConfig.default_search_bounds``. Dimension order
                follows the mapping's iteration order.

        Returns:
            A new :class:`ParamSpace`.

        Raises:
            ValueError: If ``bounds`` is empty or an entry is malformed or
                invalid.
        """
        if not isinstance(bounds, Mapping) or not bounds:
            raise ValueError("bounds must be a non-empty mapping")
        specs: list[ParamSpec] = []
        for name, entry in bounds.items():
            if isinstance(entry, Mapping):
                unknown = set(entry) - {"low", "high", "scale"}
                if unknown:
                    raise ValueError(f"{name}: unknown bound keys {sorted(unknown)}")
                if "low" not in entry or "high" not in entry:
                    raise ValueError(f"{name}: bounds need 'low' and 'high'")
                low, high, scale = (
                    entry["low"],
                    entry["high"],
                    entry.get("scale", "linear"),
                )
            elif (
                isinstance(entry, Sequence)
                and not isinstance(entry, str)
                and len(entry) in (2, 3)
            ):
                low, high = entry[0], entry[1]
                scale = entry[2] if len(entry) == 3 else "linear"
            else:
                raise ValueError(f"{name}: malformed bound entry {entry!r}")
            if isinstance(low, bool) or isinstance(high, bool):
                raise ValueError(f"{name}: bounds must be numbers")
            specs.append(ParamSpec(name=str(name), low=low, high=high, scale=scale))
        return cls(specs=tuple(specs))
