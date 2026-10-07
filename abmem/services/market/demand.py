"""Validated hourly demand series (PLAN 1.7, A8, OQ-12).

The demand column of ``marketData.xlsx`` holds 8808 valid rows followed by
empty rows. :class:`DemandSeries` keeps the column up to its last valid row,
checks before a run that the requested horizon is covered by finite, strictly
positive values and maps period numbers to rows with ``values[t - 1]``.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .demand_generator import DemandProfile

__all__ = ["DemandDataError", "DemandSeries"]


class DemandDataError(ValueError):
    """Raised when the demand data cannot serve the requested simulation horizon."""


@dataclass(frozen=True)
class DemandSeries:
    """Demand per period, indexed by period number starting at 1.

    Attributes:
        values: Demand values in MWh as a read-only ``float64`` array. Trailing
            empty rows of the source column are dropped.
        source: Human-readable origin of the data (file and column).
    """

    values: np.ndarray
    source: str

    def __post_init__(self) -> None:
        """Store a read-only one-dimensional ``float64`` copy of ``values``.

        Raises:
            ValueError: If ``values`` is not one-dimensional.
        """
        array = np.array(self.values, dtype=np.float64, copy=True)
        if array.ndim != 1:
            raise ValueError(
                f"demand values must be one-dimensional, got shape {array.shape}"
            )
        array.setflags(write=False)
        object.__setattr__(self, "values", array)

    @classmethod
    def from_excel(cls, path: pathlib.Path, column: str = "DEMAND") -> DemandSeries:
        """Read a demand column from an Excel file.

        Args:
            path: Path of the ``.xlsx`` file.
            column: Name of the demand column.

        Returns:
            The series holding the column up to its last non-empty row.

        Raises:
            DemandDataError: If the file cannot be read, the column is missing
                or the column contains non-numeric values.
        """
        path = pathlib.Path(path)
        source = f"{path} (column {column!r})"
        try:
            frame = pd.read_excel(path)
        except (OSError, ValueError) as error:
            raise DemandDataError(f"cannot read demand file {path}: {error}") from error
        if column not in frame.columns:
            raise DemandDataError(f"demand file {path} has no column {column!r}")
        try:
            series = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise DemandDataError(
                f"{source} contains non-numeric values: {error}"
            ) from error
        array = series.to_numpy(dtype=np.float64)
        valid = np.flatnonzero(~np.isnan(array))
        stop = int(valid[-1]) + 1 if valid.size else 0
        return cls(values=array[:stop], source=source)

    @classmethod
    def from_profile(cls, profile: DemandProfile, periods: int) -> DemandSeries:
        """Generate a synthetic demand series from a scenario profile (PLAN 2.8).

        Args:
            profile: Validated demand profile.
            periods: Number of periods to generate.

        Returns:
            The series of ``generate_demand(profile, periods)`` with source
            ``'generated'``.
        """
        from .demand_generator import generate_demand

        return cls(values=generate_demand(profile, periods), source="generated")

    def validate_horizon(self, period_count: int) -> None:
        """Check that periods ``1..period_count`` have usable demand values.

        Args:
            period_count: Number of periods the simulation will run.

        Raises:
            ValueError: If ``period_count`` is negative.
            DemandDataError: If ``period_count`` exceeds the number of rows or
                if any of the first ``period_count`` values is NaN, infinite,
                negative or zero.
        """
        if period_count < 0:
            raise ValueError(f"period_count must be non-negative, got {period_count}")
        if period_count > len(self.values):
            raise DemandDataError(
                f"{self.source}: requested horizon of {period_count} periods exceeds "
                f"the {len(self.values)} available demand rows "
                f"(first missing period {len(self.values) + 1})"
            )
        window = self.values[:period_count]
        bad = np.flatnonzero(~np.isfinite(window) | (window <= 0))
        if bad.size:
            row = int(bad[0])
            raise DemandDataError(
                f"{self.source}: requested horizon of {period_count} periods, invalid demand "
                f"{window[row]!r} at period {row + 1} (row index {row}). Demand must be "
                "finite and strictly positive."
            )

    def at(self, period_number: int) -> Decimal:
        """Return the demand of a period.

        Args:
            period_number: Period number ``t`` with ``1 <= t <= len(values)``.

        Returns:
            ``values[t - 1]`` as a ``Decimal`` (shortest round-trip decimal form).

        Raises:
            IndexError: If ``period_number`` is outside ``[1, len(values)]``.
            DemandDataError: If the stored value is NaN or infinite.
        """
        if not 1 <= period_number <= len(self.values):
            raise IndexError(
                f"{self.source}: period {period_number} is outside the demand series "
                f"(valid periods 1..{len(self.values)})"
            )
        value = float(self.values[period_number - 1])
        if not math.isfinite(value):
            raise DemandDataError(
                f"{self.source}: demand of period {period_number} is {value!r}"
            )
        return Decimal(repr(value))

    def window(self, start: int, stop: int) -> np.ndarray:
        """Return the demand of periods ``start <= t < stop``.

        Args:
            start: First period number, at least 1.
            stop: One past the last period number, at most ``len(values) + 1``.

        Returns:
            A read-only view ``values[start - 1 : stop - 1]``.

        Raises:
            IndexError: If the range is empty-inverted or outside the series.
        """
        if not 1 <= start <= stop <= len(self.values) + 1:
            raise IndexError(
                f"{self.source}: period window [{start}, {stop}) is outside the demand "
                f"series (valid periods 1..{len(self.values)})"
            )
        return self.values[start - 1 : stop - 1]
