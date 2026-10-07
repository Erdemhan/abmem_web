"""Parameter dataclass of market clearing algorithms (PLAN 2.12).

A clearing algorithm declares ``config_class``, a frozen subclass of
:class:`ClearingConfig` whose fields are its parameters. The base class has
no fields, so an algorithm without parameters needs no own config class.
Supported field types are ``int``, ``float``, ``float | None``, ``bool`` and
``tuple[int, ...]``. Document every field in an ``Attributes:`` section of
the class docstring, the Algorithms page shows that text next to the field.

Effective parameters of a run: class defaults, overridden by
``ClearingDefault.params`` of the algorithm, overridden by
``Market.clearing_params`` of the scenario.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from abmem.services.param_schema import coerce_field_value

__all__ = ["ClearingConfig"]


@dataclass(frozen=True)
class ClearingConfig:
    """Parameters of a clearing algorithm. The base class has no fields."""

    def __post_init__(self) -> None:
        """Coerce field types and validate value ranges.

        Raises:
            ValueError: If a field has a wrong type or an invalid value.
        """
        hints = typing.get_type_hints(type(self))
        for f in dataclasses.fields(self):
            value = coerce_field_value(f.name, hints[f.name], getattr(self, f.name))
            object.__setattr__(self, f.name, value)
        self._validate()

    def _validate(self) -> None:
        """Validate value ranges. Subclasses extend this and call super().

        Raises:
            ValueError: With a message that starts with the field name.
        """

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the names of all fields in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, params: Mapping[str, Any] | None) -> Self:
        """Build a config from a parameter mapping.

        Missing keys take the dataclass defaults, None yields the default
        config.

        Args:
            params: Mapping of field name to value, or None.

        Returns:
            A validated config instance.

        Raises:
            TypeError: If ``params`` is neither a mapping nor None.
            ValueError: If a key is not a field or a value is invalid.
        """
        if params is None:
            return cls()
        if not isinstance(params, Mapping):
            raise TypeError(
                f"params must be a mapping or None, got {type(params).__name__}"
            )
        accepted = cls.field_names()
        unknown = sorted(str(k) for k in params if k not in accepted)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {unknown} for {cls.__name__}, "
                f"accepted keys: {list(accepted)}"
            )
        return cls(**dict(params))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict that round-trips through from_dict.

        Returns:
            Mapping of field name to value. Tuples are returned as lists.
        """
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in ((n, getattr(self, n)) for n in self.field_names())
        }
