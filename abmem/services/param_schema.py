"""Describe, coerce and validate the parameter dataclasses of algorithms (PLAN 2.12).

Agent algorithms declare a frozen ``AlgorithmConfig`` subclass and clearing
algorithms a frozen ``ClearingConfig`` subclass. Both expose ``from_dict``,
``to_dict`` and ``field_names``. This module turns such a class into a list
of form fields (type, class default, docstring description and search bounds)
and validates submitted values with per-field error messages, so the
Algorithms page and the scenario wizard generate their forms from the code.

The module imports neither Django models nor torch.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import numbers
import re
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

__all__ = [
    "NON_FIELD_ERRORS",
    "FieldKind",
    "ParamField",
    "ParamValidation",
    "attribute_descriptions",
    "coerce_field_value",
    "describe_config",
    "field_kind",
    "non_default_params",
    "parse_form_value",
    "validate_params",
]

FieldKind = Literal["int", "float", "optional_float", "bool", "int_list"]

NON_FIELD_ERRORS: Final[str] = "__all__"
"""Error key of messages that do not belong to one field."""

_SECTION_HEADER = re.compile(r"^(\s*)([A-Z][A-Za-z ]*):\s*$")
_ATTRIBUTE_LINE = re.compile(
    r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?:\s*(.*)$"
)


def _is_optional_float(hint: Any) -> bool:
    """Return True if ``hint`` is ``float | None``."""
    return typing.get_origin(hint) in (typing.Union, types.UnionType) and set(
        typing.get_args(hint)
    ) == {float, type(None)}


def field_kind(hint: Any) -> FieldKind:
    """Return the form kind of a dataclass field type hint.

    Args:
        hint: Resolved type hint of the field.

    Returns:
        ``"int"``, ``"float"``, ``"optional_float"``, ``"bool"`` or
        ``"int_list"`` (``tuple[int, ...]``).

    Raises:
        TypeError: If the hint is not one of the supported types.
    """
    if hint is bool:
        return "bool"
    if hint is int:
        return "int"
    if hint is float:
        return "float"
    if _is_optional_float(hint):
        return "optional_float"
    if typing.get_origin(hint) is tuple and typing.get_args(hint) in (
        (int, ...),
        (int, Ellipsis),
    ):
        return "int_list"
    raise TypeError(f"unsupported parameter type {hint!r}")


def _coerce_int(name: str, value: Any) -> int:
    """Return ``value`` as int, accepting integral floats.

    Raises:
        ValueError: For bools, non-numbers, non-finite or non-integral values.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    as_float = float(value)
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(as_float)


def _coerce_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite float.

    Raises:
        ValueError: For bools, non-numbers, NaN or infinite values.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return as_float


def coerce_field_value(name: str, hint: Any, value: Any) -> Any:
    """Coerce a typed (already decoded) value to the field type.

    Used by ``ClearingConfig.__post_init__``. Strings are not parsed here (see
    :func:`parse_form_value` for form input).

    Args:
        name: Field name, used in error messages.
        hint: Resolved type hint of the field.
        value: Raw value.

    Returns:
        The coerced value (tuples for ``int_list``).

    Raises:
        ValueError: If the value does not fit the type.
        TypeError: If the hint is not supported.
    """
    kind = field_kind(hint)
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return value
    if kind == "int":
        return _coerce_int(name, value)
    if kind == "float":
        return _coerce_float(name, value)
    if kind == "optional_float":
        return None if value is None else _coerce_float(name, value)
    if isinstance(value, (str, bytes)) or not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence of integers, got {value!r}")
    return tuple(_coerce_int(name, item) for item in value)


def attribute_descriptions(cls: type) -> dict[str, str]:
    """Collect the ``Attributes:`` entries of the docstrings of ``cls`` and its bases.

    Google-style entries ``name: text`` (continuation lines indented deeper
    are joined) are read from every class of the MRO. A subclass entry wins
    over the entry of a base class.

    Args:
        cls: A class with Google-style docstrings.

    Returns:
        Mapping of attribute name to its one-line description.
    """
    result: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        doc = inspect.cleandoc(klass.__dict__.get("__doc__") or "")
        result.update(_parse_attributes(doc))
    return result


def _parse_attributes(doc: str) -> dict[str, str]:
    """Parse the ``Attributes:`` section of one cleaned docstring."""
    entries: dict[str, str] = {}
    in_section = False
    section_indent = 0
    entry_indent: int | None = None
    current: str | None = None
    for line in doc.splitlines():
        if not line.strip():
            continue
        header = _SECTION_HEADER.match(line)
        indent = len(line) - len(line.lstrip())
        if header is not None and (not in_section or indent <= section_indent):
            in_section = header.group(2) == "Attributes"
            section_indent = indent
            entry_indent = None
            current = None
            continue
        if not in_section:
            continue
        if indent <= section_indent:
            in_section = False
            continue
        match = _ATTRIBUTE_LINE.match(line)
        if match is not None and (entry_indent is None or indent == entry_indent):
            entry_indent = indent
            current = match.group(2)
            entries[current] = match.group(3).strip()
        elif current is not None:
            entries[current] = f"{entries[current]} {line.strip()}".strip()
    return entries


@dataclass(frozen=True)
class ParamField:
    """One editable parameter of a config class.

    Attributes:
        name: Field name.
        kind: Form kind (see :func:`field_kind`).
        default: Class default as a JSON value (lists for ``int_list``).
        description: Text of the docstring ``Attributes:`` entry.
        search_bounds: ``(low, high, scale)`` from ``default_search_bounds``,
            None when the field is not searched.
    """

    name: str
    kind: FieldKind
    default: Any
    description: str = ""
    search_bounds: tuple[float, float, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return {
            "name": self.name,
            "kind": self.kind,
            "default": self.default,
            "description": self.description,
            "search_bounds": (
                list(self.search_bounds) if self.search_bounds is not None else None
            ),
        }


def describe_config(config_class: type) -> tuple[ParamField, ...]:
    """Return the form fields of a config dataclass in declaration order.

    Args:
        config_class: A dataclass with ``to_dict`` whose defaults validate
            (``config_class()`` must not raise). An optional classmethod
            ``default_search_bounds`` supplies search hints.

    Returns:
        One :class:`ParamField` per dataclass field.

    Raises:
        TypeError: If a field type is not supported.
    """
    hints = typing.get_type_hints(config_class)
    defaults = config_class().to_dict()
    descriptions = attribute_descriptions(config_class)
    bounds_method = getattr(config_class, "default_search_bounds", None)
    bounds: Mapping[str, Any] = bounds_method() if callable(bounds_method) else {}
    result = []
    for f in dataclasses.fields(config_class):
        bound = bounds.get(f.name)
        result.append(
            ParamField(
                name=f.name,
                kind=field_kind(hints[f.name]),
                default=defaults[f.name],
                description=descriptions.get(f.name, ""),
                search_bounds=(
                    (float(bound[0]), float(bound[1]), str(bound[2]))
                    if bound is not None
                    else None
                ),
            )
        )
    return tuple(result)


def _parse_number(name: str, raw: Any, integer: bool) -> int | float:
    """Parse a number from JSON or form text.

    Raises:
        ValueError: If ``raw`` is not a number of the required kind.
    """
    value: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError(f"{name} is required.")
        try:
            value = (
                float(text)
                if not integer or not text.lstrip("+-").isdigit()
                else int(text)
            )
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}.") from None
    if integer:
        return _coerce_int(name, value)
    return _coerce_float(name, value)


def parse_form_value(param: ParamField, raw: Any) -> Any:
    """Convert a submitted value (JSON value or form text) to the field type.

    Args:
        param: The field.
        raw: Submitted value. Numbers may be given as text, an ``int_list``
            as a list or as comma-separated text, an empty text or None
            means None for an ``optional_float``.

    Returns:
        A JSON value of the field type (a list for ``int_list``).

    Raises:
        ValueError: With a message naming the field.
    """
    name = param.name
    if param.kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("true", "false"):
            return raw.strip().lower() == "true"
        raise ValueError(f"{name} must be true or false, got {raw!r}.")
    if param.kind == "optional_float":
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        return _parse_number(name, raw, integer=False)
    if param.kind in ("int", "float"):
        if raw is None:
            raise ValueError(f"{name} is required.")
        return _parse_number(name, raw, integer=param.kind == "int")
    items: Any = raw
    if isinstance(raw, str):
        items = [part for part in (p.strip() for p in raw.split(",")) if part]
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError(f"{name} must be a non-empty list of integers, got {raw!r}.")
    return [_parse_number(name, item, integer=True) for item in items]


@dataclass(frozen=True)
class ParamValidation:
    """Outcome of :func:`validate_params`.

    Attributes:
        config: The validated config, None when there are errors.
        errors: Messages per field name, :data:`NON_FIELD_ERRORS` for the
            others. Empty when valid.
        parsed: The submitted values converted to their field types (JSON
            values), without the ``base`` values.
    """

    config: Any | None
    errors: dict[str, list[str]] = field(default_factory=dict)
    parsed: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True when the parameters are valid."""
        return self.config is not None and not self.errors


def _error_field(message: str, names: tuple[str, ...]) -> str:
    """Return the field whose name occurs first in ``message``.

    Args:
        message: Validation message of the config class.
        names: Field names.

    Returns:
        The field name, or :data:`NON_FIELD_ERRORS` when none occurs.
    """
    best: tuple[int, str] | None = None
    for name in names:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", message
        )
        if match is not None and (best is None or match.start() < best[0]):
            best = (match.start(), name)
    return best[1] if best is not None else NON_FIELD_ERRORS


def validate_params(
    config_class: type,
    raw: Mapping[str, Any] | None,
    base: Mapping[str, Any] | None = None,
) -> ParamValidation:
    """Parse and validate submitted parameters with per-field errors.

    Args:
        config_class: Config dataclass with ``from_dict``.
        raw: Submitted values by field name (JSON values or text).
        base: Values applied before ``raw`` (for example the global default
            under a per-scenario override), already typed. Names that are not
            fields are ignored, so a stale stored default does not block the
            form.

    Returns:
        The validation outcome. Unknown names are reported under
        :data:`NON_FIELD_ERRORS`. A message of the config class is assigned
        to the field it names first.
    """
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        return ParamValidation(
            None, {NON_FIELD_ERRORS: ["Parameters must be an object of field values."]}
        )
    fields = {param.name: param for param in describe_config(config_class)}
    errors: dict[str, list[str]] = {}
    unknown = sorted(str(key) for key in raw if key not in fields)
    if unknown:
        errors[NON_FIELD_ERRORS] = [
            f"Unknown parameter(s) {', '.join(unknown)}. Known parameters: "
            f"{', '.join(fields) or 'none'}."
        ]
    parsed: dict[str, Any] = {}
    for name, value in raw.items():
        if name not in fields:
            continue
        try:
            parsed[name] = parse_form_value(fields[name], value)
        except ValueError as error:
            errors.setdefault(name, []).append(str(error))
    if errors:
        return ParamValidation(None, errors, parsed)
    params = {name: value for name, value in dict(base or {}).items() if name in fields}
    params.update(parsed)
    try:
        config = config_class.from_dict(params)
    except (TypeError, ValueError) as error:
        message = str(error)
        return ParamValidation(
            None, {_error_field(message, tuple(fields)): [message]}, parsed
        )
    return ParamValidation(config, {}, parsed)


def non_default_params(config: Any) -> dict[str, Any]:
    """Return the fields of ``config`` whose value differs from the class default.

    Args:
        config: A config instance with ``to_dict``.

    Returns:
        ``{name: value}`` (JSON values) of the changed fields.
    """
    defaults = type(config)().to_dict()
    return {
        name: value
        for name, value in config.to_dict().items()
        if defaults.get(name) != value
    }
