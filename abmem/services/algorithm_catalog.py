"""Data and actions of the Algorithms page (PLAN 2.12, spec algorithms_tab).

The page lists the registered agent algorithms and market clearing
algorithms and edits their global default parameters:

* agent algorithms: ``AlgorithmDefault`` (one row per ``algorithm_type``),
* clearing algorithms: ``ClearingDefault`` (one row per ``algorithm_key``).

A saved default stores only the fields whose value differs from the class
default of the config class, so a later change of a class default still
applies to the other fields. Deleting the row restores the published
defaults. Runs read the defaults once when they start
(``SimulationContext.build``) and keep a snapshot of what they used.

A stored agent default records its origin (``AlgorithmDefault.origin``).
Values applied from a hyperparameter search are ``OPTIMIZED`` with the
source study. A save on this page makes them ``CUSTOM`` again unless the
submitted values equal the stored ones. Clearing defaults are always
custom.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from django.db import transaction
from django.db.models import Count
from django.urls import reverse

from abmem.models import (
    Agent,
    AlgorithmDefault,
    AlgorithmDefaultOrigin,
    ClearingDefault,
    Market,
)
from abmem.services.algorithms import registry as algorithm_registry
from abmem.services.market.clearing_algorithms import registry as clearing_registry
from abmem.services.param_schema import (
    NON_FIELD_ERRORS,
    ParamValidation,
    describe_config,
    non_default_params,
    validate_params,
)

__all__ = [
    "AGENT",
    "CLEARING",
    "AlgorithmCard",
    "AlgorithmKind",
    "ModuleError",
    "agent_cards",
    "clearing_cards",
    "clearing_param_fields",
    "config_class_for",
    "defaults_origin",
    "param_form",
    "reset_defaults",
    "save_defaults",
    "stored_defaults",
    "validate_scenario_clearing_params",
]

AlgorithmKind = Literal["agent", "clearing"]
AGENT: Final[AlgorithmKind] = "agent"
CLEARING: Final[AlgorithmKind] = "clearing"


@dataclass(frozen=True)
class ModuleError:
    """A module of an algorithm package that failed to import.

    Attributes:
        module: Dotted module name.
        summary: Last line of the traceback.
        traceback: Full traceback text.
    """

    module: str
    summary: str
    traceback: str


@dataclass(frozen=True)
class AlgorithmCard:
    """One card of the Algorithms page.

    Attributes:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key.
        label: Display name.
        description: One-sentence description.
        is_learning: Learning algorithm (agent algorithms only).
        is_on_policy: On-policy learner (agent algorithms only).
        library_agents: Library agents using the algorithm (agent algorithms).
        scenarios: Scenarios (proxy simulations) using the algorithm.
        param_count: Number of parameters of the config class.
        has_custom_defaults: A stored default row exists.
        config_error: Message when the config class cannot be described
            (unsupported field type or invalid class defaults), else empty.
        origin: Origin of the stored default (see :func:`defaults_origin`),
            None when no row exists.
    """

    kind: AlgorithmKind
    key: str
    label: str
    description: str
    is_learning: bool = False
    is_on_policy: bool = False
    library_agents: int = 0
    scenarios: int = 0
    param_count: int = 0
    has_custom_defaults: bool = False
    config_error: str = ""
    origin: dict[str, Any] | None = None


def defaults_origin(row: Any) -> dict[str, Any] | None:
    """Describe where a stored default row comes from, for the page.

    Args:
        row: An ``AlgorithmDefault`` or ``ClearingDefault`` row, or None.
            A row without an ``origin`` field (clearing defaults) is custom.

    Returns:
        None when ``row`` is None, else ``{"origin", "label", "title",
        "study_id", "study_name", "watch_url"}``. ``origin`` is
        ``"CUSTOM"``, ``"OPTIMIZED"`` or ``"OPTIMIZED_DELETED"`` (the
        source study no longer exists). ``watch_url`` is the watch page of
        the source study, empty when there is none.
    """
    if row is None:
        return None
    origin = getattr(row, "origin", AlgorithmDefaultOrigin.CUSTOM)
    if origin != AlgorithmDefaultOrigin.OPTIMIZED:
        return {
            "origin": str(AlgorithmDefaultOrigin.CUSTOM.value),
            "label": str(AlgorithmDefaultOrigin.CUSTOM.label),
            "title": "Saved by hand on the Algorithms page.",
            "study_id": None,
            "study_name": "",
            "watch_url": "",
        }
    study = row.source_study
    if study is None:
        return {
            "origin": "OPTIMIZED_DELETED",
            "label": "Optimized (search deleted)",
            "title": "Applied from a hyperparameter search that was deleted.",
            "study_id": None,
            "study_name": "",
            "watch_url": "",
        }
    return {
        "origin": str(AlgorithmDefaultOrigin.OPTIMIZED.value),
        "label": str(AlgorithmDefaultOrigin.OPTIMIZED.label),
        "title": f"From hyperparameter search '{study.name}' (#{study.pk})",
        "study_id": study.pk,
        "study_name": study.name,
        "watch_url": reverse("hpo_watch", args=[study.pk]),
    }


def _module_errors(errors: Mapping[str, str]) -> list[ModuleError]:
    """Convert the import errors of a registry scan to page items."""
    result = []
    for module, text in sorted(errors.items()):
        lines = [line for line in text.strip().splitlines() if line.strip()]
        result.append(
            ModuleError(
                module=module, summary=lines[-1] if lines else "", traceback=text
            )
        )
    return result


def _param_count(config_class: type) -> tuple[int, str]:
    """Return the number of fields of a config class and a description error."""
    try:
        return len(describe_config(config_class)), ""
    except (TypeError, ValueError) as error:
        return 0, str(error)


def agent_cards() -> tuple[list[AlgorithmCard], list[ModuleError]]:
    """Return the cards of all agent algorithms and the failed modules.

    Returns:
        Cards (learning algorithms first, then baselines, by label) and the
        import errors of the algorithm package.
    """
    classes, errors = algorithm_registry.AGENT_ALGORITHMS.scan()
    library = dict(
        Agent.objects.filter(market__isnull=True, proxy=True)
        .values_list("algorithm_type")
        .annotate(n=Count("id"))
    )
    scenarios = dict(
        Agent.objects.filter(market__simulation__proxy=True)
        .values_list("algorithm_type")
        .annotate(n=Count("market__simulation", distinct=True))
    )
    stored = {
        row.algorithm_type: row
        for row in AlgorithmDefault.objects.select_related("source_study")
    }
    cards = []
    for info in algorithm_registry.sorted_infos(classes):
        count, error = _param_count(classes[info.key].config_class)
        cards.append(
            AlgorithmCard(
                kind=AGENT,
                key=info.key,
                label=info.label,
                description=info.description,
                is_learning=info.is_learning,
                is_on_policy=info.is_on_policy,
                library_agents=int(library.get(info.key, 0)),
                scenarios=int(scenarios.get(info.key, 0)),
                param_count=count,
                has_custom_defaults=info.key in stored,
                config_error=error,
                origin=defaults_origin(stored.get(info.key)),
            )
        )
    return cards, _module_errors(errors)


def clearing_cards() -> tuple[list[AlgorithmCard], list[ModuleError]]:
    """Return the cards of all clearing algorithms and the failed modules.

    Returns:
        Cards (the default algorithm first, then by label) and the import
        errors of the clearing algorithm package.
    """
    classes, errors = clearing_registry.CLEARING_ALGORITHMS.scan()
    scenarios = dict(
        Market.objects.filter(simulation__proxy=True)
        .values_list("clearing_algorithm")
        .annotate(n=Count("id"))
    )
    stored = {row.algorithm_key: row for row in ClearingDefault.objects.all()}
    cards = []
    for info in clearing_registry.sorted_clearing_infos(classes):
        count, error = _param_count(classes[info.key].config_class)
        cards.append(
            AlgorithmCard(
                kind=CLEARING,
                key=info.key,
                label=info.label,
                description=info.description,
                scenarios=int(scenarios.get(info.key, 0)),
                param_count=count,
                has_custom_defaults=info.key in stored,
                config_error=error,
                origin=defaults_origin(stored.get(info.key)),
            )
        )
    return cards, _module_errors(errors)


def config_class_for(kind: AlgorithmKind, key: str) -> type:
    """Return the config class of an algorithm.

    Args:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key.

    Raises:
        KeyError: Unknown kind or key.
    """
    if kind == AGENT:
        return algorithm_registry.get_config_class(key)
    if kind == CLEARING:
        return clearing_registry.get_clearing_config_class(key)
    raise KeyError(f"unknown algorithm kind {kind!r}")


def _label(kind: AlgorithmKind, key: str) -> str:
    """Return the display label of an algorithm."""
    if kind == AGENT:
        return algorithm_registry.label_for(key)
    return clearing_registry.clearing_label_for(key)


def _row_manager(kind: AlgorithmKind) -> tuple[Any, str]:
    """Return the default model and its key field for ``kind``."""
    if kind == AGENT:
        return AlgorithmDefault, "algorithm_type"
    return ClearingDefault, "algorithm_key"


def stored_defaults(kind: AlgorithmKind, key: str) -> dict[str, Any] | None:
    """Return the stored default parameters of an algorithm.

    Args:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key.

    Returns:
        A copy of the stored params, None when no row exists.
    """
    row = _stored_row(kind, key)
    return None if row is None else dict(row.params or {})


def _stored_row(kind: AlgorithmKind, key: str) -> Any:
    """Return the stored default row of an algorithm, None when absent."""
    model, key_field = _row_manager(kind)
    query = model.objects.filter(**{key_field: key})
    if kind == AGENT:
        query = query.select_related("source_study")
    return query.first()


def param_form(kind: AlgorithmKind, key: str) -> dict[str, Any]:
    """Return the parameter form of an algorithm as JSON data.

    Args:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key.

    Returns:
        ``{"kind", "key", "label", "has_custom_defaults", "origin", "stored",
        "stored_error", "fields"}``. ``origin`` is :func:`defaults_origin`
        of the stored row (None without a row). Every field carries ``name``, ``kind``,
        ``default`` (class default), ``description``, ``search_bounds``,
        ``value`` (current global default) and ``differs`` (value differs from
        the class default). When the stored row does not validate any more,
        ``stored_error`` holds the message and the values are the class
        defaults.

    Raises:
        KeyError: Unknown kind or key.
    """
    config_class = config_class_for(kind, key)
    fields = describe_config(config_class)
    row = _stored_row(kind, key)
    stored = None if row is None else dict(row.params or {})
    values = config_class().to_dict()
    stored_error = ""
    if stored:
        validation = validate_params(config_class, stored)
        if validation.is_valid:
            values = validation.config.to_dict()
        else:
            stored_error = " ".join(
                message
                for messages in validation.errors.values()
                for message in messages
            )
    return {
        "kind": kind,
        "key": key,
        "label": _label(kind, key),
        "has_custom_defaults": stored is not None,
        "origin": defaults_origin(row),
        "stored": stored or {},
        "stored_error": stored_error,
        "fields": [
            {
                **param.to_dict(),
                "value": values[param.name],
                "differs": values[param.name] != param.default,
            }
            for param in fields
        ],
    }


def save_defaults(
    kind: AlgorithmKind, key: str, raw: Mapping[str, Any] | None
) -> ParamValidation:
    """Validate submitted values and store them as the global default.

    Only fields that differ from the class default are stored. When none
    differs, the row is deleted (same effect as a reset). A save that
    changes the effective values of an agent default marks the row
    ``CUSTOM`` and clears its source study. Submitting the values that are
    already stored writes nothing, so an ``OPTIMIZED`` row keeps its origin.

    Args:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key.
        raw: Submitted values by field name.

    Returns:
        The validation. Nothing is written when it is invalid.

    Raises:
        KeyError: Unknown kind or key.
    """
    config_class = config_class_for(kind, key)
    validation = validate_params(config_class, raw)
    if not validation.is_valid:
        return validation
    params = non_default_params(validation.config)
    model, key_field = _row_manager(kind)
    with transaction.atomic():
        if not params:
            model.objects.filter(**{key_field: key}).delete()
            return validation
        row = model.objects.select_for_update().filter(**{key_field: key}).first()
        if row is None:
            model.objects.create(**{key_field: key, "params": params})
        elif not _same_values(config_class, row.params, validation.config):
            row.params = params
            update_fields = ["params", "updated_at"]
            if kind == AGENT:
                row.origin = AlgorithmDefaultOrigin.CUSTOM
                row.source_study = None
                update_fields += ["origin", "source_study"]
            row.save(update_fields=update_fields)
    return validation


def _same_values(config_class: type, stored: Any, config: Any) -> bool:
    """Return whether the stored params give the same config as ``config``.

    Args:
        config_class: Config dataclass of the algorithm.
        stored: Params of the stored row.
        config: Validated config of the submitted values.

    Returns:
        False when the stored params do not validate any more.
    """
    current = validate_params(config_class, stored or {})
    return current.is_valid and current.config.to_dict() == config.to_dict()


def reset_defaults(kind: AlgorithmKind, key: str) -> bool:
    """Delete the stored default of an algorithm (published defaults apply).

    Args:
        kind: ``"agent"`` or ``"clearing"``.
        key: Registry key. Unknown keys are allowed, so a stale row of a
            removed algorithm can be deleted.

    Returns:
        True if a row was deleted.
    """
    model, key_field = _row_manager(kind)
    deleted, _ = model.objects.filter(**{key_field: key}).delete()
    return bool(deleted)


def clearing_param_fields(key: str) -> list[dict[str, Any]]:
    """Return the wizard fields of a clearing algorithm.

    Args:
        key: Registered clearing algorithm key.

    Returns:
        Field dicts of :func:`param_form` whose ``value`` is the global
        default the scenario inherits.

    Raises:
        KeyError: Unknown key.
    """
    return param_form(CLEARING, key)["fields"]


def validate_scenario_clearing_params(
    key: str,
    raw: Any,
    *,
    defaults: Callable[[str], Mapping[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the per-scenario overrides of a clearing algorithm.

    The overrides are validated together with the global default
    (``ClearingDefault``) of the algorithm, as a run would combine them.

    Args:
        key: Registered clearing algorithm key.
        raw: Submitted overrides (mapping of field name to value), None or
            an empty mapping for none.
        defaults: Lookup of the stored default, :func:`stored_defaults` of
            the clearing kind when None.

    Returns:
        ``(overrides, errors)``: the typed overrides to store and the error
        messages, each prefixed with the field name.
    """
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        return {}, ["Clearing parameters must be an object of field values."]
    lookup = defaults or (lambda k: stored_defaults(CLEARING, k))
    validation = validate_params(
        clearing_registry.get_clearing_config_class(key), raw, base=lookup(key) or {}
    )
    if validation.is_valid:
        return validation.parsed, []
    messages = []
    for name, field_messages in validation.errors.items():
        prefix = "" if name == NON_FIELD_ERRORS else f"{name}: "
        messages.extend(prefix + message for message in field_messages)
    return {}, messages
