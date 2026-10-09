"""Scenario YAML export and import (PLAN 2.7).

File format, version 1 (exact keys)::

    format_version: 1
    scenario: {name, mode, period_type, periods, seed, evaluation_periods,
               observation_lags, demand}
    market: {strategy, clearing_algorithm, clearing_params: {}, bid_bounds: [low, high]}
    resources: [{name, energy_type, fuel_cost, om_cost, investment_cost,
                 co2_cost, emission}]
    agents: [{name, type, budget, algorithm, algorithm_params: {},
              failstack: {enabled, limit, delta, min_margin},
              plants: [{resource: <resource name>, capacity}]}]

``agents[].algorithm`` is the key of a registered agent algorithm (DDPG, TD3,
SAC, PPO, RDPG, RANDOM, MARGINAL_COST, FIXED_MARKUP or any algorithm class
discovered from the code, PLAN 2.10). ``market.clearing_algorithm`` is
optional and is the key of a registered clearing algorithm, ``MERIT_ORDER``
when missing. Export always writes it. ``market.clearing_params`` is optional
(PLAN 2.12): the per-scenario overrides of the clearing algorithm parameters,
validated with its config class. Export writes it when the scenario has
overrides.

``scenario.demand`` is optional (PLAN 2.8). Absent or null means the demand
file ``marketData.xlsx``; otherwise it is the synthetic demand profile as
stored in ``Simulation.demand_profile`` (``version`` may be left out)::

    demand: {version: 1, seed, day_start_hour, day_end_hour, autocorrelation,
             day: {distribution, mean, std, min, max}, night: {...}}

``distribution`` is NORMAL, UNIFORM or LOGNORMAL (UNIFORM has no mean and
std). Import validates it with ``DemandProfile`` (the wizard's rules);
export writes it whenever the scenario has a profile.

``scenario.demand_profile`` is optional and informational (PLAN 2.9): the
name of the library demand profile the scenario's copy came from. Export
writes it when the scenario is linked to a library profile. Import links the
new scenario to the library profile of that name only when its settings
equal ``scenario.demand``. An unknown name or different settings are
ignored, ``scenario.demand`` is always the profile that is stored.

Enum values are written as member names (``ONLYRESULT``, ``FOSSIL``, ...).
Decimal values are written as plain YAML numbers when a float holds them
exactly, else as quoted strings, and are always read back with
``Decimal(str(value))``, so no value changes on a round trip.

Import is two-phase. ``build_import_plan`` validates the whole document and
decides which resources and library agents are reused or created, without
writing. ``import_scenario`` validates again and writes everything in one
transaction. Header, run settings and agent field rules are the wizard's own
validators from ``abmem.views``, so a file is accepted exactly when the
equivalent wizard payload is accepted.

Reuse rules (approved 2026-10-07):

* A resource whose name matches an existing resource (case-insensitive, by
  the database ``LOWER``) is reused when energy type and every cost field are
  equal, otherwise the import fails and lists the differing fields.
* A library agent whose name matches an existing library agent (the
  comparison of the constraint ``uniq_library_agent_name_ci``) is reused when
  type, budget, algorithm, algorithm parameters, fail-stack fields and the
  multiset of ``(resource, capacity)`` plants are equal, otherwise the import
  fails and lists the conflicting agents. Other agents are created in the
  library (``proxy=True``, ``market=NULL``).
* The scenario is always created as a new proxy simulation whose agents are
  copies of the library agents (``library_service.copy_agent``) with the
  names of the file.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from collections.abc import Iterable
from typing import Any, Final, Mapping

import yaml
from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Value
from django.db.models.functions import Lower
from django.utils import timezone

from . import views as wizard
from .models import Agent, DemandProfileTemplate, Simulation
from .models.enums import (
    AgentType,
    EnergyType,
    MarketStrategy,
    PeriodType,
    SimulationMode,
)
from .models.models import Plant, Resource
from .services.agent import library_service
from .services.agent.library_service import AgentInput, PlantInput
from .services.agent.resource_rules import validate_agent_resources
from .services.algorithms import registry
from .services.market import demand_library, market_factory
from .services.market.clearing_algorithms import registry as clearing_registry
from .services.market.demand_library import DemandLibraryError
from .services.simulation import simulation_factory

FORMAT_VERSION: Final[int] = 1
MAX_FILE_BYTES: Final[int] = 1024 * 1024
CONTENT_TYPE: Final[str] = "application/x-yaml"

TOP_KEYS: Final[tuple[str, ...]] = (
    "format_version",
    "scenario",
    "market",
    "resources",
    "agents",
)
SCENARIO_KEYS: Final[tuple[str, ...]] = (
    "name",
    "mode",
    "period_type",
    "periods",
    "seed",
    "evaluation_periods",
    "observation_lags",
    "demand",
    "demand_profile",
)
SCENARIO_REQUIRED_KEYS: Final[tuple[str, ...]] = ("name", "mode", "period_type", "periods")
MARKET_KEYS: Final[tuple[str, ...]] = (
    "strategy",
    "clearing_algorithm",
    "clearing_params",
    "bid_bounds",
)
MARKET_REQUIRED_KEYS: Final[tuple[str, ...]] = ("strategy", "bid_bounds")
RESOURCES_KIND: Final[str] = "resources"

RESOURCE_COST_FIELDS: Final[tuple[tuple[str, str], ...]] = (
    ("fuel_cost", "fuelCost"),
    ("om_cost", "omCost"),
    ("investment_cost", "investmentCost"),
    ("co2_cost", "co2Cost"),
    ("emission", "emission"),
)
RESOURCE_KEYS: Final[tuple[str, ...]] = ("name", "energy_type") + tuple(
    key for key, _ in RESOURCE_COST_FIELDS
)
AGENT_KEYS: Final[tuple[str, ...]] = (
    "name",
    "type",
    "budget",
    "algorithm",
    "algorithm_params",
    "failstack",
    "plants",
    "library_agent",
)
AGENT_REQUIRED_KEYS: Final[tuple[str, ...]] = ("name", "type", "budget", "algorithm", "plants")
FAILSTACK_KEYS: Final[tuple[str, ...]] = ("enabled", "limit", "delta", "min_margin")
PLANT_KEYS: Final[tuple[str, ...]] = ("resource", "capacity")

RESOURCE_NAME_MAX_LENGTH: Final[int] = Resource._meta.get_field("name").max_length or 20

# Wizard payload key -> key of the YAML document, for re-keying wizard errors.
_WIZARD_KEYS: Final[Mapping[str, str]] = {
    "simulation_name": "scenario.name",
    "simulationMode": "scenario.mode",
    "periodType": "scenario.period_type",
    "periodNumber": "scenario.periods",
    "seed": "scenario.seed",
    "evaluation_periods": "scenario.evaluation_periods",
    "observation_lags": "scenario.observation_lags",
    "strategy": "market.strategy",
    "clearing_algorithm": "market.clearing_algorithm",
    "clearing_params": "market.clearing_params",
    "lowerBound": "market.bid_bounds",
    "upperBound": "market.bid_bounds",
}

# Fail-stack model field -> YAML key.
_FAILSTACK_YAML_KEYS: Final[Mapping[str, str]] = {
    "failstack_enabled": "failstack.enabled",
    "failstack_limit": "failstack.limit",
    "failstack_delta": "failstack.delta",
    "failstack_min_margin": "failstack.min_margin",
    "algorithm_type": "algorithm",
    "failstack": "failstack",
}


class ScenarioImportError(Exception):
    """The uploaded document cannot be imported. Nothing was written.

    Attributes:
        errors: Mapping from document path (``scenario.seed``,
            ``agents[1].plants``, ``file``, ...) to messages.
    """

    def __init__(self, errors: Mapping[str, list[str]]) -> None:
        """Store the errors.

        Args:
            errors: Mapping from document path to messages.
        """
        super().__init__("; ".join(m for messages in errors.values() for m in messages))
        self.errors: dict[str, list[str]] = {key: list(value) for key, value in errors.items()}


# ---------------------------------------------------------------- export


class _ExportDumper(yaml.SafeDumper):
    """Safe dumper with Decimal support and inline lists of scalars."""


def _float_holds_exactly(value: Decimal) -> bool:
    """Return whether ``float(value)`` parses back to the same Decimal.

    Args:
        value: Finite Decimal.

    Returns:
        True when ``Decimal(str(float(value))) == value``.
    """
    return Decimal(str(float(value))) == value


def _represent_decimal(dumper: yaml.SafeDumper, value: Decimal) -> yaml.Node:
    """Write a Decimal as a plain YAML float, or as a string when a float would round it.

    Args:
        dumper: The dumper.
        value: Decimal value.

    Returns:
        The scalar node.
    """
    text = format(value, "f")
    if value.is_finite() and _float_holds_exactly(value):
        if "." not in text:
            text += ".0"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)
    return dumper.represent_str(text)


def _represent_list(dumper: yaml.SafeDumper, value: list[Any]) -> yaml.Node:
    """Write lists of scalars inline (``[0, 200]``) and other lists as blocks.

    Args:
        dumper: The dumper.
        value: The list.

    Returns:
        The sequence node.
    """
    flow = bool(value) and all(not isinstance(item, (dict, list)) for item in value)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flow)


_ExportDumper.add_representer(Decimal, _represent_decimal)
_ExportDumper.add_representer(list, _represent_list)


def resource_document(resource: Resource) -> dict[str, Any]:
    """Return the document entry of one resource.

    Args:
        resource: The resource.

    Returns:
        ``{"name", "energy_type", <cost keys>}`` with Decimal costs.
    """
    return {
        "name": resource.name,
        "energy_type": EnergyType(resource.energyType).name,
        **{key: Decimal(getattr(resource, attr)) for key, attr in RESOURCE_COST_FIELDS},
    }


def scenario_document(simulation: Simulation) -> dict[str, Any]:
    """Build the version-1 document of a proxy scenario.

    Args:
        simulation: Proxy simulation with a market.

    Returns:
        The document as plain Python data (dicts, lists, ints, Decimals,
        strings), in format key order.

    Raises:
        ValueError: ``simulation`` is not a proxy simulation.
    """
    if not simulation.proxy:
        raise ValueError(f"Simulation {simulation.pk} is not a proxy simulation.")
    market = simulation.market
    resources: dict[int, Resource] = {}
    agents: list[dict[str, Any]] = []
    for agent in market.agent_set.order_by("id"):
        plants = list(
            Plant.objects.filter(portfolio__agent=agent)
            .order_by("id")
            .select_related("resource")
        )
        for plant in plants:
            resources.setdefault(plant.resource_id, plant.resource)
        agents.append(
            {
                "name": agent.name,
                "type": AgentType(agent.type).name,
                "budget": Decimal(agent.budget),
                "algorithm": str(agent.algorithm_type),
                "algorithm_params": copy.deepcopy(dict(agent.algorithm_params or {})),
                "failstack": {
                    "enabled": bool(agent.failstack_enabled),
                    "limit": int(agent.failstack_limit),
                    "delta": Decimal(agent.failstack_delta),
                    "min_margin": Decimal(agent.failstack_min_margin),
                },
                "plants": [
                    {"resource": plant.resource.name, "capacity": int(plant.capacity)}
                    for plant in plants
                ],
            }
        )
        # The library agent of a scenario copy can carry another name (e.g.
        # scenario agent SOLAR copied from library agent SOLAR_2). Exporting it
        # lets the import match the right library agent.
        source = agent.source
        if source is not None and source.name != agent.name:
            agents[-1]["library_agent"] = source.name
    resource_docs = [
        resource_document(resource)
        for resource in sorted(resources.values(), key=lambda r: (r.name.lower(), r.id))
    ]
    return {
        "format_version": FORMAT_VERSION,
        "scenario": {
            "name": simulation.name,
            "mode": SimulationMode(simulation.mode).name,
            "period_type": PeriodType(simulation.periodType).name,
            "periods": int(simulation.periodNumber),
            "seed": int(simulation.seed),
            "evaluation_periods": int(simulation.evaluation_periods),
            "observation_lags": [int(lag) for lag in simulation.observation_lags],
            **(
                {"demand": copy.deepcopy(simulation.demand_profile)}
                if simulation.demand_profile is not None
                else {}
            ),
            **(
                {"demand_profile": simulation.demand_profile_source.name}
                if simulation.demand_profile is not None
                and simulation.demand_profile_source is not None
                else {}
            ),
        },
        "market": {
            "strategy": MarketStrategy(market.strategy).name,
            "clearing_algorithm": str(market.clearing_algorithm),
            **(
                {"clearing_params": copy.deepcopy(dict(market.clearing_params))}
                if market.clearing_params
                else {}
            ),
            "bid_bounds": [int(market.lowerBidBound), int(market.upperBidBound)],
        },
        "resources": resource_docs,
        "agents": agents,
    }


def export_scenario_yaml(simulation: Simulation, *, now: datetime | None = None) -> str:
    """Serialise a proxy scenario as YAML text with a header comment.

    Args:
        simulation: Proxy simulation.
        now: Export time for the header, ``timezone.now()`` when None.

    Returns:
        The YAML text (UTF-8 safe, keys in format order).

    Raises:
        ValueError: ``simulation`` is not a proxy simulation.
    """
    document = scenario_document(simulation)
    stamp = (now or timezone.now()).isoformat(timespec="seconds")
    header = (
        "# ABMEM scenario export\n"
        f"# Exported: {stamp}\n"
        f"# Source scenario id: {simulation.pk}\n"
    )
    body = yaml.dump(
        document,
        Dumper=_ExportDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return header + body


def export_resources_yaml(resources: Iterable[Resource]) -> str:
    """Serialise a list of resources as a ``kind: resources`` document.

    The ``resources`` list has the same entries as in a scenario document, so
    :func:`import_resources` applies the same matching rules.

    Args:
        resources: Resources to export, written in name order.

    Returns:
        The YAML text.
    """
    ordered = sorted(resources, key=lambda r: (r.name.lower(), r.id))
    document = {
        "format_version": FORMAT_VERSION,
        "kind": RESOURCES_KIND,
        "resources": [resource_document(resource) for resource in ordered],
    }
    body = yaml.dump(
        document,
        Dumper=_ExportDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return "# ABMEM resource list export\n" + body


# ---------------------------------------------------------------- parsing


def load_yaml_bytes(data: bytes) -> Any:
    """Decode and parse uploaded bytes with ``yaml.safe_load``.

    Args:
        data: File content.

    Returns:
        The parsed document (any YAML value).

    Raises:
        ScenarioImportError: Larger than ``MAX_FILE_BYTES``, not UTF-8, or
            not parseable by the safe loader (python tags are rejected).
    """
    if len(data) > MAX_FILE_BYTES:
        raise ScenarioImportError(
            {"file": [f"The file is larger than {MAX_FILE_BYTES // 1024} KB."]}
        )
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ScenarioImportError({"file": ["The file is not UTF-8 text."]}) from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ScenarioImportError({"file": [f"The file is not valid YAML: {exc}"]}) from None


class _Errors:
    """Ordered collection of path-keyed error messages."""

    def __init__(self) -> None:
        """Start empty."""
        self.items: dict[str, list[str]] = {}

    def add(self, key: str, message: str) -> None:
        """Add one message.

        Args:
            key: Document path.
            message: Human-readable message.
        """
        self.items.setdefault(key, []).append(message)

    def merge(self, exc: ValidationError, key_map: Mapping[str, str], default: str, prefix: str = "") -> None:
        """Merge a Django ValidationError, re-keying its fields.

        Args:
            exc: The error.
            key_map: Field name -> document path. Unknown fields use
                ``default``.
            default: Path used for unknown fields and field-less errors.
            prefix: Text put in front of every message.
        """
        if hasattr(exc, "error_dict"):
            items = exc.message_dict.items()
        else:
            items = [("", exc.messages)]
        for name, messages in items:
            for message in messages:
                self.add(key_map.get(name, default), prefix + message)

    def __bool__(self) -> bool:
        """Return True when there is at least one error."""
        return bool(self.items)


def _check_mapping(
    value: Any,
    path: str,
    allowed: tuple[str, ...],
    required: tuple[str, ...],
    errors: _Errors,
) -> dict[str, Any] | None:
    """Check that ``value`` is a mapping with only allowed and all required keys.

    Args:
        value: Parsed value.
        path: Document path for messages.
        allowed: Allowed keys.
        required: Required keys.
        errors: Error collection.

    Returns:
        The mapping, or None when ``value`` is not a mapping. Unknown or
        missing keys are reported but the mapping is still returned.
    """
    if not isinstance(value, dict):
        errors.add(path, f"{path} must be a mapping.")
        return None
    unknown = [str(key) for key in value if key not in allowed]
    if unknown:
        errors.add(
            path,
            f"Unknown key(s) in {path}: {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(allowed)}.",
        )
    missing = [key for key in required if key not in value]
    if missing:
        errors.add(path, f"Missing key(s) in {path}: {', '.join(missing)}.")
    return value


def _enum_by_name(raw: Any, enum_cls: Any, path: str, errors: _Errors) -> Any:
    """Return the enum member named ``raw`` (case-insensitive).

    Args:
        raw: Parsed value.
        enum_cls: Enum class.
        path: Document path for messages.
        errors: Error collection.

    Returns:
        The member, or None after reporting an error.
    """
    if isinstance(raw, str):
        name = raw.strip().upper()
        if name in enum_cls.__members__:
            return enum_cls[name]
    choices = ", ".join(member.name for member in enum_cls)
    errors.add(path, f"{path} must be one of {choices}, got {raw!r}.")
    return None


def _key_by_name(
    raw: Any, is_known: Any, known: list[str], path: str, errors: _Errors
) -> str | None:
    """Return the registered plugin key named by ``raw`` (case-insensitive).

    Args:
        raw: Parsed value.
        is_known: Predicate of the registry (``registry.is_known``).
        known: Registered keys for the message.
        path: Document path for messages.
        errors: Error collection.

    Returns:
        The key, or None after reporting an error.
    """
    if isinstance(raw, str) and is_known(raw.strip().upper()):
        return raw.strip().upper()
    errors.add(path, f"{path} must be one of {', '.join(known)}, got {raw!r}.")
    return None


def _to_decimal(raw: Any) -> Any:
    """Prepare a YAML number for a Django DecimalField without float rounding.

    Args:
        raw: int, float or numeric string.

    Returns:
        ``str(raw)`` for floats (the shortest repr, as written by the
        exporter), otherwise ``raw`` unchanged.
    """
    if isinstance(raw, float) and math.isfinite(raw):
        return str(raw)
    return raw


def _is_json_value(value: Any) -> bool:
    """Return whether ``value`` can be stored in a JSONField unchanged.

    Args:
        value: Parsed YAML value.

    Returns:
        True for None, bool, int, finite float, str, lists of such values and
        mappings with string keys.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


def _lower_name_filter(queryset: Any, name: str) -> Any:
    """Filter ``queryset`` by name with the database ``LOWER`` comparison.

    Args:
        queryset: Queryset of a model with a ``name`` field.
        name: Name to match.

    Returns:
        The filtered queryset.
    """
    return queryset.annotate(_lname=Lower("name")).filter(_lname=Lower(Value(name)))


@dataclass
class ResourceSpec:
    """One validated resource of the document.

    Attributes:
        index: Position in ``resources``.
        name: Name as in the file.
        energy_type: Energy type.
        costs: Model cost field name -> Decimal.
        existing: Matching existing resource (reused), or None (created).
    """

    index: int
    name: str
    energy_type: EnergyType
    costs: dict[str, Decimal]
    existing: Resource | None = None

    def instance(self) -> Resource:
        """Return the existing resource or an unsaved one for rule checks.

        Returns:
            A ``Resource`` with name and energy type set.
        """
        if self.existing is not None:
            return self.existing
        return Resource(name=self.name, energyType=self.energy_type, **self.costs)


@dataclass
class AgentSpec:
    """One validated agent of the document.

    Attributes:
        index: Position in ``agents``.
        name: Name as in the file (scenario agent name).
        fields: ``type``, ``budget``, ``algorithm_type`` and fail-stack fields.
        algorithm_params: Per-agent algorithm parameters.
        plants: ``(resource spec, capacity)`` pairs in file order.
        library_name: Name of the library agent (``library_agent`` key,
            defaults to ``name``).
        library: Matching library agent (reused), or None (created).
    """

    index: int
    name: str
    fields: dict[str, Any]
    algorithm_params: dict[str, Any]
    plants: list[tuple[ResourceSpec, int]]
    library_name: str = ""
    library: Agent | None = None

    def config_key(self) -> tuple[Any, ...]:
        """Return a hashable key of the configuration (for in-file comparison)."""
        fields = self.fields
        return (
            fields["type"],
            Decimal(fields["budget"]),
            fields["algorithm_type"],
            json.dumps(self.algorithm_params, sort_keys=True, default=str),
            bool(fields["failstack_enabled"]),
            int(fields["failstack_limit"]),
            Decimal(fields["failstack_delta"]),
            Decimal(fields["failstack_min_margin"]),
            tuple(sorted((r.name.lower(), c) for r, c in self.plants)),
        )


@dataclass
class ImportPlan:
    """A completely validated import, nothing written yet.

    Attributes:
        header: Scenario name, mode, period type, strategy and bid bounds.
        period_number: Number of periods.
        run_settings: Seed, evaluation periods, observation lags.
        resources: Resource specs in file order.
        agents: Agent specs in file order.
    """

    header: Any
    period_number: int
    run_settings: Any
    resources: list[ResourceSpec] = field(default_factory=list)
    agents: list[AgentSpec] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Return the preview shown before confirmation.

        Returns:
            ``{"scenario", "resources": {"reuse", "create"}, "agents":
            {"reuse", "create"}, "errors": {}}``.
        """
        return {
            "scenario": {
                "name": self.header.name,
                "mode": SimulationMode(self.header.mode).name,
                "period_type": PeriodType(self.header.period_type).name,
                "periods": self.period_number,
                "seed": self.run_settings.seed,
                "evaluation_periods": self.run_settings.evaluation_periods,
                "observation_lags": list(self.run_settings.observation_lags),
                "demand": self.run_settings.demand_profile,
                "demand_profile": _library_profile_name(
                    self.run_settings.demand_profile_source_id
                ),
                "strategy": MarketStrategy(self.header.strategy).name,
                "clearing_algorithm": self.header.clearing_algorithm,
                "bid_bounds": [self.header.lower_bound, self.header.upper_bound],
                "agents": [spec.name for spec in self.agents],
            },
            "resources": {
                "reuse": [s.existing.name for s in self.resources if s.existing is not None],
                "create": [s.name for s in self.resources if s.existing is None],
            },
            "agents": {
                "reuse": sorted({s.library.name for s in self.agents if s.library is not None}),
                "create": sorted(
                    {s.library_name for s in self.agents if s.library is None}, key=str.lower
                ),
            },
            "errors": {},
        }


def _format_value(value: Any) -> str:
    """Format a compared value for a conflict message.

    Args:
        value: Value of the file or of the database.

    Returns:
        Display text.
    """
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _settings_keys(exc: ValidationError) -> dict[str, str]:
    """Return the document paths of the field names of a settings error.

    ``demand_profile`` and ``demand_profile.<field>`` map to
    ``scenario.demand`` and ``scenario.demand.<field>``.

    Args:
        exc: Error of ``_parse_scenario_settings``.

    Returns:
        ``_WIZARD_KEYS`` extended by the demand profile fields of ``exc``.
    """
    keys = dict(_WIZARD_KEYS)
    prefix = wizard.DEMAND_PROFILE_KEY
    for name in getattr(exc, "error_dict", {}):
        if name == prefix or name.startswith(prefix + "."):
            keys[name] = "scenario.demand" + name[len(prefix):]
    return keys


def _library_profile_name(source_id: int | None) -> str | None:
    """Return the name of a library demand profile for the import preview.

    Args:
        source_id: Id of the linked library profile, or None.

    Returns:
        The library name, or None.
    """
    if source_id is None:
        return None
    template = DemandProfileTemplate.objects.filter(id=source_id).first()
    return template.name if template is not None else None


def _demand_source_id(scenario: Mapping[str, Any], errors: _Errors) -> int | None:
    """Return the library profile a scenario's demand is linked to on import.

    Args:
        scenario: The ``scenario`` mapping of the document.
        errors: Error collection (``scenario.demand_profile`` must be text
            or null).

    Returns:
        The id of the library profile named ``scenario.demand_profile`` when
        its settings equal ``scenario.demand``, else None.
    """
    name = scenario.get("demand_profile")
    if name is None:
        return None
    if not isinstance(name, str):
        errors.add(
            "scenario.demand_profile",
            "scenario.demand_profile must be the name of a library demand profile.",
        )
        return None
    template = demand_library.find_by_name(name)
    if template is None or scenario.get("demand") is None:
        return None
    try:
        equal = demand_library.validate_profile(
            scenario["demand"]
        ) == demand_library.validate_profile(template.profile)
    except DemandLibraryError:
        # scenario.demand is reported by the wizard validation.
        return None
    return template.id if equal else None


def _parse_header(document: dict[str, Any], errors: _Errors) -> tuple[Any, int | None, Any]:
    """Validate ``scenario`` and ``market`` with the wizard rules.

    Args:
        document: Top-level mapping.
        errors: Error collection.

    Returns:
        ``(header, period_number, run_settings)``, None entries on error.
    """
    scenario = _check_mapping(
        document.get("scenario"), "scenario", SCENARIO_KEYS, SCENARIO_REQUIRED_KEYS, errors
    )
    market = _check_mapping(
        document.get("market"), "market", MARKET_KEYS, MARKET_REQUIRED_KEYS, errors
    )
    if scenario is None or market is None:
        return None, None, None
    payload: dict[str, Any] = {
        "simulation_name": scenario.get("name"),
        "periodNumber": scenario.get("periods"),
    }
    for key in ("seed", "evaluation_periods", "observation_lags"):
        if key in scenario:
            payload[key] = scenario[key]
    if "demand" in scenario:
        payload[wizard.DEMAND_PROFILE_KEY] = scenario["demand"]
    source_id = _demand_source_id(scenario, errors)
    if source_id is not None:
        payload[wizard.DEMAND_SOURCE_KEY] = source_id
    enum_ok = True
    for yaml_key, source, wizard_key, enum_cls in (
        ("scenario.mode", scenario, "simulationMode", SimulationMode),
        ("scenario.period_type", scenario, "periodType", PeriodType),
        ("market.strategy", market, "strategy", MarketStrategy),
    ):
        member = _enum_by_name(source.get(yaml_key.split(".")[1]), enum_cls, yaml_key, errors)
        enum_ok = enum_ok and member is not None
        # A placeholder keeps the wizard validator from repeating the error.
        payload[wizard_key] = (member or next(iter(enum_cls))).value
    if market.get("clearing_algorithm") is not None:
        clearing = _key_by_name(
            market.get("clearing_algorithm"),
            clearing_registry.is_known_clearing,
            [info.key for info in clearing_registry.available_clearing_algorithms()],
            "market.clearing_algorithm",
            errors,
        )
        enum_ok = enum_ok and clearing is not None
        payload["clearing_algorithm"] = clearing
    if market.get("clearing_params") is not None:
        payload["clearing_params"] = market.get("clearing_params")
    bounds = market.get("bid_bounds")
    if isinstance(bounds, list) and len(bounds) == 2:
        payload["lowerBound"], payload["upperBound"] = bounds
    else:
        errors.add("market.bid_bounds", "market.bid_bounds must be a list [low, high].")
        payload["lowerBound"], payload["upperBound"] = 0, 1

    header = settings = None
    try:
        header = wizard._parse_scenario_header(payload)
    except ValidationError as exc:
        errors.merge(exc, _WIZARD_KEYS, "scenario")
    try:
        settings = wizard._parse_scenario_settings(payload, None)
    except ValidationError as exc:
        errors.merge(exc, _settings_keys(exc), "scenario")
    if not enum_ok or header is None or settings is None:
        return None, None, None
    return header, settings[0], settings[1]


def _parse_resources(raw: Any, errors: _Errors) -> dict[str, ResourceSpec]:
    """Validate ``resources`` and match them against existing resources.

    Args:
        raw: Value of ``resources``.
        errors: Error collection.

    Returns:
        Lower-cased name -> spec, for the resources without errors.
    """
    specs: dict[str, ResourceSpec] = {}
    if not isinstance(raw, list):
        errors.add("resources", "resources must be a list.")
        return specs
    seen: set[str] = set()
    for index, item in enumerate(raw):
        path = f"resources[{index}]"
        mapping = _check_mapping(item, path, RESOURCE_KEYS, RESOURCE_KEYS, errors)
        if mapping is None:
            continue
        ok = True
        name = mapping.get("name")
        name = name.strip() if isinstance(name, str) else ""
        if not name:
            errors.add(f"{path}.name", f"{path}.name is required.")
            ok = False
        elif len(name) > RESOURCE_NAME_MAX_LENGTH:
            errors.add(
                f"{path}.name",
                f"Resource name '{name}' must have at most {RESOURCE_NAME_MAX_LENGTH} "
                f"characters ({len(name)} given).",
            )
            ok = False
        elif name.lower() in seen:
            errors.add(f"{path}.name", f"Resource '{name}' is listed twice.")
            ok = False
        if name:
            seen.add(name.lower())
        energy_type = _enum_by_name(mapping.get("energy_type"), EnergyType, f"{path}.energy_type", errors)
        ok = ok and energy_type is not None
        costs: dict[str, Decimal] = {}
        for key, attr in RESOURCE_COST_FIELDS:
            if key not in mapping:
                ok = False
                continue
            raw_value = mapping[key]
            try:
                if isinstance(raw_value, bool) or raw_value is None:
                    raise ValidationError("a number is required.")
                costs[attr] = Resource._meta.get_field(attr).clean(_to_decimal(raw_value), None)
            except ValidationError as exc:
                ok = False
                for message in exc.messages:
                    errors.add(f"{path}.{key}", f"Resource '{name}' {key}: {message}")
        if not ok:
            continue
        spec = ResourceSpec(index, name, energy_type, costs)
        existing = _lower_name_filter(Resource.objects.all(), name).order_by("id").first()
        if existing is not None:
            differences: list[str] = []
            if int(existing.energyType) != energy_type.value:
                differences.append(
                    f"energy_type: file {energy_type.name} vs existing "
                    f"{EnergyType(existing.energyType).name}"
                )
            for key, attr in RESOURCE_COST_FIELDS:
                if Decimal(getattr(existing, attr)) != costs[attr]:
                    differences.append(
                        f"{key}: file {_format_value(costs[attr])} vs existing "
                        f"{_format_value(Decimal(getattr(existing, attr)))}"
                    )
            if differences:
                errors.add(
                    path,
                    f"Resource '{name}' already exists as '{existing.name}' with "
                    f"different values ({'; '.join(differences)}). Rename the "
                    "resource in the file or use the existing values.",
                )
                continue
            spec.existing = existing
        specs[name.lower()] = spec
    return specs


def _parse_agent(
    index: int,
    mapping: dict[str, Any],
    resources: dict[str, ResourceSpec],
    resource_errors: bool,
    errors: _Errors,
) -> AgentSpec | None:
    """Validate one agent entry with the wizard rules.

    Args:
        index: Position in ``agents``.
        mapping: The entry (keys already checked).
        resources: Valid resources by lower-cased name.
        resource_errors: Whether any resource had an error (unknown plant
            resources are then not reported again).
        errors: Error collection.

    Returns:
        The spec, or None after reporting errors.
    """
    path = f"agents[{index}]"
    label = f"Agent {index + 1} ({mapping.get('name', '')}): "
    ok = True
    name = ""
    try:
        name = wizard._validate_entry_name(mapping.get("name"))
    except ValidationError as exc:
        ok = False
        errors.add(f"{path}.name", label + exc.messages[0])
    library_name = name
    if "library_agent" in mapping:
        try:
            library_name = wizard._validate_entry_name(mapping.get("library_agent"))
        except ValidationError as exc:
            ok = False
            errors.add(f"{path}.library_agent", label + exc.messages[0])
    agent_type = _enum_by_name(mapping.get("type"), AgentType, f"{path}.type", errors)
    ok = ok and agent_type is not None
    fields: dict[str, Any] = {"type": agent_type}
    raw_budget = mapping.get("budget")
    try:
        fields["budget"] = wizard._coerce_budget(_to_decimal(raw_budget))
    except ValidationError as exc:
        ok = False
        for message in exc.messages:
            errors.add(f"{path}.budget", label + message)

    algorithm = _key_by_name(
        mapping.get("algorithm"),
        registry.is_known,
        [info.key for info in registry.available_algorithms()],
        f"{path}.algorithm",
        errors,
    )
    raw_failstack = mapping.get("failstack", {})
    failstack = _check_mapping(raw_failstack, f"{path}.failstack", FAILSTACK_KEYS, (), errors)
    if algorithm is None or failstack is None:
        ok = False
    else:
        try:
            fields.update(
                wizard._parse_agent_algorithm(
                    {"algorithm_type": algorithm, "failstack": failstack}
                )
            )
        except ValidationError as exc:
            ok = False
            errors.merge(
                exc,
                {key: f"{path}.{value}" for key, value in _FAILSTACK_YAML_KEYS.items()},
                f"{path}.failstack",
                label,
            )

    params = mapping.get("algorithm_params", {})
    if params is None:
        params = {}
    if not isinstance(params, dict) or not _is_json_value(params):
        ok = False
        errors.add(
            f"{path}.algorithm_params",
            label + "algorithm_params must be a mapping of names to plain values "
            "(numbers, strings, booleans, lists).",
        )
    elif algorithm is not None:
        try:
            registry.get_config_class(algorithm).from_dict(params)
        except (TypeError, ValueError) as exc:
            ok = False
            errors.add(f"{path}.algorithm_params", f"{label}{exc}")

    plants: list[tuple[ResourceSpec, int]] = []
    raw_plants = mapping.get("plants")
    if not isinstance(raw_plants, list) or not raw_plants:
        ok = False
        errors.add(f"{path}.plants", label + "At least one plant is required.")
        raw_plants = []
    plants_ok = True
    for number, item in enumerate(raw_plants, start=1):
        plant_path = f"{path}.plants[{number - 1}]"
        plant = _check_mapping(item, plant_path, PLANT_KEYS, PLANT_KEYS, errors)
        if plant is None or any(key not in plant for key in PLANT_KEYS):
            plants_ok = False
            continue
        try:
            capacity = wizard._coerce_capacity(plant.get("capacity"))
        except ValidationError as exc:
            plants_ok = False
            errors.add(f"{path}.plants", f"{label}Plant {number}: {exc.messages[0]}")
            continue
        resource_name = plant.get("resource")
        spec = resources.get(resource_name.strip().lower()) if isinstance(resource_name, str) else None
        if spec is None:
            plants_ok = False
            if not resource_errors:
                errors.add(
                    f"{path}.plants",
                    f"{label}Plant {number}: resource {resource_name!r} is not listed "
                    "under resources.",
                )
            continue
        plants.append((spec, capacity))
    ok = ok and plants_ok
    if ok:
        try:
            validate_agent_resources(agent_type, [spec.instance() for spec, _ in plants])
        except ValidationError as exc:
            ok = False
            errors.merge(exc, {}, f"{path}.plants", label)
    if not ok:
        return None
    return AgentSpec(index, name, fields, dict(params), plants, library_name=library_name)


def _library_differences(spec: AgentSpec, library: Agent) -> list[str]:
    """List the configuration differences between a file agent and a library agent.

    Args:
        spec: Validated file agent.
        library: Library agent with the same name.

    Returns:
        One ``'<key>: file X vs existing Y'`` text per differing field.
    """
    differences: list[str] = []
    fields = spec.fields
    comparisons = (
        ("type", fields["type"].name, AgentType(library.type).name if library.type is not None else None),
        ("budget", Decimal(fields["budget"]), Decimal(library.budget)),
        ("algorithm", fields["algorithm_type"], str(library.algorithm_type)),
        ("algorithm_params", spec.algorithm_params, dict(library.algorithm_params or {})),
        ("failstack.enabled", bool(fields["failstack_enabled"]), bool(library.failstack_enabled)),
        ("failstack.limit", int(fields["failstack_limit"]), int(library.failstack_limit)),
        ("failstack.delta", Decimal(fields["failstack_delta"]), Decimal(library.failstack_delta)),
        (
            "failstack.min_margin",
            Decimal(fields["failstack_min_margin"]),
            Decimal(library.failstack_min_margin),
        ),
    )
    for key, file_value, stored_value in comparisons:
        if file_value != stored_value:
            differences.append(
                f"{key}: file {_format_value(file_value)} vs existing {_format_value(stored_value)}"
            )
    stored_plants = list(
        Plant.objects.filter(portfolio__agent=library).select_related("resource")
    )
    file_counter = Counter(
        (spec_r.existing.id if spec_r.existing is not None else None, capacity)
        for spec_r, capacity in spec.plants
    )
    stored_counter = Counter((plant.resource_id, plant.capacity) for plant in stored_plants)
    if file_counter != stored_counter:
        file_text = ", ".join(sorted(f"{r.name} {c} MW" for r, c in spec.plants))
        stored_text = ", ".join(sorted(f"{p.resource.name} {p.capacity} MW" for p in stored_plants))
        differences.append(f"plants: file [{file_text}] vs existing [{stored_text}]")
    return differences


def build_import_plan(document: Any) -> ImportPlan:
    """Validate a parsed document and decide what is reused and created.

    Nothing is written. Every problem of the document is collected before
    raising.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        The plan.

    Raises:
        ScenarioImportError: Any validation error or conflict.
    """
    errors = _Errors()
    if not isinstance(document, dict):
        raise ScenarioImportError({"file": ["The file must contain a YAML mapping."]})
    if "format_version" not in document:
        raise ScenarioImportError({"format_version": ["format_version is missing."]})
    version = document.get("format_version")
    if isinstance(version, bool) or version != FORMAT_VERSION:
        raise ScenarioImportError(
            {
                "format_version": [
                    f"Unsupported format_version {version!r}; "
                    f"this version reads format_version {FORMAT_VERSION}."
                ]
            }
        )
    _check_mapping(document, "file", TOP_KEYS, TOP_KEYS, errors)
    header, period_number, run_settings = _parse_header(document, errors)
    resources = _parse_resources(document.get("resources", []), errors)
    resource_errors = any(key.startswith("resources") for key in errors.items)

    raw_agents = document.get("agents", [])
    agents: list[AgentSpec] = []
    if not isinstance(raw_agents, list):
        errors.add("agents", "agents must be a list.")
        raw_agents = []
    seen: dict[str, int] = {}
    for index, item in enumerate(raw_agents):
        mapping = _check_mapping(item, f"agents[{index}]", AGENT_KEYS, AGENT_REQUIRED_KEYS, errors)
        if mapping is None or any(key not in mapping for key in AGENT_REQUIRED_KEYS):
            continue
        spec = _parse_agent(index, mapping, resources, resource_errors, errors)
        if spec is None:
            continue
        key = spec.name.lower()
        if key in seen:
            errors.add(
                f"agents[{index}].name",
                f"Agent {index + 1} ({spec.name}): the name is also used by agent "
                f"{seen[key] + 1} (agent names are compared case-insensitively).",
            )
            continue
        seen[key] = index
        agents.append(spec)

    # A library agent may appear at most once in a scenario (OQ-14). When
    # several file agents name the same library agent (e.g. CCGT and CCGT2 of
    # an older scenario, both copied from library agent CCGT), the first one
    # uses it and every later one is matched under its own scenario name.
    used_library_names: set[str] = set()
    for spec in agents:
        key = spec.library_name.lower()
        if key in used_library_names:
            spec.library_name = spec.name
            key = spec.name.lower()
        used_library_names.add(key)
    for spec in agents:
        if sum(1 for other in agents if other.library_name.lower() == spec.library_name.lower()) > 1:
            errors.add(
                f"agents[{spec.index}].library_agent",
                f"Agent {spec.index + 1} ({spec.name}): library agent '{spec.library_name}' "
                "would be used twice in this scenario.",
            )

    conflicts: list[str] = []
    for spec in agents:
        library = (
            _lower_name_filter(Agent.objects.filter(market__isnull=True), spec.library_name)
            .order_by("id")
            .first()
        )
        if library is None:
            continue
        differences = _library_differences(spec, library)
        if differences:
            conflicts.append(spec.library_name)
            errors.add(
                f"agents[{spec.index}]",
                f"Library agent '{library.name}' already exists with a different "
                f"configuration ({'; '.join(differences)}).",
            )
        else:
            spec.library = library
    if conflicts:
        errors.add(
            "agents",
            "Conflicting library agents: " + ", ".join(conflicts)
            + ". Rename them in the file or align them with the library.",
        )
    if errors or header is None:
        raise ScenarioImportError(errors.items)
    return ImportPlan(header, period_number, run_settings, list(resources.values()), agents)


@transaction.atomic
def apply_import_plan(plan: ImportPlan) -> Simulation:
    """Write a plan: resources, library agents, scenario, market and agent copies.

    Args:
        plan: Output of ``build_import_plan`` in the same transaction.

    Returns:
        The new proxy simulation.

    Raises:
        LibraryValidationError: The library service rejected an agent (for
            example a name taken after validation). The transaction is
            rolled back.
    """
    for spec in plan.resources:
        if spec.existing is None:
            spec.existing = Resource.objects.create(
                name=spec.name, energyType=spec.energy_type, **spec.costs
            )
    created: dict[str, Agent] = {}
    for spec in plan.agents:
        if spec.library is not None:
            continue
        key = spec.library_name.lower()
        if key in created:
            spec.library = created[key]
            continue
        fields = spec.fields
        spec.library = created[key] = library_service.create_library_agent(
            AgentInput(
                name=spec.library_name,
                budget=fields["budget"],
                type=fields["type"],
                algorithm_type=fields["algorithm_type"],
                failstack_enabled=fields["failstack_enabled"],
                failstack_limit=fields["failstack_limit"],
                failstack_delta=fields["failstack_delta"],
                failstack_min_margin=fields["failstack_min_margin"],
                plants=tuple(
                    PlantInput(resource.existing.id, capacity)
                    for resource, capacity in spec.plants
                ),
                algorithm_params=spec.algorithm_params,
            )
        )
    header = plan.header
    simulation = simulation_factory.create(
        name=header.name,
        mode=header.mode,
        periodType=header.period_type,
        periodNumber=plan.period_number,
        proxy=True,
    )
    plan.run_settings.apply(simulation)
    simulation.save()
    market = market_factory.create(
        sim=simulation,
        strategy=header.strategy,
        lowerBound=header.lower_bound,
        upperBound=header.upper_bound,
        clearing_algorithm=header.clearing_algorithm,
        clearing_params=header.clearing_params,
    )
    for spec in plan.agents:
        library_service.copy_agent(spec.library, market, proxy=True, name=spec.name)
    return simulation


def import_resources(document: Any) -> list[Resource]:
    """Validate a ``kind: resources`` document and create the new resources.

    A resource whose name exists (case-insensitively) with the same values is
    reused. The same name with different values is an error, as in a
    scenario import.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        The resources of the document in file order (reused or created).

    Raises:
        ScenarioImportError: Validation error or conflict (nothing written).
    """
    errors = _Errors()
    mapping = _check_mapping(
        document,
        "document",
        ("format_version", "kind", "resources"),
        ("format_version", "kind", "resources"),
        errors,
    )
    specs: dict[str, ResourceSpec] = {}
    if mapping is not None:
        if mapping.get("format_version") != FORMAT_VERSION:
            errors.add(
                "format_version",
                f"Unsupported format_version {mapping.get('format_version')!r}; "
                f"expected {FORMAT_VERSION}.",
            )
        if mapping.get("kind") != RESOURCES_KIND:
            errors.add("kind", f"kind must be '{RESOURCES_KIND}'.")
        specs = _parse_resources(mapping.get("resources"), errors)
    if errors:
        raise ScenarioImportError(errors.items)
    with transaction.atomic():
        result: list[Resource] = []
        for spec in sorted(specs.values(), key=lambda item: item.index):
            if spec.existing is None:
                spec.existing = Resource.objects.create(
                    name=spec.name, energyType=spec.energy_type, **spec.costs
                )
            result.append(spec.existing)
    return result


def import_scenario(document: Any) -> tuple[ImportPlan, Simulation]:
    """Validate and write a document in one transaction.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        ``(plan, simulation)``.

    Raises:
        ScenarioImportError: Validation error or conflict (nothing written).
        LibraryValidationError: Rejected by the library service during the
            write (nothing written).
    """
    with transaction.atomic():
        plan = build_import_plan(document)
        simulation = apply_import_plan(plan)
    return plan, simulation

