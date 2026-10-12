"""Agent library service: library agent creation and agent copying (PLAN 2.3, ADR-005).

Every creation of a library agent and every copy of an agent (library ->
scenario, scenario -> run) goes through this module.

* A library agent has ``proxy=True`` and ``market=None``. Library names are
  unique case-insensitively, as computed by the database ``LOWER`` function
  (constraint ``uniq_library_agent_name_ci``).
* Inside one market names are unique case-sensitively
  (constraint ``uniq_agent_name_per_market``).
* A copy carries ``COPIED_AGENT_FIELDS``, the portfolio and all plants.
  ``source`` of a copy is the library agent it descends from.

Name comparison semantics: the library check delegates lower-casing to the
database (``Lower('name') == Lower(Value(name))``) instead of Python's
``str.lower``/``str.casefold``. The service therefore accepts and rejects
exactly the names the database constraint accepts and rejects. On PostgreSQL
``LOWER`` depends on the database collation/ctype. For example it does not
map the Turkish dotted capital ``'İ'`` to ``'i'`` (Python's ``'İ'.lower()``
is ``'i̇'``), so ``'İzmir'`` and ``'izmir'`` are two different library names
there, both for the database and for this service. A database
whose server encoding cannot represent ``'İ'`` (for example WIN1252)
rejects such names with ``DataError`` before any comparison.

All validation failures raise ``LibraryValidationError``, a
``django.core.exceptions.ValidationError`` subclass whose ``message_dict`` is
keyed by field name (``name``, ``plants``, ``type``, ...). Views map it to
HTTP 400.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Iterable, Mapping

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import Value
from django.db.models.functions import Lower

from ...models.agent import Agent
from ...models.enums import (
    AgentState,
    AgentType,
    MarketState,
    SimulationState,
)
from ...models.market import Market
from ...models.models import Plant, Portfolio, Resource
from ...models.simulation import Simulation
from .resource_rules import validate_agent_resources

NAME_MAX_LENGTH: Final[int] = Agent._meta.get_field("name").max_length or 20

COPIED_AGENT_FIELDS: Final[tuple[str, ...]] = (
    "name",
    "budget",
    "type",
    "algorithm_type",
    "algorithm_params",
    "failstack_enabled",
    "failstack_limit",
    "failstack_delta",
    "failstack_min_margin",
)
"""Agent fields carried by ``copy_agent``.

Every concrete ``Agent`` field except ``NOT_COPIED_AGENT_FIELDS`` must be
listed here (enforced by a test), so a new field cannot be silently dropped.
"""

NOT_COPIED_AGENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "market",
        "proxy",
        "state",
        "source",
        "created_at",
        "updated_at",
        "effective_params",
    }
)

_NOT_COPIED_ROW_FIELDS: Final[frozenset[str]] = frozenset(
    {"id", "created_at", "updated_at"}
)

# Market fields a scenario or run copy never takes over. The parameter
# snapshot belongs to the run that wrote it (PLAN 2.12).
_NOT_COPIED_MARKET_FIELDS: Final[frozenset[str]] = _NOT_COPIED_ROW_FIELDS | {
    "simulation",
    "effective_clearing_params",
}


class LibraryValidationError(ValidationError):
    """Deterministic, field-keyed validation error of the library service.

    Subclass of ``ValidationError`` so that existing view helpers that merge
    ``ValidationError`` messages keep working. Views map it to HTTP 400.
    """


@dataclass(frozen=True)
class PlantInput:
    """One plant of an agent input.

    Attributes:
        resource_id: Primary key of the ``Resource``.
        capacity: Installed capacity in MW, must be positive.
    """

    resource_id: int
    capacity: int


@dataclass(frozen=True)
class AgentInput:
    """Validated agent input built by views from JSON or form data.

    Attributes:
        name: Agent name (surrounding whitespace is stripped by the service).
        budget: Initial budget.
        type: Agent type.
        algorithm_type: Key of a registered agent algorithm.
        failstack_enabled: Whether the fail-stack rule is active.
        failstack_limit: Fail-stack limit, at least 1.
        failstack_delta: Fail-stack price step, at least 0.
        failstack_min_margin: Fail-stack minimum margin.
        plants: Plants of the agent, at least one.
        algorithm_params: Per-agent algorithm parameters. ``None`` means an
            empty mapping on creation and "keep the stored value" on update.
    """

    name: str
    budget: Decimal
    type: AgentType
    algorithm_type: str
    failstack_enabled: bool
    failstack_limit: int
    failstack_delta: Decimal
    failstack_min_margin: Decimal
    plants: tuple[PlantInput, ...]
    algorithm_params: Mapping[str, Any] | None = None


def normalize_name(name: str) -> str:
    """Strip surrounding whitespace from an agent name.

    Args:
        name: Raw name.

    Returns:
        The name without leading and trailing whitespace. No other change is
        made (case is preserved).
    """
    return name.strip()


def scenarios_using(library_agent: Agent) -> list[Simulation]:
    """Return the proxy scenarios that contain a copy of a library agent.

    Runs (non-proxy simulations) are not included: they are historical
    records with their own copies.

    Args:
        library_agent: A library agent.

    Returns:
        Distinct proxy simulations ordered by name and id.
    """
    return list(
        Simulation.objects.filter(proxy=True, market__agent__source=library_agent)
        .distinct()
        .order_by("name", "id")
    )


@transaction.atomic
def delete_library_agent(library_agent: Agent) -> None:
    """Delete a library agent unless a scenario still uses it.

    Copies in runs keep their configuration and lose the ``source`` link.

    Args:
        library_agent: The library agent to delete.

    Raises:
        LibraryValidationError: ``library_agent`` is not a library agent, or
            one or more proxy scenarios contain a copy of it (key ``agent``,
            code ``in_use``; the message names the scenarios).
    """
    if not library_agent.is_library:
        raise LibraryValidationError(
            {"agent": ["Only library agents can be deleted here."]}, code="not_library"
        )
    used = scenarios_using(library_agent)
    if used:
        names = ", ".join(f"'{s.name}' (#{s.id})" for s in used)
        raise LibraryValidationError(
            {
                "agent": [
                    f"Library agent '{library_agent.name}' is used by the scenario(s) "
                    f"{names}. Remove it from those scenarios or delete them first."
                ]
            },
            code="in_use",
        )
    library_agent.delete()


def check_name_available(
    name: str, *, market: Market | None = None, exclude_id: int | None = None
) -> bool:
    """Return whether ``name`` can be used without violating a name constraint.

    Args:
        name: Candidate name. It is normalised with ``normalize_name``.
        market: ``None`` checks the library (case-insensitive, database
            ``LOWER``), otherwise the names inside this market
            (case-sensitive).
        exclude_id: Agent id to ignore (the agent being edited).

    Returns:
        False for an empty name or a name that is already taken, else True.
    """
    normalized = normalize_name(name)
    if not normalized:
        return False
    if market is None:
        queryset = Agent.objects.annotate(_lname=Lower("name")).filter(
            market__isnull=True, _lname=Lower(Value(normalized))
        )
    else:
        queryset = Agent.objects.filter(market=market, name=normalized)
    if exclude_id is not None:
        queryset = queryset.exclude(id=exclude_id)
    return not queryset.exists()


def _error(field: str, message: str, code: str) -> LibraryValidationError:
    """Build a single-field ``LibraryValidationError``.

    Args:
        field: Field key.
        message: Human-readable message.
        code: Machine-readable error code.

    Returns:
        The error instance.
    """
    return LibraryValidationError({field: [ValidationError(message, code=code)]})


def _name_taken_error(name: str, market: Market | None) -> LibraryValidationError:
    """Return the error for a name that is already in use.

    Args:
        name: The normalised name.
        market: ``None`` for the library, else the target market.

    Returns:
        The error instance keyed by ``name``.
    """
    if market is None:
        message = (
            f"A library agent named '{name}' already exists "
            "(names are compared case-insensitively)."
        )
    else:
        message = f"An agent named '{name}' already exists in this scenario."
    return _error("name", message, "name_taken")


def _validate_name(name: str, *, exclude_id: int | None = None) -> str:
    """Normalise and validate a library agent name.

    Args:
        name: Raw name.
        exclude_id: Agent id to ignore in the uniqueness check.

    Returns:
        The normalised name.

    Raises:
        LibraryValidationError: Empty, too long or already taken.
    """
    if not isinstance(name, str):
        raise _error("name", "Agent name must be a string.", "invalid")
    normalized = normalize_name(name)
    if not normalized:
        raise _error("name", "Agent name is required.", "required")
    if len(normalized) > NAME_MAX_LENGTH:
        raise _error(
            "name",
            f"Agent name must have at most {NAME_MAX_LENGTH} characters "
            f"({len(normalized)} given).",
            "max_length",
        )
    if not check_name_available(normalized, market=None, exclude_id=exclude_id):
        raise _name_taken_error(normalized, None)
    return normalized


def _validate_enum(value: Any, enum_cls: Any, field: str) -> Any:
    """Return ``value`` as a member of ``enum_cls``.

    Args:
        value: Enum member or integer value.
        enum_cls: ``django_enumfield`` enum class.
        field: Field key used in the error.

    Returns:
        The enum member.

    Raises:
        LibraryValidationError: Value is not a valid member.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(field, f"Invalid {field}: {value!r}.", "invalid_choice")
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise _error(field, f"Invalid {field}: {value!r}.", "invalid_choice") from exc


def validate_algorithm_key(value: Any, field: str = "algorithm_type") -> str:
    """Return ``value`` when it is the key of a registered agent algorithm.

    Args:
        value: Algorithm key.
        field: Field key used in the error.

    Returns:
        The key.

    Raises:
        LibraryValidationError: ``value`` is not a registered key (also for a
            key whose class was removed from the code).
    """
    from ..algorithms import registry

    if not registry.is_known(value):
        known = ", ".join(info.key for info in registry.available_algorithms())
        raise _error(
            field,
            f"Algorithm {value!r} is not available. Known algorithms: {known}.",
            "invalid_choice",
        )
    return str(value)


def _validate_failstack(limit: Any, delta: Any, min_margin: Any) -> None:
    """Check the fail-stack values against the database constraints.

    Args:
        limit: Fail-stack limit.
        delta: Fail-stack price step.
        min_margin: Fail-stack minimum margin.

    Raises:
        LibraryValidationError: Limit below 1, negative or non-finite delta,
            non-finite minimum margin.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise _error(
            "failstack_limit", "Fail-stack limit must be an integer >= 1.", "invalid"
        )
    if not Decimal(delta).is_finite() or Decimal(delta) < 0:
        raise _error(
            "failstack_delta",
            "Fail-stack delta must be a finite number >= 0.",
            "invalid",
        )
    if not Decimal(min_margin).is_finite():
        raise _error(
            "failstack_min_margin",
            "Fail-stack minimum margin must be a finite number.",
            "invalid",
        )


def _resolve_plants(
    agent_type: AgentType, plants: tuple[PlantInput, ...]
) -> list[tuple[Resource, int]]:
    """Validate plant rows and the agent-type resource rule.

    Args:
        agent_type: Agent type of the owner.
        plants: Plant inputs.

    Returns:
        ``(resource, capacity)`` pairs in input order.

    Raises:
        LibraryValidationError: No plants, unknown resource, capacity not a
            positive integer, or a resource not allowed for ``agent_type``.
    """
    rows = tuple(plants)
    if not rows:
        raise _error("plants", "At least one plant is required.", "no_plants")
    for plant in rows:
        capacity = plant.capacity
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise _error(
                "plants",
                f"Plant capacity must be a positive integer, got {capacity!r}.",
                "invalid_capacity",
            )
    resource_ids = {plant.resource_id for plant in rows}
    resources = Resource.objects.in_bulk(resource_ids)
    missing = sorted(rid for rid in resource_ids if rid not in resources)
    if missing:
        raise _error(
            "plants",
            f"Unknown resource id(s): {', '.join(str(r) for r in missing)}.",
            "unknown_resource",
        )
    resolved = [(resources[plant.resource_id], plant.capacity) for plant in rows]
    try:
        validate_agent_resources(agent_type, [resource for resource, _ in resolved])
    except ValidationError as exc:
        if hasattr(exc, "error_dict"):
            raise LibraryValidationError(exc.error_dict) from exc
        raise LibraryValidationError({"plants": exc.error_list}) from exc
    return resolved


def _validated_fields(data: AgentInput, *, exclude_id: int | None) -> dict[str, Any]:
    """Validate an ``AgentInput`` and return the agent field values.

    Args:
        data: Agent input.
        exclude_id: Agent id ignored by the name uniqueness check.

    Returns:
        Mapping of agent field name to value (without ``algorithm_params``).

    Raises:
        LibraryValidationError: On the first invalid field.
    """
    name = _validate_name(data.name, exclude_id=exclude_id)
    agent_type = _validate_enum(data.type, AgentType, "type")
    algorithm_type = validate_algorithm_key(data.algorithm_type)
    try:
        budget = Decimal(data.budget)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise _error("budget", "Budget must be a number.", "invalid") from exc
    if not budget.is_finite():
        raise _error("budget", "Budget must be a finite number.", "invalid")
    _validate_failstack(
        data.failstack_limit, data.failstack_delta, data.failstack_min_margin
    )
    return {
        "name": name,
        "budget": budget,
        "type": agent_type,
        "algorithm_type": algorithm_type,
        "failstack_enabled": bool(data.failstack_enabled),
        "failstack_limit": data.failstack_limit,
        "failstack_delta": Decimal(data.failstack_delta),
        "failstack_min_margin": Decimal(data.failstack_min_margin),
    }


def _clean_fields(agent: Agent) -> None:
    """Run Django field validation (lengths, decimal digits) on ``agent``.

    Args:
        agent: Unsaved or modified agent.

    Raises:
        LibraryValidationError: With the field-keyed messages of
            ``Model.clean_fields``.
    """
    try:
        agent.clean_fields(exclude={"market", "source", "state"})
    except ValidationError as exc:
        raise LibraryValidationError(exc.message_dict) from exc


def _save_agent(agent: Agent, market: Market | None) -> None:
    """Save ``agent`` and convert a name constraint race into a validation error.

    Args:
        agent: Agent to save.
        market: Market used for the name check when converting the error.

    Raises:
        LibraryValidationError: The name was taken between the check and the
            insert (database constraint raised ``IntegrityError``).
        IntegrityError: Any other constraint violation.
    """
    try:
        with transaction.atomic():
            agent.save()
    except IntegrityError:
        if not check_name_available(agent.name, market=market, exclude_id=agent.id):
            raise _name_taken_error(agent.name, market) from None
        raise


def _write_plants(agent: Agent, plants: list[tuple[Resource, int]]) -> None:
    """Replace the plants of ``agent`` (portfolio created when missing).

    Args:
        agent: Saved agent.
        plants: ``(resource, capacity)`` pairs.
    """
    portfolio, _ = Portfolio.objects.get_or_create(
        agent=agent, defaults={"proxy": agent.proxy}
    )
    portfolio.plant_set.all().delete()
    Plant.objects.bulk_create(
        [
            Plant(
                portfolio=portfolio,
                resource=resource,
                capacity=capacity,
                proxy=agent.proxy,
            )
            for resource, capacity in plants
        ]
    )


@transaction.atomic
def create_library_agent(data: AgentInput) -> Agent:
    """Create a library agent with its portfolio and plants.

    Args:
        data: Agent input.

    Returns:
        The saved library agent (``proxy=True``, ``market=None``).

    Raises:
        LibraryValidationError: Name empty, longer than ``NAME_MAX_LENGTH``
            or already used by a library agent (case-insensitive), invalid
            enum, fail-stack or budget value, missing/unknown plants,
            non-positive capacity, or a resource not allowed for the type.
    """
    fields = _validated_fields(data, exclude_id=None)
    plants = _resolve_plants(fields["type"], data.plants)
    agent = Agent(
        market=None,
        proxy=True,
        state=AgentState.CREATED,
        algorithm_params=copy.deepcopy(dict(data.algorithm_params or {})),
        **fields,
    )
    _clean_fields(agent)
    _save_agent(agent, None)
    _write_plants(agent, plants)
    return agent


@transaction.atomic
def update_library_agent(agent: Agent, data: AgentInput) -> Agent:
    """Update a library agent and replace its plants.

    Copies in scenarios and runs are separate rows and are not changed.

    Args:
        agent: The library agent to update.
        data: New values.

    Returns:
        The updated agent.

    Raises:
        ValueError: ``agent`` is not a library agent.
        LibraryValidationError: Same conditions as ``create_library_agent``.
    """
    if not agent.is_library:
        raise ValueError(f"Agent {agent.pk} is not a library agent.")
    fields = _validated_fields(data, exclude_id=agent.pk)
    plants = _resolve_plants(fields["type"], data.plants)
    for key, value in fields.items():
        setattr(agent, key, value)
    if data.algorithm_params is not None:
        agent.algorithm_params = copy.deepcopy(dict(data.algorithm_params))
    _clean_fields(agent)
    _save_agent(agent, None)
    _write_plants(agent, plants)
    return agent


def free_agent_name(base: str, taken: Iterable[str]) -> str:
    """Return the first of ``base``, ``base_2``, ``base_3``, ... not in ``taken``.

    Names are compared case-insensitively after ``normalize_name``. When a
    suffixed name would be longer than ``NAME_MAX_LENGTH``, the base is
    shortened (and trailing whitespace removed) so that the suffix fits.

    Args:
        base: Preferred name, usually the name of the copied agent.
        taken: Names already in use, for example those of a market.

    Returns:
        The normalised ``base`` when it is free (also when it is empty or too
        long, so that the caller's validation reports it), else the first
        free suffixed name, which has at most ``NAME_MAX_LENGTH`` characters.
    """
    used = {normalize_name(item).lower() for item in taken}
    normalized = normalize_name(base)
    if normalized.lower() not in used:
        return normalized
    number = 2
    while True:
        suffix = f"_{number}"
        stem = normalized[: max(NAME_MAX_LENGTH - len(suffix), 0)].rstrip()
        candidate = f"{stem}{suffix}"
        if candidate.lower() not in used:
            return candidate
        number += 1


@transaction.atomic
def copy_agent(
    src: Agent, market: Market, *, proxy: bool, name: str | None = None
) -> Agent:
    """Deep-copy an agent with its portfolio and plants into ``market``.

    Args:
        src: Library, scenario or run agent to copy.
        market: Target market.
        proxy: ``proxy`` flag of the copy (True for scenario agents, False
            for run agents). Portfolio and plants get the same flag.
        name: Name of the copy. When None, the first of ``src.name``,
            ``<src.name>_2``, ``<src.name>_3``, ... that is free in
            ``market`` is used (case-insensitive, see ``free_agent_name``).
            A library agent can therefore be copied into one market several
            times (OQ-14).

    Returns:
        The saved copy. ``source`` is ``src`` when ``src`` is a library agent,
        else ``src.source``. ``state`` is ``CREATED``.

    Raises:
        LibraryValidationError: The name is empty, too long or already used
            in ``market``.
    """
    if name is None:
        new_name = free_agent_name(
            src.name, market.agent_set.values_list("name", flat=True)
        )
    else:
        new_name = normalize_name(name)
    if not new_name:
        raise _error("name", "Agent name is required.", "required")
    if len(new_name) > NAME_MAX_LENGTH:
        raise _error(
            "name",
            f"Agent name must have at most {NAME_MAX_LENGTH} characters "
            f"({len(new_name)} given).",
            "max_length",
        )
    if not check_name_available(new_name, market=market):
        raise _name_taken_error(new_name, market)

    values = {
        field: copy.deepcopy(getattr(src, field)) for field in COPIED_AGENT_FIELDS
    }
    values["name"] = new_name
    new_agent = Agent(
        market=market,
        proxy=proxy,
        state=AgentState.CREATED,
        source=src if src.is_library else src.source,
        **values,
    )
    _save_agent(new_agent, market)

    try:
        plants = [
            (plant.resource, plant.capacity)
            for plant in src.portfolio.plant_set.order_by("id").select_related(
                "resource"
            )
        ]
    except Agent.portfolio.RelatedObjectDoesNotExist:
        plants = []
    _write_plants(new_agent, plants)
    return new_agent


@transaction.atomic
def add_new_agent_to_scenario(data: AgentInput, market: Market) -> Agent:
    """'New agent' path: add the agent to the library, then copy it into ``market``.

    Args:
        data: Agent input.
        market: Market of the proxy scenario.

    Returns:
        The scenario copy (``proxy=True``, ``source`` = the new library agent).

    Raises:
        LibraryValidationError: From ``create_library_agent`` or
            ``copy_agent``. Nothing is saved in that case.
    """
    library_agent = create_library_agent(data)
    return copy_agent(library_agent, market, proxy=True, name=library_agent.name)


def _copy_row_values(instance: Any, exclude: frozenset[str]) -> dict[str, Any]:
    """Return the concrete field values of ``instance`` for a new row.

    Args:
        instance: Model instance.
        exclude: Field names (``Field.name``) to leave out.

    Returns:
        Mapping of ``Field.attname`` to a deep copy of its value.
    """
    return {
        field.attname: copy.deepcopy(getattr(instance, field.attname))
        for field in instance._meta.concrete_fields
        if field.name not in exclude
    }


COPY_NAME_SUFFIX: Final[str] = " (copy)"


def copy_scenario_name(name: str) -> str:
    """Return the name of a duplicated scenario, ``'<name> (copy)'``.

    The original name is shortened so that the result fits the
    ``Simulation.name`` column.

    Args:
        name: Name of the original scenario.

    Returns:
        The name of the copy.
    """
    max_length = Simulation._meta.get_field("name").max_length or 20
    base = name.strip()[: max(max_length - len(COPY_NAME_SUFFIX), 0)].rstrip()
    return f"{base}{COPY_NAME_SUFFIX}"


@transaction.atomic
def duplicate_scenario(proxy_sim: Simulation, *, name: str | None = None) -> Simulation:
    """Deep-copy a proxy scenario into a new proxy scenario (PLAN 2.6).

    Simulation settings, market settings and every agent with its portfolio
    and plants are copied. Agent copies keep the library agent they descend
    from (``source``) and their names (names are unique per market only).
    Run bookkeeping (state, current period, error, scenario and warm-start
    links) starts fresh.

    Args:
        proxy_sim: Proxy simulation to copy.
        name: Name of the copy, ``copy_scenario_name(proxy_sim.name)`` when None.

    Returns:
        The new proxy simulation (state ``CREATED``, ``currentPeriod=-1``).

    Raises:
        ValueError: ``proxy_sim`` is not a proxy simulation.
        LibraryValidationError: An agent could not be copied.
    """
    if not proxy_sim.proxy:
        raise ValueError(f"Simulation {proxy_sim.pk} is not a proxy simulation.")
    sim_values = _copy_row_values(proxy_sim, _NOT_COPIED_ROW_FIELDS)
    sim_values.update(
        name=copy_scenario_name(proxy_sim.name) if name is None else name,
        proxy=True,
        state=SimulationState.CREATED,
        currentPeriod=-1,
        error="",
        scenario_id=None,
        warm_start_from_id=None,
        freeze_policies=False,
    )
    new_sim = Simulation.objects.create(**sim_values)

    proxy_market = proxy_sim.market
    market_values = _copy_row_values(proxy_market, _NOT_COPIED_MARKET_FIELDS)
    market_values.update(proxy=True, state=MarketState.CREATED, simulation=new_sim)
    new_market = Market.objects.create(**market_values)

    for agent in proxy_market.agent_set.order_by("id"):
        copy_agent(agent, new_market, proxy=True)
    return new_sim


@transaction.atomic
def clone_scenario_for_run(
    proxy_sim: Simulation,
    *,
    seed: int | None = None,
    forced_algorithm: str | None = None,
    algorithm_overrides: Mapping[str, Any] | None = None,
    fixed_agent_fields: Mapping[str, Any] | None = None,
    warm_start_from: Simulation | None = None,
    freeze_policies: bool = False,
) -> Simulation:
    """Copy a proxy scenario into a new run (simulation, market, agents).

    The run records the scenario it was cloned from (``Simulation.scenario``)
    and its warm-start origin (finding A9).

    Args:
        proxy_sim: Proxy simulation (scenario template).
        seed: Seed of the run, ``proxy_sim.seed`` when ``None`` (0 is a valid
            seed and is kept).
        forced_algorithm: When given, the algorithm key of every copied agent
            (K7, HPO).
        algorithm_overrides: Merged into ``algorithm_params`` of every copied
            agent (keys of the override win).
        fixed_agent_fields: Values set on every copied agent (HPO fixed
            fail-stack settings). Keys must be in ``COPIED_AGENT_FIELDS``
            and must not be ``name``.
        warm_start_from: Finished run of this scenario whose saved policies
            initialise the run. Only matched against the scenario here; the
            files are checked by ``checkpoints.check_warm_start`` and loaded
            when the run starts.
        freeze_policies: Run the loaded policies without learning. Requires
            ``warm_start_from``.

    Returns:
        The new run simulation (``proxy=False``, state ``CREATED``,
        ``currentPeriod=-1``) with a market in state ``CREATED`` and one
        ``proxy=False`` copy of every scenario agent.

    Raises:
        ValueError: ``proxy_sim`` is not a proxy simulation,
            ``fixed_agent_fields`` contains a key that cannot be set, or
            ``freeze_policies`` is set without ``warm_start_from``.
        WarmStartError: ``warm_start_from`` is not a finished run of this
            scenario with saved policies (a ``ValueError``).
        LibraryValidationError: A forced algorithm or fixed fail-stack value
            is invalid.
    """
    if not proxy_sim.proxy:
        raise ValueError(f"Simulation {proxy_sim.pk} is not a proxy simulation.")
    fixed = dict(fixed_agent_fields or {})
    invalid_keys = sorted(
        key for key in fixed if key not in COPIED_AGENT_FIELDS or key == "name"
    )
    if invalid_keys:
        raise ValueError(f"fixed_agent_fields keys not allowed: {invalid_keys}.")
    if forced_algorithm is not None:
        forced_algorithm = validate_algorithm_key(forced_algorithm)
    if freeze_policies and warm_start_from is None:
        raise ValueError("freeze_policies requires warm_start_from.")
    if warm_start_from is not None:
        from ..simulation.checkpoints import resolve_warm_start

        resolve_warm_start(proxy_sim, warm_start_from)

    sim_values = _copy_row_values(proxy_sim, _NOT_COPIED_ROW_FIELDS)
    sim_values.update(
        proxy=False,
        state=SimulationState.CREATED,
        currentPeriod=-1,
        seed=proxy_sim.seed if seed is None else seed,
        error="",
        scenario_id=proxy_sim.pk,
        warm_start_from_id=(
            warm_start_from.pk if warm_start_from is not None else None
        ),
        freeze_policies=bool(freeze_policies),
    )
    new_sim = Simulation.objects.create(**sim_values)

    proxy_market = proxy_sim.market
    market_values = _copy_row_values(proxy_market, _NOT_COPIED_MARKET_FIELDS)
    market_values.update(proxy=False, state=MarketState.CREATED, simulation=new_sim)
    new_market = Market.objects.create(**market_values)

    for agent in proxy_market.agent_set.order_by("id"):
        new_agent = copy_agent(agent, new_market, proxy=False)
        changed = False
        if forced_algorithm is not None:
            new_agent.algorithm_type = forced_algorithm
            changed = True
        if algorithm_overrides:
            new_agent.algorithm_params = {
                **new_agent.algorithm_params,
                **copy.deepcopy(dict(algorithm_overrides)),
            }
            changed = True
        for key, value in fixed.items():
            setattr(new_agent, key, value)
            changed = True
        if changed:
            _validate_failstack(
                new_agent.failstack_limit,
                new_agent.failstack_delta,
                new_agent.failstack_min_margin,
            )
            _clean_fields(new_agent)
            new_agent.save()
    return new_sim
