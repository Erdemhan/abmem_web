import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models import Count
from django.db.models.functions import Lower
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from .forms import AgentForm, PlantForm, ResourceForm
from .models import Agent, DemandProfileTemplate, Simulation, enums
from .models.market import Market
from .models.simulation import (
    ALLOWED_OBSERVATION_LAGS,
    DEFAULT_SEED,
    default_observation_lags,
)
from .models.models import Plant, Portfolio, Resource
from .services import algorithm_catalog
from .services.agent import library_service
from .services.agent.library_service import (
    NAME_MAX_LENGTH,
    AgentInput,
    LibraryValidationError,
    PlantInput,
)
from .services.agent.resource_rules import (
    allowed_energy_types_by_agent_type,
    validate_agent_resources,
)
from .services.algorithms import registry as algorithm_registry
from .services.market.clearing_algorithms.registry import (
    DEFAULT_CLEARING_ALGORITHM,
    available_clearing_algorithms,
    is_known_clearing,
)
from .services.market import market_factory as MarketFactory
from .services.market.demand_generator import PAYLOAD_KEY, normalize_profile
from .services.simulation import checkpoints
from .services.simulation.progress import simulation_progress
from .services.simulation import simulation_factory as SimulationFactory
from .services.starter import starter_service as StarterService

logger = logging.getLogger(__name__)

# Payload key of the fail-stack object -> Agent model field (PLAN 2.1, K3).
FAILSTACK_PAYLOAD_FIELDS: tuple[tuple[str, str], ...] = (
    ("limit", "failstack_limit"),
    ("delta", "failstack_delta"),
    ("min_margin", "failstack_min_margin"),
)

# Agent entry modes of the scenario payload (spec agent_library,
# scenario_payload_contract). "scenario" keeps an agent that already belongs
# to the edited scenario (update page only).
MODE_NEW: Final[str] = "new"
MODE_LIBRARY: Final[str] = "library"
MODE_SCENARIO: Final[str] = "scenario"
SCENARIO_AGENT_MODES: Final[tuple[str, ...]] = (MODE_NEW, MODE_LIBRARY, MODE_SCENARIO)

# Exclusive upper bound of Simulation.seed. The column is a signed 32-bit
# IntegerField, so values >= 2**31 raise "integer out of range" before the
# check constraint simulation_seed_range (< 2**32) can apply.
SEED_UPPER_BOUND: Final[int] = 2**31

# Defaults of a new scenario in the wizard (PLAN 2.6): one week of hourly
# periods and bid bounds of 0-200 USD/MWh.
WIZARD_DEFAULT_PERIODS: Final[int] = 168
WIZARD_DEFAULT_LOWER_BOUND: Final[int] = 0
WIZARD_DEFAULT_UPPER_BOUND: Final[int] = 200

# Duration units of the wizard: (key, label, hours). periodNumber is always
# stored in hours. A month is 30 days and a year 365 days.
PERIOD_UNITS: Final[tuple[tuple[str, str, int], ...]] = (
    ("hour", "Hour", 1),
    ("day", "Day", 24),
    ("week", "Week", 168),
    ("month", "Month", 720),
    ("year", "Year", 8760),
)


def period_unit_for(period_number: int) -> tuple[str, int]:
    """Return the largest duration unit that divides a period count exactly.

    Args:
        period_number: Number of hourly periods.

    Returns:
        ``(unit key, count)`` with ``count * hours(unit) == period_number``.
        The unit is ``hour`` when no larger unit divides ``period_number``
        or when it is not positive.
    """
    if period_number > 0:
        for key, _, hours in sorted(PERIOD_UNITS, key=lambda unit: -unit[2]):
            if period_number % hours == 0:
                return key, period_number // hours
    return PERIOD_UNITS[0][0], period_number


def _add_errors(
    errors: dict[str, list[str]], exc: ValidationError, default_key: str, prefix: str
) -> None:
    """Merge the messages of a ValidationError into an error mapping.

    Args:
        errors: Mapping from field name to messages, updated in place.
        exc: The error to merge.
        default_key: Field name used when ``exc`` carries no field names.
        prefix: Text put in front of every message (for example the agent).
    """
    if hasattr(exc, "error_dict"):
        items = exc.message_dict.items()
    else:
        items = [(default_key, exc.messages)]
    for key, messages in items:
        errors.setdefault(key, []).extend(f"{prefix}{message}" for message in messages)


def _validation_error_response(exc: ValidationError) -> JsonResponse:
    """Build the HTTP 400 response for a rejected write.

    Args:
        exc: Error raised during payload validation.

    Returns:
        ``JsonResponse`` with body ``{"errors": {field: [messages]}}`` and status 400.
    """
    errors: dict[str, list[str]] = {}
    _add_errors(errors, exc, "plants", "")
    return JsonResponse({"errors": errors}, status=400)


def _coerce_agent_type(raw_type: Any) -> enums.AgentType:
    """Convert a JSON agent type (int or digit string) to ``AgentType``.

    Args:
        raw_type: Value of the ``type`` key in the payload.

    Returns:
        The matching ``AgentType`` member.

    Raises:
        ValidationError: ``{'type': [...]}`` when the value is missing, not an
            integer or not a valid agent type.
    """
    if isinstance(raw_type, bool):
        raise ValidationError({"type": ["Agent type must be an integer."]})
    if isinstance(raw_type, str) and raw_type.strip().lstrip("-").isdigit():
        raw_type = int(raw_type)
    if raw_type is None or raw_type == "":
        raise ValidationError({"type": ["Agent type is required."]})
    if not isinstance(raw_type, int):
        raise ValidationError({"type": ["Agent type must be an integer."]})
    try:
        return enums.AgentType(raw_type)
    except ValueError:
        raise ValidationError({"type": [f"Unknown agent type: {raw_type}."]})


def _coerce_algorithm_type(raw_type: Any) -> str:
    """Return the registered algorithm key named by a JSON value (PLAN 2.10).

    Args:
        raw_type: Value of the ``algorithm_type`` key, an algorithm key such as
            ``"DDPG"`` (surrounding spaces ignored). None or "" selects the
            model default (DDPG).

    Returns:
        The algorithm key.

    Raises:
        ValidationError: ``{'algorithm_type': [...]}`` for a value that is not
            the key of a registered algorithm.
    """
    if raw_type is None or raw_type == "":
        return str(Agent._meta.get_field("algorithm_type").get_default())
    key = raw_type.strip() if isinstance(raw_type, str) else raw_type
    if not algorithm_registry.is_known(key):
        known = ", ".join(
            info.key for info in algorithm_registry.available_algorithms()
        )
        raise ValidationError(
            {
                "algorithm_type": [
                    f"Algorithm {raw_type!r} is not available. Known algorithms: "
                    f"{known}."
                ]
            }
        )
    return str(key)


def _coerce_failstack_value(field_name: str, raw_value: Any) -> Any:
    """Validate one fail-stack value with the ``Agent`` model field rules.

    Args:
        field_name: ``failstack_limit``, ``failstack_delta`` or
            ``failstack_min_margin``.
        raw_value: Payload value (number or numeric string).

    Returns:
        The cleaned value (``int`` for the limit, ``Decimal`` otherwise).

    Raises:
        ValidationError: With a plain message when the value is not a finite
            number, does not fit the field, ``limit < 1`` or ``delta < 0``.
    """
    if isinstance(raw_value, bool) or raw_value is None or raw_value == "":
        raise ValidationError("a number is required.")
    if isinstance(raw_value, float) and not math.isfinite(raw_value):
        raise ValidationError(f"{raw_value} is not a finite number.")
    if isinstance(raw_value, float):
        raw_value = repr(raw_value)
    if field_name == "failstack_limit":
        text = str(raw_value).strip()
        if not text.lstrip("-").isdigit():
            raise ValidationError(f"{raw_value!r} is not an integer.")
        limit = int(text)
        if limit < 1:
            raise ValidationError(f"must be an integer >= 1, got {limit}.")
        return limit
    value = Agent._meta.get_field(field_name).clean(raw_value, None)
    if field_name == "failstack_delta" and value < 0:
        raise ValidationError(f"must be >= 0 USD/MWh, got {value}.")
    return value


def _coerce_switch(raw_value: Any) -> bool:
    """Convert a JSON switch value to bool.

    Args:
        raw_value: bool, or one of the strings "true", "false", "on", "off",
            "1", "0", "" (case-insensitive). None means False.

    Returns:
        The switch state.

    Raises:
        ValidationError: For any other value.
    """
    if raw_value is None:
        return False
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if text in ("true", "on", "1"):
            return True
        if text in ("false", "off", "0", ""):
            return False
    raise ValidationError(
        f"Fail-stack switch must be true or false, got {raw_value!r}."
    )


def _parse_agent_algorithm(agent_data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the algorithm and fail-stack settings of one agent payload.

    Expected keys: ``algorithm_type`` (registered algorithm key) and
    ``failstack`` (``{"enabled": bool, "limit": int, "delta": number,
    "min_margin": number}``). Values sent with ``enabled`` False are stored
    so the settings are kept for later, the rule stays inactive.

    Args:
        agent_data: One entry of the ``agents`` list.

    Returns:
        ``{algorithm_type, failstack_enabled, failstack_limit,
        failstack_delta, failstack_min_margin}``. Missing values take the
        model defaults.

    Raises:
        ValidationError: Mapping from field name to messages for an unknown
            algorithm, malformed or out-of-range fail-stack values, or an
            enabled fail-stack rule on a baseline algorithm.
    """
    errors: dict[str, list[str]] = {}
    result: dict[str, Any] = {
        model_field: Agent._meta.get_field(model_field).get_default()
        for _, model_field in FAILSTACK_PAYLOAD_FIELDS
    }
    try:
        result["algorithm_type"] = _coerce_algorithm_type(
            agent_data.get("algorithm_type")
        )
    except ValidationError as exc:
        _add_errors(errors, exc, "algorithm_type", "")
        result["algorithm_type"] = None

    raw_failstack = agent_data.get("failstack")
    if raw_failstack is not None and not isinstance(raw_failstack, Mapping):
        errors.setdefault("failstack", []).append(
            "Fail-stack settings must be an object."
        )
        raw_failstack = None
    raw_failstack = raw_failstack or {}

    try:
        result["failstack_enabled"] = _coerce_switch(raw_failstack.get("enabled"))
    except ValidationError as exc:
        errors.setdefault("failstack_enabled", []).extend(exc.messages)
        result["failstack_enabled"] = False

    for payload_key, model_field in FAILSTACK_PAYLOAD_FIELDS:
        if raw_failstack.get(payload_key) in (None, ""):
            continue
        try:
            result[model_field] = _coerce_failstack_value(
                model_field, raw_failstack[payload_key]
            )
        except ValidationError as exc:
            errors.setdefault(model_field, []).extend(
                f"Fail-stack {payload_key}: {message}" for message in exc.messages
            )

    if result["failstack_enabled"] and algorithm_registry.is_baseline(
        result["algorithm_type"]
    ):
        errors.setdefault("failstack_enabled", []).append(
            "The fail-stack rule cannot be enabled for the baseline algorithm "
            f"{result['algorithm_type']}."
        )
    if errors:
        raise ValidationError(errors)
    return result


def _parse_plants(raw_plants: Any) -> list[tuple[Resource, Any]]:
    """Resolve the plant rows of one agent payload.

    Args:
        raw_plants: Value of the ``plants`` key, a list of
            ``{"resource": <id>, "capacity": <value>}`` objects.

    Returns:
        Pairs of (resource, capacity) in payload order.

    Raises:
        ValidationError: ``{'plants': [...]}`` for a malformed row or an
            unknown resource id.
    """
    if raw_plants is None:
        return []
    if not isinstance(raw_plants, list):
        raise ValidationError({"plants": ["Plants must be a list."]})
    plants: list[tuple[Resource, Any]] = []
    messages: list[str] = []
    for index, plant_data in enumerate(raw_plants, start=1):
        raw_id = plant_data.get("resource") if isinstance(plant_data, dict) else None
        try:
            if isinstance(raw_id, bool):
                raise ValueError
            resource_id = int(raw_id)
        except (TypeError, ValueError):
            messages.append(f"Plant {index}: a resource must be selected.")
            continue
        try:
            resource = Resource.objects.get(id=resource_id)
        except Resource.DoesNotExist:
            messages.append(f"Plant {index}: resource {resource_id} does not exist.")
            continue
        plants.append((resource, plant_data.get("capacity")))
    if messages:
        raise ValidationError({"plants": messages})
    return plants


def _entry_key(index: int, field: str) -> str:
    """Return the error key of one field of one payload agent.

    Args:
        index: 0-based position of the agent in the ``agents`` list.
        field: Field name.

    Returns:
        ``"agents[<index>].<field>"``.
    """
    return f"agents[{index}].{field}"


def _coerce_budget(raw_value: Any) -> Decimal:
    """Validate a payload budget with the ``Agent.budget`` field rules.

    Args:
        raw_value: Number or numeric string.

    Returns:
        The budget as ``Decimal``.

    Raises:
        ValidationError: Missing, not a finite number or too many digits.
    """
    if isinstance(raw_value, bool) or raw_value is None or raw_value == "":
        raise ValidationError("Budget is required.")
    if isinstance(raw_value, float):
        if not math.isfinite(raw_value):
            raise ValidationError(f"Budget {raw_value} is not a finite number.")
        raw_value = repr(raw_value)
    return Agent._meta.get_field("budget").clean(raw_value, None)


def _coerce_capacity(raw_value: Any) -> int:
    """Convert a payload plant capacity to a positive integer.

    Args:
        raw_value: int, integral float or digit string.

    Returns:
        The capacity in MW.

    Raises:
        ValidationError: Not an integer or not positive.
    """
    value: int | None = None
    if isinstance(raw_value, int) and not isinstance(raw_value, bool):
        value = raw_value
    elif isinstance(raw_value, float) and math.isfinite(raw_value):
        if raw_value.is_integer():
            value = int(raw_value)
    elif isinstance(raw_value, str):
        text = raw_value.strip()
        if text.lstrip("-").isdigit():
            value = int(text)
    if value is None or value <= 0:
        raise ValidationError(
            f"capacity must be a positive integer, got {raw_value!r}."
        )
    return value


def _validate_entry_name(raw_name: Any) -> str:
    """Normalise and check the format of an agent name.

    Args:
        raw_name: Value of the ``name`` key.

    Returns:
        The name without surrounding whitespace.

    Raises:
        ValidationError: Not a string, empty, or longer than ``NAME_MAX_LENGTH``.
    """
    if not isinstance(raw_name, str) or not library_service.normalize_name(raw_name):
        raise ValidationError("Agent name is required.", code="required")
    name = library_service.normalize_name(raw_name)
    if len(name) > NAME_MAX_LENGTH:
        raise ValidationError(
            f"Agent name must have at most {NAME_MAX_LENGTH} characters "
            f"({len(name)} given).",
            code="max_length",
        )
    return name


def _library_name_taken_message(name: str) -> str:
    """Return the message for a name already used by a library agent.

    Args:
        name: The normalised name.

    Returns:
        Human-readable message (same wording as the library service).
    """
    return (
        f"A library agent named '{name}' already exists "
        "(names are compared case-insensitively)."
    )


def _parse_agent_fields(
    agent_data: Mapping[str, Any], prefix: str
) -> tuple[dict[str, Any], tuple[PlantInput, ...]]:
    """Validate the content fields of one 'new' or 'scenario' payload agent.

    Args:
        agent_data: One entry of the ``agents`` list.
        prefix: Text put in front of every message (the agent label).

    Returns:
        ``(fields, plants)``: ``fields`` holds ``type``, ``budget``,
        ``algorithm_type`` and the fail-stack fields, ``plants`` the plant
        inputs in payload order.

    Raises:
        ValidationError: Mapping from field name (``type``, ``budget``,
            ``algorithm_type``, ``failstack_*``, ``plants``) to prefixed
            messages.
    """
    errors: dict[str, list[str]] = {}
    fields: dict[str, Any] = {}
    agent_type = None
    resources: list[tuple[Resource, Any]] = []
    try:
        agent_type = _coerce_agent_type(agent_data.get("type"))
        fields["type"] = agent_type
    except ValidationError as exc:
        _add_errors(errors, exc, "type", prefix)
    try:
        fields["budget"] = _coerce_budget(agent_data.get("budget"))
    except ValidationError as exc:
        _add_errors(errors, exc, "budget", prefix)
    try:
        fields.update(_parse_agent_algorithm(agent_data))
    except ValidationError as exc:
        _add_errors(errors, exc, "algorithm_type", prefix)
    try:
        resources = _parse_plants(agent_data.get("plants"))
    except ValidationError as exc:
        _add_errors(errors, exc, "plants", prefix)
    plants: list[PlantInput] = []
    if "plants" not in errors:
        for number, (resource, raw_capacity) in enumerate(resources, start=1):
            try:
                plants.append(PlantInput(resource.id, _coerce_capacity(raw_capacity)))
            except ValidationError as exc:
                errors.setdefault("plants", []).extend(
                    f"{prefix}Plant {number}: {message}" for message in exc.messages
                )
    if agent_type is not None and "plants" not in errors:
        try:
            validate_agent_resources(
                agent_type, [resource for resource, _ in resources]
            )
        except ValidationError as exc:
            _add_errors(errors, exc, "plants", prefix)
    if errors:
        raise ValidationError(errors)
    return fields, tuple(plants)


@dataclass(frozen=True)
class ScenarioAgentEntry:
    """One validated entry of the ``agents`` list of a scenario payload.

    Attributes:
        index: 0-based position in the payload.
        mode: ``MODE_NEW``, ``MODE_LIBRARY`` or ``MODE_SCENARIO``.
        name: Name the scenario agent will have.
        data: Agent input for the ``new`` and ``scenario`` modes.
        library_agent: Selected library agent (``library`` mode).
        scenario_agent: Kept agent of the edited scenario (``scenario`` mode).
    """

    index: int
    mode: str
    name: str
    data: AgentInput | None = None
    library_agent: Agent | None = None
    scenario_agent: Agent | None = None


def _coerce_id(raw_value: Any) -> int | None:
    """Return ``raw_value`` as a positive integer id, or None when it is not one.

    Args:
        raw_value: int or digit string.

    Returns:
        The id or None.
    """
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value if raw_value > 0 else None
    if isinstance(raw_value, str) and raw_value.strip().isdigit():
        return int(raw_value.strip()) or None
    return None


def _parse_scenario_agents(
    agents_data: Any, market: Market | None
) -> list[ScenarioAgentEntry]:
    """Validate every agent of a scenario payload before any write.

    Entry modes (``mode`` key, default ``new`` for payloads of older pages):

    * ``new``: name, type, budget, algorithm, fail-stack and plants of a new
      agent. It is added to the library and copied into the scenario, so
      its name must be free in the library (case-insensitive).
    * ``library``: ``library_agent_id`` of a library agent to copy. The same
      library agent may appear only once (OQ-14).
    * ``scenario`` (update only): ``agent_id`` of an agent of ``market`` plus
      its edited fields. The agent is kept and updated in place, the
      library is not changed.

    Args:
        agents_data: Value of the ``agents`` key of the JSON payload.
        market: Market of the edited scenario, None when creating.

    Returns:
        One entry per payload agent, in payload order.

    Raises:
        ValidationError: Mapping from error key to messages. Content errors
            use the field name (``type``, ``plants``, ...), identity errors
            ``agents[<i>].name``, ``agents[<i>].library_agent_id``,
            ``agents[<i>].agent_id`` or ``agents[<i>].mode``. Every message
            names the agent it belongs to.
    """
    if not isinstance(agents_data, list):
        raise ValidationError({"agents": ["Agents must be a list."]})
    errors: dict[str, list[str]] = {}
    entries: list[ScenarioAgentEntry] = []
    market_agents = (
        {agent.id: agent for agent in market.agent_set.all()} if market else {}
    )
    seen_library_ids: set[int] = set()
    seen_agent_ids: set[int] = set()

    def add(key: str, message: str) -> None:
        errors.setdefault(key, []).append(message)

    for index, agent_data in enumerate(agents_data):
        label = f"Agent {index + 1}"
        if not isinstance(agent_data, dict):
            add("agents", f"{label}: malformed entry.")
            continue
        mode = agent_data.get("mode") or MODE_NEW
        if mode not in SCENARIO_AGENT_MODES or (mode == MODE_SCENARIO and not market):
            allowed = SCENARIO_AGENT_MODES if market else (MODE_NEW, MODE_LIBRARY)
            add(
                _entry_key(index, "mode"),
                f"{label}: unknown mode {mode!r}, expected one of {', '.join(allowed)}.",
            )
            continue

        if mode == MODE_LIBRARY:
            library_id = _coerce_id(agent_data.get("library_agent_id"))
            library_agent = (
                Agent.objects.filter(
                    id=library_id, proxy=True, market__isnull=True
                ).first()
                if library_id is not None
                else None
            )
            if library_agent is None:
                add(
                    _entry_key(index, "library_agent_id"),
                    f"{label}: library agent "
                    f"{agent_data.get('library_agent_id')!r} does not exist.",
                )
            elif library_agent.id in seen_library_ids:
                add(
                    _entry_key(index, "library_agent_id"),
                    f"{label} ({library_agent.name}): this library agent is "
                    "already part of the scenario. A library agent can be "
                    "added to a scenario only once.",
                )
            else:
                seen_library_ids.add(library_agent.id)
                entries.append(
                    ScenarioAgentEntry(
                        index, mode, library_agent.name, library_agent=library_agent
                    )
                )
            continue

        scenario_agent = None
        if mode == MODE_SCENARIO:
            agent_id = _coerce_id(agent_data.get("agent_id"))
            scenario_agent = market_agents.get(agent_id) if agent_id else None
            if scenario_agent is None:
                add(
                    _entry_key(index, "agent_id"),
                    f"{label}: agent {agent_data.get('agent_id')!r} is not part "
                    "of this scenario.",
                )
                continue
            if scenario_agent.id in seen_agent_ids:
                add(
                    _entry_key(index, "agent_id"),
                    f"{label}: agent {scenario_agent.id} appears twice.",
                )
                continue
            seen_agent_ids.add(scenario_agent.id)

        prefix = f"{label} ({agent_data.get('name', '')}): "
        entry_ok = True
        name = ""
        try:
            name = _validate_entry_name(agent_data.get("name"))
        except ValidationError as exc:
            entry_ok = False
            add(_entry_key(index, "name"), prefix + exc.messages[0])
        if (
            entry_ok
            and mode == MODE_NEW
            and not library_service.check_name_available(name)
        ):
            entry_ok = False
            add(_entry_key(index, "name"), prefix + _library_name_taken_message(name))
        try:
            fields, plants = _parse_agent_fields(agent_data, prefix)
        except ValidationError as exc:
            entry_ok = False
            _add_errors(errors, exc, "agents", "")
            continue
        if not entry_ok:
            continue
        data = AgentInput(name=name, plants=plants, **fields)
        entries.append(
            ScenarioAgentEntry(
                index, mode, name, data=data, scenario_agent=scenario_agent
            )
        )

    # Names must be unique inside the scenario (case-sensitive, per-market
    # constraint) and new agents must not collide with each other in the
    # library (case-insensitive). The database constraints stay the final guard.
    used_names: set[str] = set()
    new_names: set[str] = set()
    for entry in entries:
        message = None
        if entry.name in used_names:
            message = (
                f"the name '{entry.name}' is used by another agent of this scenario."
            )
        elif entry.mode == MODE_NEW and entry.name.lower() in new_names:
            message = (
                f"the name '{entry.name}' is used by another new agent "
                "(library names are compared case-insensitively)."
            )
        if message is not None:
            add(
                _entry_key(entry.index, "name"),
                f"Agent {entry.index + 1} ({entry.name}): {message}",
            )
        used_names.add(entry.name)
        if entry.mode == MODE_NEW:
            new_names.add(entry.name.lower())
    # OQ-14 also covers a library agent whose copy is kept in the scenario.
    kept_sources = {
        entry.scenario_agent.source_id
        for entry in entries
        if entry.scenario_agent is not None and entry.scenario_agent.source_id
    }
    for entry in entries:
        if entry.library_agent is not None and entry.library_agent.id in kept_sources:
            add(
                _entry_key(entry.index, "library_agent_id"),
                f"Agent {entry.index + 1} ({entry.name}): this library agent is "
                "already part of the scenario. A library agent can be added to "
                "a scenario only once.",
            )
    if errors:
        raise ValidationError(errors)
    return entries


def _entry_error(entry: ScenarioAgentEntry, exc: ValidationError) -> ValidationError:
    """Re-key a service error raised while writing one payload agent.

    Args:
        entry: The entry being written.
        exc: Error of the library service or of model validation.

    Returns:
        ``ValidationError`` keyed by ``agents[<i>].<field>`` with messages
        prefixed by the agent label.
    """
    prefix = f"Agent {entry.index + 1} ({entry.name}): "
    if hasattr(exc, "error_dict"):
        items = exc.message_dict.items()
    else:
        items = [("name", exc.messages)]
    return ValidationError(
        {
            _entry_key(entry.index, key): [prefix + message for message in messages]
            for key, messages in items
        }
    )


def _update_scenario_agent(agent: Agent, data: AgentInput) -> None:
    """Update a kept scenario agent in place and replace its plants.

    The agent's ``source``, ``algorithm_params`` and portfolio row are kept.
    The library agent it was copied from is not changed.

    Args:
        agent: Agent of the edited scenario.
        data: Validated values from the payload.

    Raises:
        ValidationError: Field validation failed or the name is already used
            in the scenario (database constraint).
    """
    agent.name = data.name
    agent.budget = data.budget
    agent.type = data.type
    agent.algorithm_type = data.algorithm_type
    agent.failstack_enabled = data.failstack_enabled
    agent.failstack_limit = data.failstack_limit
    agent.failstack_delta = data.failstack_delta
    agent.failstack_min_margin = data.failstack_min_margin
    agent.clean_fields(exclude={"market", "source", "state"})
    try:
        with transaction.atomic():
            agent.save()
    except IntegrityError:
        raise ValidationError(
            {
                "name": [
                    f"An agent named '{agent.name}' already exists in this scenario."
                ]
            }
        ) from None
    resources = Resource.objects.in_bulk({plant.resource_id for plant in data.plants})
    portfolio, _ = Portfolio.objects.get_or_create(
        agent=agent, defaults={"proxy": agent.proxy}
    )
    portfolio.plant_set.all().delete()
    Plant.objects.bulk_create(
        [
            Plant(
                portfolio=portfolio,
                resource=resources[plant.resource_id],
                capacity=plant.capacity,
                proxy=agent.proxy,
            )
            for plant in data.plants
        ]
    )


def _write_scenario_agents(market: Market, entries: list[ScenarioAgentEntry]) -> None:
    """Replace the agents of a proxy market with the validated payload entries.

    Must run inside ``transaction.atomic``. Agents of ``market`` that are not
    kept by a ``scenario`` entry are deleted first, kept agents are updated,
    then ``new`` entries go through ``library_service.add_new_agent_to_scenario``
    and ``library`` entries through ``library_service.copy_agent``.

    Args:
        market: Market of the proxy scenario.
        entries: Output of ``_parse_scenario_agents``.

    Raises:
        ValidationError: Keyed by ``agents[<i>].<field>`` when the library
            service or the database rejects an entry (for example a name
            taken between validation and write).
    """
    kept_ids = [entry.scenario_agent.id for entry in entries if entry.scenario_agent]
    market.agent_set.exclude(id__in=kept_ids).delete()
    for entry in entries:
        if entry.mode != MODE_SCENARIO:
            continue
        try:
            _update_scenario_agent(entry.scenario_agent, entry.data)
        except ValidationError as exc:
            raise _entry_error(entry, exc) from exc
    for entry in entries:
        try:
            if entry.mode == MODE_NEW:
                library_service.add_new_agent_to_scenario(entry.data, market)
            elif entry.mode == MODE_LIBRARY:
                library_service.copy_agent(entry.library_agent, market, proxy=True)
        except LibraryValidationError as exc:
            raise _entry_error(entry, exc) from exc


def _check_learning_start(simulation: Simulation) -> None:
    """Reject a scenario whose learning start no agent configuration allows (PLAN 2.14).

    Args:
        simulation: The scenario after its market and agents were written.

    Raises:
        ValidationError: ``{"learning_starts": [message]}``.
    """
    from .services.simulation.context import (
        SimulationValidationError,
        scenario_learning_start,
    )

    try:
        scenario_learning_start(simulation)
    except SimulationValidationError as exc:
        raise ValidationError({"learning_starts": [str(exc)]}) from exc


def _entry_learning_agent(
    entry: ScenarioAgentEntry,
) -> tuple[str, str, Mapping[str, Any]]:
    """Return the learning-start description of one validated payload agent.

    The parameters are those the saved agent will have: the library agent's
    for a ``library`` entry, the kept agent's for a ``scenario`` entry (they
    are not changed by an edit) and none for a ``new`` entry.

    Args:
        entry: Output of ``_parse_scenario_agents``.

    Returns:
        ``(label, algorithm_key, params)`` for ``resolve_learning_start``.
    """
    if entry.library_agent is not None:
        key = str(entry.library_agent.algorithm_type)
        params: Mapping[str, Any] = entry.library_agent.algorithm_params or {}
    else:
        key = str(entry.data.algorithm_type)
        if entry.scenario_agent is not None:
            params = entry.scenario_agent.algorithm_params or {}
        else:
            params = entry.data.algorithm_params or {}
    return f"{entry.name!r} ({key})", key, params


@require_POST
def learning_start_preview(request: HttpRequest) -> JsonResponse:
    """Return the common learning start of unsaved wizard settings (PLAN 2.14).

    Nothing is written. The body is a JSON object with ``learning_starts``
    (empty or null for automatic), ``observation_lags``, ``agents`` (entries
    of the scenario payload, see ``_parse_scenario_agents``) and, on the
    update page, ``simulation_id`` of the edited scenario (needed by
    ``scenario`` entries). The values are validated by the parsers of the
    scenario save, and the range check is the one of the save.

    Args:
        request: POST request with a JSON body.

    Returns:
        ``{"automatic", "value", "first_period", "minimum", "limit"}``.
        ``value``, ``first_period``, ``minimum`` and ``limit`` are null
        without learning agents (``limit`` also when no agent has a replay
        buffer). HTTP 400 ``{"errors": {...}}`` for an invalid body, an
        invalid agent entry or a learning start out of range
        (key ``learning_starts``).
    """
    from .services.simulation.context import (
        SimulationValidationError,
        first_learning_period,
        resolve_learning_start,
    )

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)

    errors: dict[str, list[str]] = {}
    learning_starts = _parse_learning_starts(data, None, errors)
    observation_lags = default_observation_lags()
    if "observation_lags" in data:
        lags = _coerce_observation_lags(data["observation_lags"])
        if lags is None:
            errors["observation_lags"] = [_observation_lags_message()]
        else:
            observation_lags = lags
    market = None
    if data.get("simulation_id") not in (None, ""):
        scenario = Simulation.objects.filter(
            id=_coerce_id(data["simulation_id"]), proxy=True
        ).first()
        if scenario is None:
            errors["simulation_id"] = [
                f"Scenario {data['simulation_id']!r} does not exist."
            ]
        else:
            market = scenario.market
    if errors:
        return JsonResponse({"errors": errors}, status=400)

    try:
        entries = _parse_scenario_agents(data.get("agents", []), market)
    except ValidationError as exc:
        return _validation_error_response(exc)
    try:
        resolved = resolve_learning_start(
            learning_starts, [_entry_learning_agent(entry) for entry in entries]
        )
    except SimulationValidationError as exc:
        return JsonResponse({"errors": {"learning_starts": [str(exc)]}}, status=400)
    if resolved is None:
        return JsonResponse(
            {
                "automatic": learning_starts is None,
                "value": None,
                "first_period": None,
                "minimum": None,
                "limit": None,
            }
        )
    return JsonResponse(
        {
            "automatic": resolved.automatic,
            "value": resolved.value,
            "first_period": first_learning_period(observation_lags, resolved.value),
            "minimum": resolved.minimum,
            "limit": resolved.limit,
        }
    )


def _save_scenario(
    write: Callable[[], None], entries: list[ScenarioAgentEntry]
) -> JsonResponse | None:
    """Run a scenario write in one transaction and map rejections to HTTP 400.

    Args:
        write: Callable without arguments that writes the simulation, the
            market and (through ``_write_scenario_agents``) the agents.
        entries: The validated agent entries (used for the error message of
            an unexpected constraint violation).

    Returns:
        None on success, else the HTTP 400 ``JsonResponse``.
    """
    try:
        with transaction.atomic():
            write()
    except ValidationError as exc:
        return _validation_error_response(exc)
    except IntegrityError as exc:
        logger.warning("Scenario write rejected by a database constraint: %s", exc)
        return JsonResponse(
            {
                "errors": {
                    "agents": [
                        "The agents could not be saved because a database "
                        "constraint rejected them (for example two agents with "
                        f"the same name among {len(entries)} agents)."
                    ]
                }
            },
            status=400,
        )
    return None


@dataclass(frozen=True)
class RunSettings:
    """Validated run settings of a scenario payload (OQ-16).

    Attributes:
        seed: Seed copied to every run of the scenario.
        evaluation_periods: Number of final periods without exploration and
            learning (0 or less than ``periodNumber``).
        observation_lags: One of ``ALLOWED_OBSERVATION_LAGS``.
        demand_profile: Validated synthetic demand profile
            (``DemandProfile.to_dict``), or None for the demand file
            (PLAN 2.8).
        demand_profile_source_id: Id of the library profile the stored copy
            came from (PLAN 2.9), None for the file or a generated profile.
        learning_starts: Common learning start of the learning agents in
            stored transitions (PLAN 2.14), None for the automatic value.
    """

    seed: int
    evaluation_periods: int
    observation_lags: list[int]
    demand_profile: dict[str, Any] | None = None
    demand_profile_source_id: int | None = None
    learning_starts: int | None = None

    def apply(self, simulation: Simulation) -> None:
        """Set the settings on a simulation (the caller saves it).

        Args:
            simulation: The simulation to update in place.
        """
        simulation.seed = self.seed
        simulation.evaluation_periods = self.evaluation_periods
        simulation.learning_starts = self.learning_starts
        simulation.observation_lags = list(self.observation_lags)
        simulation.demand_profile = (
            None if self.demand_profile is None else dict(self.demand_profile)
        )
        simulation.demand_profile_source_id = (
            None if self.demand_profile is None else self.demand_profile_source_id
        )


def _coerce_payload_int(raw_value: Any) -> int | None:
    """Convert a JSON integer or decimal digit string to ``int``.

    Args:
        raw_value: Payload value.

    Returns:
        The integer, or None when ``raw_value`` is not an integer (booleans,
        floats with a fraction, empty or non-numeric strings, other types).
    """
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value) if raw_value.is_integer() else None
    if isinstance(raw_value, str) and re.fullmatch(r"\s*-?\d+\s*", raw_value):
        return int(raw_value)
    return None


def _coerce_observation_lags(raw_value: Any) -> list[int] | None:
    """Convert a payload lag set (list or comma separated string) to a list.

    Args:
        raw_value: Payload value, for example ``[1, 24]`` or ``"1,24,168"``.

    Returns:
        The lag list when it equals one of ``ALLOWED_OBSERVATION_LAGS``, else None.
    """
    if isinstance(raw_value, str):
        items: list[Any] = [part for part in raw_value.split(",")]
    elif isinstance(raw_value, list):
        items = raw_value
    else:
        return None
    lags = [_coerce_payload_int(item) for item in items]
    if any(lag is None for lag in lags):
        return None
    for allowed in ALLOWED_OBSERVATION_LAGS:
        if lags == allowed:
            return list(allowed)
    return None


def _observation_lags_message() -> str:
    """Return the error message of an observation lag set that is not allowed.

    Returns:
        The message listing ``ALLOWED_OBSERVATION_LAGS``.
    """
    return (
        "Observation lags must be one of: "
        + " or ".join(", ".join(map(str, lags)) for lags in ALLOWED_OBSERVATION_LAGS)
        + "."
    )


DEMAND_PROFILE_KEY: Final[str] = PAYLOAD_KEY
DEMAND_SOURCE_KEY: Final[str] = "demand_profile_source"


def _parse_run_settings(
    data: Mapping[str, Any], period_number: int, current: Simulation | None
) -> RunSettings:
    """Validate ``seed``, ``evaluation_periods``, ``observation_lags`` (OQ-16),
    ``demand_profile`` (PLAN 2.8) and ``learning_starts`` (PLAN 2.14).

    A missing key takes the value of ``current`` (update page) or the model
    default (create page). The rules mirror the check constraints of
    ``Simulation``.

    Args:
        data: Decoded scenario payload.
        period_number: Validated ``periodNumber`` of the payload.
        current: The edited scenario, or None on create.

    Returns:
        The validated settings.

    Raises:
        ValidationError: With field keys ``seed``, ``evaluation_periods``,
            ``learning_starts``, ``observation_lags`` and
            ``demand_profile[.<field>]``.
    """
    errors: dict[str, list[str]] = {}

    if "seed" in data:
        seed = _coerce_payload_int(data["seed"])
        if seed is None or not 0 <= seed < SEED_UPPER_BOUND:
            errors["seed"] = [
                f"Seed must be an integer from 0 to {SEED_UPPER_BOUND - 1}."
            ]
            seed = DEFAULT_SEED
    else:
        seed = current.seed if current is not None else DEFAULT_SEED

    if "evaluation_periods" in data:
        evaluation_periods = _coerce_payload_int(data["evaluation_periods"])
    else:
        evaluation_periods = current.evaluation_periods if current is not None else 0
    if evaluation_periods is None or evaluation_periods < 0:
        errors["evaluation_periods"] = [
            "Evaluation periods must be a non-negative integer."
        ]
        evaluation_periods = 0
    elif evaluation_periods > 0 and evaluation_periods >= period_number:
        errors["evaluation_periods"] = [
            "Evaluation periods must be less than the number of periods "
            f"({period_number})."
        ]

    learning_starts = _parse_learning_starts(data, current, errors)

    if "observation_lags" in data:
        observation_lags = _coerce_observation_lags(data["observation_lags"])
        if observation_lags is None:
            errors["observation_lags"] = [_observation_lags_message()]
            observation_lags = default_observation_lags()
    elif current is not None:
        observation_lags = list(current.observation_lags)
    else:
        observation_lags = default_observation_lags()

    demand_profile, source_id = _parse_demand(data, current, errors)

    if errors:
        raise ValidationError(errors)
    return RunSettings(
        seed,
        evaluation_periods,
        observation_lags,
        demand_profile,
        source_id,
        learning_starts,
    )


def _parse_learning_starts(
    data: Mapping[str, Any], current: Simulation | None, errors: dict[str, list[str]]
) -> int | None:
    """Validate the optional common learning start (PLAN 2.14).

    An empty string or null means the automatic value (stored as NULL). A
    missing key keeps the value of ``current`` (update page) or takes the
    automatic value (create page). The rule mirrors the check constraint of
    ``Simulation.learning_starts``.

    Args:
        data: Decoded scenario payload.
        current: The edited scenario, or None on create.
        errors: Error mapping, updated in place (key ``learning_starts``).

    Returns:
        The learning start in stored transitions, or None for automatic.
    """
    if "learning_starts" not in data:
        return current.learning_starts if current is not None else None
    raw_value = data["learning_starts"]
    if raw_value is None or (isinstance(raw_value, str) and not raw_value.strip()):
        return None
    learning_starts = _coerce_payload_int(raw_value)
    if learning_starts is None or learning_starts < 1:
        errors["learning_starts"] = [
            "Learning start must be empty (automatic) or a positive integer."
        ]
        return None
    return learning_starts


def _parse_demand(
    data: Mapping[str, Any], current: Simulation | None, errors: dict[str, list[str]]
) -> tuple[dict[str, Any] | None, int | None]:
    """Resolve the demand source of a scenario payload (PLAN 2.8, 2.9).

    * ``demand_profile_source`` set to a library profile id: the scenario
      stores a copy of that library profile and links to it. The library
      profile is authoritative, a posted ``demand_profile`` is ignored.
    * Otherwise ``demand_profile`` (null = demand file, a mapping = generated
      profile) is validated and the link is cleared.
    * Neither key (update page): the stored profile and link are kept, a
      ``demand_profile_source: null`` alone only clears the link.

    Args:
        data: Decoded scenario payload.
        current: The edited scenario, or None on create.
        errors: Error mapping, updated in place (keys
            ``demand_profile[.<field>]`` and ``demand_profile_source``).

    Returns:
        ``(demand_profile, demand_profile_source_id)``.
    """
    raw_source = data.get(DEMAND_SOURCE_KEY)
    if raw_source not in (None, ""):
        source_id = _coerce_payload_int(raw_source)
        template = (
            DemandProfileTemplate.objects.filter(id=source_id).first()
            if source_id is not None
            else None
        )
        if template is None:
            errors[DEMAND_SOURCE_KEY] = [
                "The selected demand profile is not in the library any more."
            ]
            return None, None
        try:
            return normalize_profile(template.profile), template.id
        except ValidationError as exc:
            _add_errors(errors, exc, DEMAND_PROFILE_KEY, "")
            return None, None
    if DEMAND_PROFILE_KEY in data:
        try:
            return normalize_profile(data[DEMAND_PROFILE_KEY]), None
        except ValidationError as exc:
            _add_errors(errors, exc, DEMAND_PROFILE_KEY, "")
            return None, None
    if current is None or current.demand_profile is None:
        return None, None
    keep_link = DEMAND_SOURCE_KEY not in data
    return (
        dict(current.demand_profile),
        current.demand_profile_source_id if keep_link else None,
    )


def _parse_period_number(data: Mapping[str, Any]) -> int:
    """Validate ``periodNumber`` of a scenario payload.

    Args:
        data: Decoded scenario payload.

    Returns:
        The positive number of periods.

    Raises:
        ValidationError: With field key ``periodNumber``.
    """
    period_number = _coerce_payload_int(data.get("periodNumber"))
    if period_number is None or period_number < 1:
        raise ValidationError(
            {"periodNumber": ["The number of periods must be a positive integer."]}
        )
    return period_number


def _parse_scenario_settings(
    data: Mapping[str, Any], current: Simulation | None
) -> tuple[int, RunSettings]:
    """Validate ``periodNumber`` and the run settings of a scenario payload.

    Args:
        data: Decoded scenario payload.
        current: The edited scenario, or None on create.

    Returns:
        ``(period_number, run_settings)``.

    Raises:
        ValidationError: With the field keys of ``_parse_period_number`` and
            ``_parse_run_settings``.
    """
    period_number = _parse_period_number(data)
    return period_number, _parse_run_settings(data, period_number, current)


# Upper limit of Simulation.name (varchar column).
SIMULATION_NAME_MAX_LENGTH: Final[int] = (
    Simulation._meta.get_field("name").max_length or 20
)


@dataclass(frozen=True)
class ScenarioHeader:
    """Validated scenario fields outside the agents and run settings.

    Attributes:
        name: Scenario name (stripped, at most ``SIMULATION_NAME_MAX_LENGTH``).
        mode: Simulation mode.
        period_type: Period type.
        strategy: Market pricing strategy (settlement rule).
        lower_bound: Lower bid bound in USD/MWh.
        upper_bound: Upper bid bound in USD/MWh, greater than ``lower_bound``.
        clearing_algorithm: Key of a registered clearing algorithm (PLAN 2.10).
        clearing_params: Per-scenario overrides of the clearing algorithm
            parameters, typed values by field name (PLAN 2.12).
    """

    name: str
    mode: enums.SimulationMode
    period_type: enums.PeriodType
    strategy: enums.MarketStrategy
    lower_bound: int
    upper_bound: int
    clearing_algorithm: str = DEFAULT_CLEARING_ALGORITHM
    clearing_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioPayload:
    """A completely validated scenario payload (nothing written yet).

    Attributes:
        header: Name, mode, period type and market settings.
        period_number: Number of periods.
        run_settings: Seed, evaluation periods and observation lags.
        entries: Validated agent entries in payload order.
    """

    header: ScenarioHeader
    period_number: int
    run_settings: RunSettings
    entries: list[ScenarioAgentEntry]


def _coerce_payload_enum(raw_value: Any, enum_cls: Any, field: str, label: str) -> Any:
    """Convert a JSON integer or digit string to a member of ``enum_cls``.

    Args:
        raw_value: Payload value.
        enum_cls: Enum class with integer values.
        field: Error key.
        label: Human-readable field name for the message.

    Returns:
        The enum member.

    Raises:
        ValidationError: With key ``field`` for a missing or unknown value.
    """
    value = _coerce_payload_int(raw_value)
    try:
        return enum_cls(value)
    except ValueError:
        choices = ", ".join(f"{member.value} ({member.name})" for member in enum_cls)
        raise ValidationError({field: [f"{label} must be one of {choices}."]}) from None


def _parse_scenario_header(data: Mapping[str, Any]) -> ScenarioHeader:
    """Validate name, mode, period type, strategy and bid bounds of a payload.

    Args:
        data: Decoded scenario payload.

    Returns:
        The validated header.

    Raises:
        ValidationError: With field keys ``simulation_name``,
            ``simulationMode``, ``periodType``, ``strategy``, ``lowerBound``,
            ``upperBound``, ``clearing_algorithm`` (optional, MERIT_ORDER
            when missing, must be a registered clearing algorithm key) and
            ``clearing_params`` (optional overrides of the clearing
            algorithm parameters, validated with its config class together
            with ``ClearingDefault``, messages prefixed by the field name).
    """
    errors: dict[str, list[str]] = {}
    values: dict[str, Any] = {}

    raw_name = data.get("simulation_name")
    name = raw_name.strip() if isinstance(raw_name, str) else ""
    if not name:
        errors["simulation_name"] = ["Scenario name is required."]
    elif len(name) > SIMULATION_NAME_MAX_LENGTH:
        errors["simulation_name"] = [
            f"Scenario name must have at most {SIMULATION_NAME_MAX_LENGTH} "
            f"characters ({len(name)} given)."
        ]
    for key, field, enum_cls, label in (
        ("mode", "simulationMode", enums.SimulationMode, "Simulation mode"),
        ("period_type", "periodType", enums.PeriodType, "Period type"),
        ("strategy", "strategy", enums.MarketStrategy, "Pricing strategy"),
    ):
        try:
            values[key] = _coerce_payload_enum(data.get(field), enum_cls, field, label)
        except ValidationError as exc:
            _add_errors(errors, exc, field, "")
    clearing = data.get("clearing_algorithm")
    if clearing is None or clearing == "":
        clearing = DEFAULT_CLEARING_ALGORITHM
    if not is_known_clearing(clearing):
        known = ", ".join(info.key for info in available_clearing_algorithms())
        errors["clearing_algorithm"] = [
            f"Clearing algorithm {clearing!r} is not available. Known clearing "
            f"algorithms: {known}."
        ]
    clearing_params: dict[str, Any] = {}
    if "clearing_algorithm" not in errors:
        clearing_params, clearing_errors = (
            algorithm_catalog.validate_scenario_clearing_params(
                str(clearing), data.get("clearing_params")
            )
        )
        if clearing_errors:
            errors["clearing_params"] = clearing_errors
    lower = _coerce_payload_int(data.get("lowerBound"))
    upper = _coerce_payload_int(data.get("upperBound"))
    if lower is None:
        errors["lowerBound"] = ["Lower bid bound must be an integer (USD/MWh)."]
    if upper is None:
        errors["upperBound"] = ["Upper bid bound must be an integer (USD/MWh)."]
    elif lower is not None and upper <= lower:
        errors["upperBound"] = [
            "Upper bid bound must be greater than the lower bid bound."
        ]
    if errors:
        raise ValidationError(errors)
    return ScenarioHeader(
        name=name,
        mode=values["mode"],
        period_type=values["period_type"],
        strategy=values["strategy"],
        lower_bound=lower,
        upper_bound=upper,
        clearing_algorithm=str(clearing),
        clearing_params=clearing_params,
    )


def _validate_scenario_payload(
    data: Mapping[str, Any], current: Simulation | None, market: Market | None
) -> ScenarioPayload | JsonResponse:
    """Validate the header, the settings and the agents of a scenario payload.

    Errors of all parts are merged into one HTTP 400 response, so the page
    can show every field error at once. Nothing is written.

    Args:
        data: Decoded scenario payload.
        current: The edited scenario, or None on create.
        market: Market of the edited scenario, or None on create.

    Returns:
        The validated payload, or the HTTP 400 ``JsonResponse`` with
        ``{"errors": {field: [messages]}}``.
    """
    errors: dict[str, list[str]] = {}
    header: ScenarioHeader | None = None
    settings: tuple[int, RunSettings] | None = None
    entries: list[ScenarioAgentEntry] = []
    try:
        header = _parse_scenario_header(data)
    except ValidationError as exc:
        _add_errors(errors, exc, "settings", "")
    try:
        settings = _parse_scenario_settings(data, current)
    except ValidationError as exc:
        _add_errors(errors, exc, "settings", "")
    try:
        entries = _parse_scenario_agents(data.get("agents", []), market=market)
    except ValidationError as exc:
        _add_errors(errors, exc, "plants", "")
    if errors or settings is None or header is None:
        return JsonResponse({"errors": errors}, status=400)
    return ScenarioPayload(header, settings[0], settings[1], entries)


def _plant_form_errors(plant_forms: list[PlantForm]) -> dict[str, list[str]]:
    """Collect the errors of invalid plant forms under the ``plants`` key.

    Args:
        plant_forms: One bound ``PlantForm`` per submitted plant row.

    Returns:
        ``{'plants': [...]}`` or an empty mapping when every form is valid.
    """
    messages: list[str] = []
    for index, form in enumerate(plant_forms, start=1):
        if not form.is_valid():
            for field, field_errors in form.errors.items():
                messages.extend(
                    f"Plant {index} {field}: {error}" for error in field_errors
                )
    return {"plants": messages} if messages else {}


def _agent_form_errors(form: AgentForm) -> dict[str, list[str]]:
    """Return the errors of an invalid AgentForm with rule errors under ``plants``.

    Args:
        form: A bound, validated ``AgentForm``.

    Returns:
        Mapping from field name to messages.
    """
    errors = {key: list(messages) for key, messages in form.errors.items()}
    if "__all__" in errors:
        errors.setdefault("plants", []).extend(errors.pop("__all__"))
    return errors


def _wants_json(request: HttpRequest) -> bool:
    """Return whether the client asked for a JSON response (AJAX form submit).

    Args:
        request: The request.

    Returns:
        True when ``Accept`` contains ``application/json`` or the request is
        marked with ``X-Requested-With: XMLHttpRequest``.
    """
    return (
        "application/json" in request.headers.get("Accept", "")
        or request.headers.get("X-Requested-With") == "XMLHttpRequest"
    )


def _library_agent_payload(agent: Agent) -> dict[str, Any]:
    """Serialise a library agent for the library dialog and the agent form.

    Args:
        agent: Library agent (portfolio and plants may be prefetched).

    Returns:
        ``{id, name, type, type_name, budget, algorithm_type,
        algorithm_name, algorithm_available, failstack: {enabled, active,
        limit, delta, min_margin}, plants: [{resource_id, resource_name,
        energy_type, capacity}]}``. ``algorithm_type`` is the algorithm key,
        ``algorithm_name`` its registry label. Decimals are strings.
    """
    agent_type = enums.AgentType.get(getattr(agent.type, "value", agent.type))
    algorithm = str(agent.algorithm_type)
    try:
        plant_rows = list(agent.portfolio.plant_set.all())
    except Portfolio.DoesNotExist:
        plant_rows = []
    plant_rows.sort(key=lambda plant: plant.id)
    return {
        "id": agent.id,
        "name": agent.name,
        "type": agent_type.value if agent_type is not None else None,
        "type_name": agent_type.name if agent_type is not None else "",
        "budget": str(agent.budget),
        "algorithm_type": algorithm,
        "algorithm_name": algorithm_registry.label_for(algorithm),
        "algorithm_available": algorithm_registry.is_known(algorithm),
        "failstack": {
            "enabled": bool(agent.failstack_enabled),
            "active": bool(agent.failstack_enabled)
            and not algorithm_registry.is_baseline(algorithm),
            "limit": agent.failstack_limit,
            "delta": str(agent.failstack_delta),
            "min_margin": str(agent.failstack_min_margin),
        },
        "plants": [
            {
                "resource_id": plant.resource_id,
                "resource_name": plant.resource.name,
                "energy_type": getattr(
                    plant.resource.energyType, "value", plant.resource.energyType
                ),
                "capacity": plant.capacity,
            }
            for plant in plant_rows
        ],
    }


def _library_agents():
    """Return the queryset of library agents with plants prefetched, by name.

    Returns:
        ``QuerySet[Agent]`` of agents with ``proxy=True`` and ``market=NULL``.
    """
    return (
        Agent.objects.filter(proxy=True, market__isnull=True)
        .select_related("portfolio")
        .prefetch_related("portfolio__plant_set__resource")
        .order_by(Lower("name"), "id")
    )


# SYSTEM
@dataclass(frozen=True)
class RunOptions:
    """Options of a POST to ``run_simulation``.

    Attributes:
        seed: Seed of this run, None for the scenario seed (PLAN 2.5).
        warm_start_from: Id of the finished run whose saved policies
            initialise the run, None for fresh policies (finding A9).
        freeze_policies: Evaluate the loaded policies without learning.
        dry_run: Only validate the options and return the warnings, create
            no run.
    """

    seed: int | None = None
    warm_start_from: int | None = None
    freeze_policies: bool = False
    dry_run: bool = False


def _run_request_data(request: HttpRequest) -> Mapping[str, Any]:
    """Return the JSON body or the form fields of a POST to ``run_simulation``.

    Args:
        request: POST request.

    Returns:
        The decoded JSON object, or ``request.POST``.

    Raises:
        ValidationError: With key ``seed`` for a malformed JSON body.
    """
    if request.content_type != "application/json":
        return request.POST
    try:
        data = json.loads(request.body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationError({"seed": ["Invalid JSON body."]}) from exc
    if not isinstance(data, dict):
        raise ValidationError({"seed": ["Invalid JSON body."]})
    return data


def _coerce_payload_flag(raw: Any, field: str) -> bool:
    """Convert a JSON boolean or a form checkbox value to ``bool``.

    Args:
        raw: Payload value (None when missing).
        field: Field name for the error.

    Returns:
        The flag. Missing or empty values are False.

    Raises:
        ValidationError: With key ``field`` for any other value.
    """
    if raw is None or isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, str) and raw.strip().lower() in {"", "0", "false", "off"}:
        return False
    if isinstance(raw, str) and raw.strip().lower() in {"1", "true", "on"}:
        return True
    raise ValidationError({field: [f"{field} must be true or false."]})


def _parse_run_options(request: HttpRequest) -> RunOptions:
    """Read the options of a POST to ``run_simulation``.

    Args:
        request: POST request with a JSON body or form fields ``seed``,
            ``warm_start_from``, ``freeze_policies`` and ``dry_run``.

    Returns:
        The options.

    Raises:
        ValidationError: Keyed by field for a malformed body, a seed outside
            ``[0, SEED_UPPER_BOUND)``, a non-integer ``warm_start_from``, a
            non-boolean flag, or ``freeze_policies`` without
            ``warm_start_from``.
    """
    data = _run_request_data(request)
    raw = data.get("seed")
    seed: int | None = None
    if not (raw is None or (isinstance(raw, str) and not raw.strip())):
        seed = _coerce_payload_int(raw)
        if seed is None or not 0 <= seed < SEED_UPPER_BOUND:
            raise ValidationError(
                {"seed": [f"Seed must be an integer from 0 to {SEED_UPPER_BOUND - 1}."]}
            )
    raw_source = data.get("warm_start_from")
    source: int | None = None
    if not (
        raw_source is None or (isinstance(raw_source, str) and not raw_source.strip())
    ):
        source = _coerce_payload_int(raw_source)
        if source is None or source < 1:
            raise ValidationError(
                {"warm_start_from": ["Initial policies must name a run id."]}
            )
    freeze = _coerce_payload_flag(data.get("freeze_policies"), "freeze_policies")
    if freeze and source is None:
        raise ValidationError(
            {"freeze_policies": ["Evaluate only requires initial policies from a run."]}
        )
    return RunOptions(
        seed=seed,
        warm_start_from=source,
        freeze_policies=freeze,
        dry_run=_coerce_payload_flag(data.get("dry_run"), "dry_run"),
    )


def run_simulation(request: HttpRequest, simulation_id: int) -> HttpResponse:
    """Clone a proxy scenario into a new run and start it.

    The scenario (simulation, market, agents, portfolios, plants) is copied
    atomically by ``library_service.clone_scenario_for_run``. A run
    (non-proxy simulation) is not cloned again, the request redirects to its
    watch page.

    GET runs with the scenario seed. POST accepts an optional ``seed`` (form
    field or JSON body) that applies to this run only (PLAN 2.5), and
    optionally ``warm_start_from`` (a finished run of this scenario with saved
    policies) and ``freeze_policies`` (finding A9). The warm start is checked
    completely (files, shapes) before the run is created. ``dry_run`` only
    performs that check and returns ``{"warnings": [...]}``. A JSON client
    (``Accept: application/json``) receives ``{"run_id", "watch_url"}``
    instead of a redirect and errors as ``{"errors": {field: [messages]}}``.

    Args:
        request: The request.
        simulation_id: Id of the simulation.

    Returns:
        Redirect (or JSON) to the watch page of the new run, redirect to the
        watch page of ``simulation_id`` for a run, HTTP 400 for an invalid
        seed or warm start (no run is created) or when the scenario cannot be
        cloned, HTTP 500 when the run process cannot be started.
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    wants_json = _wants_json(request)

    def _done(run_id: int) -> HttpResponse:
        if wants_json:
            return JsonResponse(
                {
                    "run_id": run_id,
                    "watch_url": reverse(
                        "watch_simulation", kwargs={"simulation_id": run_id}
                    ),
                }
            )
        return redirect("watch_simulation", simulation_id=run_id)

    def _error(field: str, message: str, status: int) -> HttpResponse:
        if wants_json:
            return JsonResponse({"errors": {field: [message]}}, status=status)
        return HttpResponse(message, status=status)

    options = RunOptions()
    if request.method == "POST":
        try:
            options = _parse_run_options(request)
        except ValidationError as exc:
            return _validation_error_response(exc)
    if not simulation.proxy:
        return _done(simulation.id)
    source: Simulation | None = None
    warnings: list[str] = []
    if options.warm_start_from is not None:
        source = Simulation.objects.filter(pk=options.warm_start_from).first()
        if source is None:
            return _error(
                "warm_start_from", f"Run {options.warm_start_from} does not exist.", 400
            )
        try:
            warnings = checkpoints.check_warm_start(
                simulation, source, freeze=options.freeze_policies
            )
        except checkpoints.WarmStartError as exc:
            return _error("warm_start_from", str(exc), 400)
    try:
        _check_learning_start(simulation)
    except ValidationError as exc:
        return _error("learning_starts", exc.message_dict["learning_starts"][0], 400)
    if options.dry_run:
        if wants_json:
            return JsonResponse({"warnings": warnings})
        return HttpResponse("\n".join(warnings) or "OK")
    try:
        run = library_service.clone_scenario_for_run(
            simulation,
            seed=options.seed,
            warm_start_from=source,
            freeze_policies=options.freeze_policies,
        )
    except (ValidationError, ValueError) as exc:
        logger.warning("Scenario %s could not be cloned: %s", simulation.id, exc)
        return _error("__all__", f"The scenario could not be cloned: {exc}", 400)
    try:
        run_id = StarterService.start(run)
    except Exception:
        logger.exception(
            "Run %s of scenario %s could not be started", run.id, simulation.id
        )
        run.delete()
        return _error("__all__", "The simulation run could not be started.", 500)
    return _done(run_id)


def _user_runs():
    """Return the user's simulation runs without HPO evaluation runs.

    HPO evaluation simulations are linked to an ``HPOTrial`` and are listed on
    the watch page of their study instead (ARCHITECTURE.md 7.2).

    Returns:
        Queryset of ``Simulation`` with ``proxy=False`` and no HPO trial.
    """
    return Simulation.objects.filter(proxy=False, hpo_trial__isnull=True)


def dashboard(request: HttpRequest) -> HttpResponse:
    """Render the dashboard with counters and the latest runs.

    Args:
        request: The request.

    Returns:
        The page. ``num_library_agents`` counts library agents only
        (``proxy=True``, ``market`` NULL), not scenario or run copies.
        ``num_demand_profiles`` counts the demand profile library (PLAN 2.9).
    """
    num_resources = Resource.objects.count()
    num_library_agents = Agent.objects.filter(proxy=True, market__isnull=True).count()
    num_simulations = _user_runs().count()
    num_proxy_simulations = Simulation.objects.filter(proxy=True).count()
    latest_simulations = _user_runs().order_by("-created_at")[:5]
    context = {
        "num_resources": num_resources,
        "num_library_agents": num_library_agents,
        "num_simulations": num_simulations,
        "num_proxy_simulations": num_proxy_simulations,
        "num_demand_profiles": DemandProfileTemplate.objects.count(),
        "latest_simulations": latest_simulations,
    }
    return render(request, "pages/dashboard.html", context)


def blank(request):
    resources = Resource.objects.all()
    return render(request, "pages/blank.html", {"resources": resources})


def watch_simulation(request: HttpRequest, simulation_id: str) -> HttpResponse:
    """Show the watch page of a simulation with its seed and analysis link.

    Args:
        request: The request.
        simulation_id: Id of the simulation.

    Returns:
        The watch page. ``analysis_url`` is set for runs (not for scenario
        templates). The page shows the Analysis button when the polled status
        is FINISHED.
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    context = {
        "simulation_id": simulation.id,
        "seed": simulation.seed,
        "analysis_url": (
            None
            if simulation.proxy
            else reverse("simulation_analysis", args=[simulation.id])
        ),
    }
    return render(request, "pages/watch_simulation.html", context)


def get_simulation_status(request: HttpRequest) -> JsonResponse:
    """Return the state and progress of a simulation for the watch page.

    The offers of the periods are not part of the response (PLAN 2.11). They
    are shown in the offers table of the analysis tab.

    Args:
        request: GET request with ``simulation_id``.

    Returns:
        ``status``, ``mode``, ``total`` (periodNumber), ``period``
        (currentPeriod), ``marketStatus``, ``agents``, ``error`` and
        ``progress`` (:class:`RunProgress` as a dict).
    """
    simulation_id = request.GET.get("simulation_id")
    simulation = get_object_or_404(Simulation, id=simulation_id)
    status = simulation.state.name
    mode = simulation.mode.name
    market = simulation.market
    marketStatus = market.state.name
    currentPeriod = simulation.currentPeriod
    agents = list(market.agent_set.all().order_by("-budget"))
    agent_list = []
    for agent in agents:
        algorithm_type = str(agent.algorithm_type)
        agent_list.append(
            {
                "name": agent.name,
                "state": agent.state.name,
                "budget": agent.budget,
                "algorithm": algorithm_type,
                "failstack_enabled": bool(agent.failstack_enabled)
                and not algorithm_registry.is_baseline(algorithm_type),
            }
        )

    return JsonResponse(
        {
            "status": status,
            "mode": mode,
            "total": simulation.periodNumber,
            "period": currentPeriod,
            "marketStatus": marketStatus,
            "agents": agent_list,
            "error": simulation.error,
            "progress": simulation_progress(simulation, timezone.now()).to_dict(),
        }
    )


def nextPeriod(request):
    simulation_id = request.GET.get("simulation_id")
    simulation = get_object_or_404(Simulation, id=simulation_id)
    simulation.market.state = enums.MarketState.INITIALIZED
    fb = simulation.market.save()
    return JsonResponse({"status": "OK"})


# RESOURCE
def resource_list(request):
    resources = Resource.objects.all()
    return render(request, "pages/resource-list.html", {"resources": resources})


"""def resource_detail(request, resource_id):
    resource = Resource.objects.filter(id=resource_id).first()
    if resource:
        resourceDTO={'energyType':resource.energyType, 
                'name': resource.name, 
                'fuelCost': resource.fuelCost, 
                'emission': resource.emission}
        return render(request, 'pages/resource-view.html', {'resource': resourceDTO})
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")"""


def resource_create(request):
    if request.method == "POST":
        form = ResourceForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("resource_list")
        else:
            print(form.errors)
            return HttpResponseBadRequest()
    return redirect(resource_list)


def resource_delete(request, resource_id):
    resource = Resource.objects.get(id=resource_id)
    if request.method == "GET" and resource:
        resource.delete()
        return redirect("resource_list")
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")


def resource_update(request, resource_id):
    resource = Resource.objects.get(id=resource_id)
    if request.method == "POST":
        form = ResourceForm(request.POST, instance=resource)
        if form.is_valid():
            form.save()
            return redirect("resource_list")
    else:
        return HttpResponseNotFound("KAYNAK BULUNAMADI")


# SIMULATION
def _warm_start_sources(scenario_ids: list[int]) -> dict[str, list[dict[str, Any]]]:
    """Return the runs that can initialise new runs of each scenario (A9).

    Args:
        scenario_ids: Primary keys of proxy scenarios.

    Returns:
        Mapping of scenario id (as a string, for JSON) to the FINISHED runs of
        that scenario with saved policies, newest first, each as ``{"id",
        "name", "seed", "created", "checkpoints", "warm_start_from"}``.
    """
    runs = (
        Simulation.objects.filter(
            proxy=False,
            state=enums.SimulationState.FINISHED,
            scenario_id__in=scenario_ids,
            checkpoints__isnull=False,
        )
        .annotate(n_checkpoints=Count("checkpoints"))
        .order_by("-created_at", "-id")
    )
    sources: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        sources.setdefault(str(run.scenario_id), []).append(
            {
                "id": run.id,
                "name": run.name,
                "seed": run.seed,
                "created": run.created_at.strftime("%Y-%m-%d %H:%M"),
                "checkpoints": run.n_checkpoints,
                "warm_start_from": run.warm_start_from_id,
            }
        )
    return sources


def simulation_list_proxy(request):
    simulations = list(
        Simulation.objects.filter(proxy=True)
        .select_related("market")
        .order_by("-updated_at")
    )
    return render(
        request,
        "pages/simulation-list-proxy.html",
        {
            "simulations": simulations,
            "warm_start_sources": _warm_start_sources([s.id for s in simulations]),
        },
    )


def simulation_list(request):
    simulations = _user_runs().select_related("market").order_by("-updated_at")
    return render(request, "pages/simulation-list.html", {"simulations": simulations})


def _lags_value(lags: list[int]) -> str:
    """Return the select option value of a lag set, for example ``"1,24"``.

    Args:
        lags: Observation lag set.

    Returns:
        The lags joined by commas.
    """
    return ",".join(str(lag) for lag in lags)


def _enum_int(value: Any) -> Any:
    """Return the integer value of an enum member, or ``value`` unchanged."""
    return getattr(value, "value", value)


def _scenario_agent_payload(agent: Agent) -> dict[str, Any]:
    """Serialise an agent of the edited scenario for the wizard (mode ``scenario``).

    Args:
        agent: Agent of the scenario market (``source`` selected, plants
            prefetched).

    Returns:
        The ``_library_agent_payload`` fields plus ``mode``, ``agent_id``,
        ``source_id`` and ``source_name``.
    """
    payload = _library_agent_payload(agent)
    payload.update(
        mode=MODE_SCENARIO,
        agent_id=agent.id,
        source_id=agent.source_id,
        source_name=agent.source.name if agent.source is not None else "",
    )
    return payload


def _stored_demand_source_id(simulation: Simulation) -> int | None:
    """Return the library profile shown as the demand source of a scenario.

    The link is shown only while the stored copy still equals the library
    profile. Otherwise the wizard shows the copy as a generated profile.

    Args:
        simulation: The edited proxy scenario.

    Returns:
        The library profile id, or None.
    """
    template = simulation.demand_profile_source
    if template is None or simulation.demand_profile is None:
        return None
    try:
        equal = normalize_profile(template.profile) == normalize_profile(
            simulation.demand_profile
        )
    except ValidationError:
        return None
    return template.id if equal else None


def _demand_library_options() -> list[dict[str, Any]]:
    """Return the library profiles offered by the wizard's demand source.

    Returns:
        ``[{"id", "name"}]`` ordered by name.
    """
    return [
        {"id": template.id, "name": template.name}
        for template in DemandProfileTemplate.objects.order_by("name", "id")
    ]


def _wizard_context(simulation: Simulation | None = None) -> dict[str, Any]:
    """Return the template context of the scenario wizard (create and update).

    Args:
        simulation: The edited proxy scenario, or None for the create page
            (the fields take the wizard defaults and the model defaults).

    Returns:
        Context with ``resources``, ``allowed_energy_types``, the run settings
        ``run_seed``, ``run_evaluation_periods``, ``run_learning_starts``,
        ``run_observation_lags``
        (option value), ``period_units``, ``run_period_unit`` and
        ``run_period_count`` (duration selector), ``observation_lag_choices`` (value, label pairs),
        ``wizard_config`` (JSON for the page script) and ``simulation`` when
        editing.
    """
    resources = list(Resource.objects.all().order_by("name", "id"))
    if simulation is not None:
        market = simulation.market
        agents = (
            market.agent_set.select_related("source", "portfolio")
            .prefetch_related("portfolio__plant_set__resource")
            .order_by("id")
        )
        initial = {
            "simulation_name": simulation.name,
            "simulationMode": _enum_int(simulation.mode),
            "periodType": _enum_int(simulation.periodType),
            "periodNumber": simulation.periodNumber,
            "strategy": _enum_int(market.strategy),
            "clearing_algorithm": market.clearing_algorithm,
            "clearing_params": dict(market.clearing_params or {}),
            "lowerBound": market.lowerBidBound,
            "upperBound": market.upperBidBound,
            "seed": simulation.seed,
            "evaluation_periods": simulation.evaluation_periods,
            "learning_starts": simulation.learning_starts,
            "observation_lags": list(simulation.observation_lags),
            "demand_profile": simulation.demand_profile,
            "demand_profile_source": _stored_demand_source_id(simulation),
            "agents": [_scenario_agent_payload(agent) for agent in agents],
        }
        save_url = reverse("update_simulation", args=[simulation.id])
        draft_key = f"abmem.scenarioWizard.update.{simulation.id}"
    else:
        initial = {
            "simulation_name": "",
            "simulationMode": enums.SimulationMode.ONLYRESULT.value,
            "periodType": enums.PeriodType.HOUR.value,
            "periodNumber": WIZARD_DEFAULT_PERIODS,
            "strategy": enums.MarketStrategy.PAYASPTF.value,
            "clearing_algorithm": DEFAULT_CLEARING_ALGORITHM,
            "clearing_params": {},
            "lowerBound": WIZARD_DEFAULT_LOWER_BOUND,
            "upperBound": WIZARD_DEFAULT_UPPER_BOUND,
            "seed": DEFAULT_SEED,
            "evaluation_periods": 0,
            "learning_starts": None,
            "observation_lags": default_observation_lags(),
            "demand_profile": None,
            "demand_profile_source": None,
            "agents": [],
        }
        save_url = reverse("create_simulation")
        draft_key = "abmem.scenarioWizard.create"
    config = {
        "is_update": simulation is not None,
        "simulation_id": simulation.id if simulation is not None else None,
        "save_url": save_url,
        "list_url": reverse("simulation_list_proxy"),
        "library_url": reverse("agent_library_json"),
        "check_name_url": reverse("agent_check_name"),
        "demand_stats_url": reverse("demand_stats"),
        "demand_preview_url": reverse("demand_preview"),
        "learning_start_preview_url": reverse("learning_start_preview"),
        "period_units": [
            {"key": key, "label": label, "hours": hours}
            for key, label, hours in PERIOD_UNITS
        ],
        "demand_library": _demand_library_options(),
        # The page replaces the id 0 by the selected profile id.
        "demand_profile_detail_url": reverse("demand_profile_detail_json", args=[0]),
        "demand_yaml_parse_url": reverse("demand_profile_parse_yaml"),
        "demand_save_to_library_url": reverse("demand_profile_create"),
        "demand_library_list_url": reverse("demand_profile_list"),
        "draft_key": draft_key,
        "seed_upper_bound": SEED_UPPER_BOUND,
        "name_max_length": SIMULATION_NAME_MAX_LENGTH,
        "agent_name_max_length": NAME_MAX_LENGTH,
        "initial": initial,
        "resources": {
            str(resource.id): {
                "name": resource.name,
                "energy_type": _enum_int(resource.energyType),
            }
            for resource in resources
        },
        "agent_types": {
            str(member.value): member.name.title() for member in enums.AgentType
        },
        "algorithms": {
            info.key: {"name": info.label, "baseline": info.is_baseline}
            for info in algorithm_registry.available_algorithms()
        },
        "clearing_algorithms": [
            {
                "key": info.key,
                "label": info.label,
                "description": info.description,
                "params": algorithm_catalog.clearing_param_fields(info.key),
            }
            for info in available_clearing_algorithms()
        ],
    }
    period_unit, period_count = period_unit_for(int(initial["periodNumber"]))
    context: dict[str, Any] = {
        "resources": resources,
        "allowed_energy_types": allowed_energy_types_by_agent_type(),
        "period_units": PERIOD_UNITS,
        "run_period_unit": period_unit,
        "run_period_count": period_count,
        "run_seed": initial["seed"],
        "run_evaluation_periods": initial["evaluation_periods"],
        "run_learning_starts": initial["learning_starts"],
        "run_observation_lags": _lags_value(initial["observation_lags"]),
        "observation_lag_choices": [
            (_lags_value(allowed), ", ".join(str(lag) for lag in allowed))
            for allowed in ALLOWED_OBSERVATION_LAGS
        ],
        "wizard_config": config,
    }
    if simulation is not None:
        context["simulation"] = simulation
    return context


def _scenario_saved_response(simulation: Simulation) -> JsonResponse:
    """Build the success response of a scenario save.

    Args:
        simulation: The saved proxy scenario.

    Returns:
        ``{"status": "success", "simulation_id", "edit_url", "run_url"}``.
    """
    return JsonResponse(
        {
            "status": "success",
            "simulation_id": simulation.id,
            "edit_url": reverse("update_simulation", args=[simulation.id]),
            "run_url": reverse(
                "run_simulation", kwargs={"simulation_id": simulation.id}
            ),
        },
        status=200,
    )


@csrf_exempt
def create_simulation(request: HttpRequest) -> HttpResponse:
    """Render the scenario wizard (GET) or create a proxy scenario (POST, JSON).

    Agent entries follow ``_parse_scenario_agents`` (modes ``new`` and
    ``library``). Everything is validated first and written in one
    transaction.

    Args:
        request: The request.

    Returns:
        The page, ``_scenario_saved_response``, or HTTP 400 ``{"errors": {...}}``.
    """
    if request.method != "POST":
        return render(request, "pages/scenario-wizard.html", _wizard_context())
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)
    if not isinstance(data, dict):
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)

    # Validate the header, the settings and every agent before the first
    # write (PLAN 2.2, 2.3, 2.6, OQ-16).
    validated = _validate_scenario_payload(data, current=None, market=None)
    if isinstance(validated, JsonResponse):
        return validated
    header = validated.header

    created: dict[str, Simulation] = {}

    def write() -> None:
        simulation = SimulationFactory.create(
            name=header.name,
            mode=header.mode,
            periodType=header.period_type,
            periodNumber=validated.period_number,
            proxy=True,
        )
        validated.run_settings.apply(simulation)
        simulation.save()
        market = MarketFactory.create(
            sim=simulation,
            lowerBound=header.lower_bound,
            upperBound=header.upper_bound,
            strategy=header.strategy,
            clearing_algorithm=header.clearing_algorithm,
            clearing_params=header.clearing_params,
        )
        _write_scenario_agents(market, validated.entries)
        _check_learning_start(simulation)
        created["simulation"] = simulation

    error_response = _save_scenario(write, validated.entries)
    if error_response is not None:
        return error_response
    return _scenario_saved_response(created["simulation"])


def delete_simulation(request, simulation_id):
    simulation = get_object_or_404(Simulation, id=simulation_id)
    if simulation.proxy:
        simulation.delete()
        return redirect("simulation_list_proxy")
    else:
        simulation.delete()
        return redirect("simulation_list")


@csrf_exempt
def update_simulation(request: HttpRequest, simulation_id: int) -> HttpResponse:
    """Render the prefilled scenario wizard (GET) or update a proxy scenario (POST).

    Agent entries follow ``_parse_scenario_agents`` (modes ``new``,
    ``library`` and ``scenario``). Agents of the scenario that the payload
    does not keep are deleted. Everything is written in one transaction.

    Args:
        request: The request.
        simulation_id: Id of the proxy simulation.

    Returns:
        The page, ``_scenario_saved_response``, or HTTP 400.
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    if request.method == "GET":
        if not simulation.proxy:
            return HttpResponseBadRequest("Only scenario templates can be edited.")
        return render(
            request, "pages/scenario-wizard.html", _wizard_context(simulation)
        )

    if request.method != "POST":
        return HttpResponseBadRequest()
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)
    if not isinstance(data, dict) or not simulation.proxy:
        return HttpResponseBadRequest()

    # Validate the header, the settings and every agent before the first
    # write (PLAN 2.2, 2.3, 2.6, OQ-16).
    validated = _validate_scenario_payload(
        data, current=simulation, market=simulation.market
    )
    if isinstance(validated, JsonResponse):
        return validated
    header = validated.header

    def write() -> None:
        simulation.name = header.name
        simulation.mode = header.mode
        simulation.periodNumber = validated.period_number
        simulation.periodType = header.period_type
        simulation.state = enums.SimulationState.CREATED
        simulation.currentPeriod = -1
        simulation.proxy = True
        validated.run_settings.apply(simulation)
        simulation.save()

        market = simulation.market
        market.lowerBidBound = header.lower_bound
        market.upperBidBound = header.upper_bound
        market.strategy = header.strategy
        market.clearing_algorithm = header.clearing_algorithm
        market.clearing_params = dict(header.clearing_params)
        market.save()

        _write_scenario_agents(market, validated.entries)
        _check_learning_start(simulation)

    error_response = _save_scenario(write, validated.entries)
    if error_response is not None:
        return error_response
    return _scenario_saved_response(simulation)


def simulation_view(request, sim_id):
    simulations = Simulation.objects.order_by("-created_at")
    return render(request, "pages/blank.html", {"simulations": simulations})


# AGENT (library, PLAN 2.3)
def get_agents(request):
    agents = Agent.objects.all().values("name", "type", "budget")
    return JsonResponse(list(agents), safe=False)


def _agent_input_from_form(form: AgentForm, plant_forms: list[PlantForm]) -> AgentInput:
    """Build the library service input from a valid AgentForm and plant forms.

    The values are read from ``form.instance``, which holds the submitted
    values merged with the defaults (create) or the stored values (edit) for
    omitted optional fields.

    Args:
        form: A valid ``AgentForm``.
        plant_forms: Valid ``PlantForm`` instances.

    Returns:
        The agent input.
    """
    agent = form.instance
    return AgentInput(
        name=agent.name,
        budget=Decimal(agent.budget),
        type=agent.type,
        algorithm_type=agent.algorithm_type,
        failstack_enabled=bool(agent.failstack_enabled),
        failstack_limit=agent.failstack_limit,
        failstack_delta=Decimal(agent.failstack_delta),
        failstack_min_margin=Decimal(agent.failstack_min_margin),
        plants=tuple(
            PlantInput(
                plant_form.cleaned_data["resource"].id,
                plant_form.cleaned_data["capacity"],
            )
            for plant_form in plant_forms
        ),
    )


def _save_library_agent(request: HttpRequest, instance: Agent | None) -> HttpResponse:
    """Validate the agent form POST and create or update a library agent.

    Args:
        request: POST with the ``AgentForm`` fields plus ``resources[]`` and
            ``capacities[]``.
        instance: Library agent to update, None to create one.

    Returns:
        HTTP 400 ``{"errors": {...}}`` on invalid input. On success
        ``{"status": "success", "agent": {...}}`` for a JSON client, else a
        redirect to the agent list.
    """
    resources = request.POST.getlist("resources[]")
    capacities = request.POST.getlist("capacities[]")
    plant_forms = [
        PlantForm({"resource": resource, "capacity": capacity})
        for resource, capacity in zip(resources, capacities)
    ]
    plant_errors = _plant_form_errors(plant_forms)
    if plant_errors:
        return JsonResponse({"errors": plant_errors}, status=400)

    agent_form = AgentForm(
        request.POST,
        instance=instance,
        plant_resources=[form.cleaned_data["resource"] for form in plant_forms],
    )
    if not agent_form.is_valid():
        return JsonResponse({"errors": _agent_form_errors(agent_form)}, status=400)

    data = _agent_input_from_form(agent_form, plant_forms)
    try:
        if instance is None:
            agent = library_service.create_library_agent(data)
        else:
            agent = library_service.update_library_agent(instance, data)
    except LibraryValidationError as exc:
        return _validation_error_response(exc)

    if _wants_json(request):
        return JsonResponse(
            {"status": "success", "agent": _library_agent_payload(agent)}, status=200
        )
    return redirect("agent_list")


def create_agent_and_related(request: HttpRequest) -> HttpResponse:
    """Create a library agent with its plants (POST only).

    Args:
        request: The request.

    Returns:
        See ``_save_library_agent``. HTTP 400 for other methods.
    """
    if request.method != "POST":
        return HttpResponseBadRequest()
    return _save_library_agent(request, None)


def update_agent(request: HttpRequest, agent_id: int) -> HttpResponse:
    """Update a library agent and replace its plants (POST only).

    Copies of the agent in scenarios and runs are not changed.

    Args:
        request: The request.
        agent_id: Id of the library agent.

    Returns:
        See ``_save_library_agent``. HTTP 400 for other methods or when the
        agent is not a library agent.
    """
    agent = get_object_or_404(Agent, id=agent_id)
    if request.method != "POST":
        return HttpResponseBadRequest()
    if not agent.is_library:
        return JsonResponse(
            {"errors": {"agent": ["Only library agents can be edited here."]}},
            status=400,
        )
    return _save_library_agent(request, agent)


def delete_agent(request: HttpRequest, agent_id: int) -> HttpResponse:
    """Delete a library agent (POST only) unless a scenario still uses it.

    Copies in runs keep working with ``source`` NULL.

    Args:
        request: The request.
        agent_id: Id of the library agent.

    Returns:
        Redirect to the agent list (JSON ``{"status": "success"}`` for a JSON
        client). HTTP 400 for other methods, a non-library agent, or an agent
        used by a scenario (JSON ``{"errors": {"agent": [...]}}`` naming the
        scenarios).
    """
    agent = get_object_or_404(Agent, id=agent_id)
    if request.method != "POST" or not agent.is_library:
        return HttpResponseBadRequest()
    try:
        library_service.delete_library_agent(agent)
    except library_service.LibraryValidationError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    if _wants_json(request):
        return JsonResponse({"status": "success"}, status=200)
    return redirect("agent_list")


def check_agent_name(request: HttpRequest) -> JsonResponse:
    """Live uniqueness check of a library agent name.

    Uses ``library_service.check_name_available`` (database ``LOWER``, the
    same comparison as the constraint ``uniq_library_agent_name_ci``).

    Args:
        request: GET with ``name`` and optional ``exclude_id`` (the agent
            being edited).

    Returns:
        ``{"available": bool, "message": str}``. HTTP 400 for a malformed
        ``exclude_id``, HTTP 405 for other methods.
    """
    if request.method != "GET":
        return JsonResponse(
            {"available": False, "message": "Method not allowed."}, status=405
        )
    raw_exclude = request.GET.get("exclude_id", "").strip()
    exclude_id = None
    if raw_exclude:
        exclude_id = _coerce_id(raw_exclude)
        if exclude_id is None:
            return JsonResponse(
                {"available": False, "message": "exclude_id must be an agent id."},
                status=400,
            )
    try:
        name = _validate_entry_name(request.GET.get("name", ""))
    except ValidationError as exc:
        return JsonResponse({"available": False, "message": exc.messages[0]})
    if not library_service.check_name_available(name, exclude_id=exclude_id):
        return JsonResponse(
            {"available": False, "message": _library_name_taken_message(name)}
        )
    return JsonResponse({"available": True, "message": ""})


def library_agents_json(request: HttpRequest) -> JsonResponse:
    """List the library agents for the "select from library" dialog.

    Args:
        request: The request.

    Returns:
        JSON list of ``_library_agent_payload`` objects ordered by name.
    """
    return JsonResponse(
        [_library_agent_payload(agent) for agent in _library_agents()], safe=False
    )


def agent_list(request: HttpRequest) -> HttpResponse:
    """Render the agent library page (library agents only).

    Args:
        request: The request.

    Returns:
        The page, HTTP 400 for other methods.
    """
    if request.method != "GET":
        return HttpResponseBadRequest()
    return render(
        request,
        "pages/agent-list.html",
        {
            "resources": Resource.objects.all(),
            "agents": _library_agents(),
            "allowed_energy_types": allowed_energy_types_by_agent_type(),
        },
    )


"""
def get_agents(request):
    agents = Agent.objects.all()
    agents_data = []
    for agent in agents:
        agents_data.append({
            'id': agent.id,
            'name': agent.name,
            'type': agent.type,
            'budget': agent.budget
        })
    return JsonResponse({'agents': agents_data})

"""


# HELPERS
def increment_suffix(name):
    # Regex pattern to find an underscore followed by digits at the end of the string
    pattern = r"_(\d+)$"

    # Search for the pattern in the given string
    match = re.search(pattern, name)

    if match:
        # If a match is found, increment the number by 1
        number = int(match.group(1)) + 1
        # Replace the old number with the new number
        new_name = re.sub(pattern, f"_{number}", name)
    else:
        # If no match is found, append _2 to the string
        new_name = f"{name}_2"

    return new_name
