import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

from django.core.exceptions import ValidationError
from django.db import IntegrityError, transaction
from django.db.models.functions import Lower
from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotFound,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from .forms import AgentForm, PlantForm, ResourceForm
from .models import Agent, Simulation, enums
from .models.market import Market
from .models.models import Plant, Portfolio, Resource
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
from .services.algorithms.registry import BASELINE_TYPES
from .services.market import market_factory as MarketFactory
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


def _coerce_algorithm_type(raw_type: Any) -> enums.AlgorithmType:
    """Convert a JSON algorithm type (int, digit string or name) to ``AlgorithmType``.

    Args:
        raw_type: Value of the ``algorithm_type`` key. None or "" selects the
            model default (DDPG).

    Returns:
        The matching ``AlgorithmType`` member.

    Raises:
        ValidationError: ``{'algorithm_type': [...]}`` for a value outside the enum.
    """
    if raw_type is None or raw_type == "":
        default = Agent._meta.get_field("algorithm_type").get_default()
        return enums.AlgorithmType.get(getattr(default, "value", default))
    member = None
    if isinstance(raw_type, int) and not isinstance(raw_type, bool):
        member = enums.AlgorithmType.get(raw_type)
    elif isinstance(raw_type, str):
        text = raw_type.strip()
        if text.isdigit():
            member = enums.AlgorithmType.get(int(text))
        elif text in enums.AlgorithmType.__members__:
            member = enums.AlgorithmType[text]
    if member is None:
        known = ", ".join(f"{t.value} ({t.name})" for t in enums.AlgorithmType)
        raise ValidationError(
            {
                "algorithm_type": [
                    f"Unknown algorithm type: {raw_type!r}. Known types: {known}."
                ]
            }
        )
    return member


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
        raise ValidationError(f"must be >= 0 TL/MWh, got {value}.")
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

    Expected keys: ``algorithm_type`` (int, digit string or enum name) and
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

    if result["failstack_enabled"] and result["algorithm_type"] in BASELINE_TYPES:
        errors.setdefault("failstack_enabled", []).append(
            "The fail-stack rule cannot be enabled for the baseline algorithm "
            f"{result['algorithm_type'].name}."
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
        algorithm_name, failstack: {enabled, active, limit, delta,
        min_margin}, plants: [{resource_id, resource_name, energy_type,
        capacity}]}``. Decimals are strings.
    """
    agent_type = enums.AgentType.get(getattr(agent.type, "value", agent.type))
    algorithm = enums.AlgorithmType.get(
        getattr(agent.algorithm_type, "value", agent.algorithm_type)
    )
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
        "algorithm_type": algorithm.value if algorithm is not None else None,
        "algorithm_name": algorithm.name if algorithm is not None else "",
        "failstack": {
            "enabled": bool(agent.failstack_enabled),
            "active": bool(agent.failstack_enabled) and algorithm not in BASELINE_TYPES,
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
def run_simulation(request: HttpRequest, simulation_id: int) -> HttpResponse:
    """Clone a proxy scenario into a new run and start it.

    The scenario (simulation, market, agents, portfolios, plants) is copied
    atomically by ``library_service.clone_scenario_for_run``. A run
    (non-proxy simulation) is not cloned again, the request redirects to its
    watch page.

    Args:
        request: The request.
        simulation_id: Id of the simulation.

    Returns:
        Redirect to the watch page of the new run, redirect to the watch page
        of ``simulation_id`` for a run, HTTP 400 when the scenario cannot be
        cloned, HTTP 500 when the run process cannot be started.
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    if not simulation.proxy:
        return redirect("watch_simulation", simulation_id=simulation.id)
    try:
        run = library_service.clone_scenario_for_run(simulation)
    except (ValidationError, ValueError) as exc:
        logger.warning("Scenario %s could not be cloned: %s", simulation.id, exc)
        return HttpResponseBadRequest(f"The scenario could not be cloned: {exc}")
    try:
        run_id = StarterService.start(run)
    except Exception:
        logger.exception(
            "Run %s of scenario %s could not be started", run.id, simulation.id
        )
        run.delete()
        return HttpResponse("The simulation run could not be started.", status=500)
    return redirect("watch_simulation", simulation_id=run_id)


def _user_runs():
    """Return the user's simulation runs without HPO evaluation runs.

    HPO evaluation simulations are linked to an ``HPOTrial`` and are listed on
    the watch page of their study instead (ARCHITECTURE.md 7.2).

    Returns:
        Queryset of ``Simulation`` with ``proxy=False`` and no HPO trial.
    """
    return Simulation.objects.filter(proxy=False, hpo_trial__isnull=True)


def dashboard(request):
    num_resources = Resource.objects.count()
    num_agents = Agent.objects.count()
    num_simulations = _user_runs().count()
    num_proxy_simulations = Simulation.objects.filter(proxy=True).count()
    latest_simulations = _user_runs().order_by("-created_at")[:5]
    context = {
        "num_resources": num_resources,
        "num_agents": num_agents,
        "num_simulations": num_simulations,
        "num_proxy_simulations": num_proxy_simulations,
        "latest_simulations": latest_simulations,
    }
    return render(request, "pages/dashboard.html", context)


def blank(request):
    resources = Resource.objects.all()
    return render(request, "pages/blank.html", {"resources": resources})


def watch_simulation(request, simulation_id):
    simulation = get_object_or_404(Simulation, id=simulation_id)
    context = {"simulation_id": simulation.id}
    return render(request, "pages/watch_simulation.html", context)


# Simülasyon durumu kontrol view fonksiyonu
def get_simulation_status(request):
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
        algorithm_type = enums.AlgorithmType.get(
            getattr(agent.algorithm_type, "value", agent.algorithm_type)
        )
        agent_list.append(
            {
                "name": agent.name,
                "state": agent.state.name,
                "budget": agent.budget,
                "algorithm": algorithm_type.name if algorithm_type is not None else "",
                "failstack_enabled": bool(agent.failstack_enabled)
                and algorithm_type not in BASELINE_TYPES,
            }
        )

    periods_list = []
    if simulation.state == enums.SimulationState.FINISHED:
        periods = list(market.period_set.all())
        for period in periods:
            offersList = []
            if period != None:
                offers = list(period.offer_set.all().order_by("-agent_id"))
                for offer in offers:
                    offersList.append(offer.to_dict())
                if period.ptf != -1:
                    offersList.append(create_mcp(period.ptf))
                periods_list.append({"id": period.periodNumber, "offers": offersList})
    else:
        period = market.period_set.filter(periodNumber=currentPeriod).first()
        offersList = []
        if period != None:
            offers = list(period.offer_set.all().order_by("-agent_id"))
            for offer in offers:
                offersList.append(offer.to_dict())
            if period.ptf != -1:
                offersList.append(create_mcp(period.ptf))
            periods_list.append({"id": period.periodNumber, "offers": offersList})

    return JsonResponse(
        {
            "status": status,
            "mode": mode,
            "total": simulation.periodNumber,
            "period": currentPeriod,
            "marketStatus": marketStatus,
            "agents": agent_list,
            "periods": periods_list,
            "error": simulation.error,
        }
    )


def nextPeriod(request):
    simulation_id = request.GET.get("simulation_id")
    simulation = get_object_or_404(Simulation, id=simulation_id)
    simulation.market.state = enums.MarketState.INITIALIZED
    fb = simulation.market.save()
    return JsonResponse({"status": "OK"})


def create_mcp(mcp):
    return {
        "id": -1,
        "agent": "MCP",
        "resource": mcp,
        "amount": mcp,
        "offerPrice": mcp,
        "acceptance": mcp,
        "acceptancePrice": mcp,
        "acceptanceAmount": mcp,
    }


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
def simulation_list_proxy(request):
    simulations = Simulation.objects.filter(proxy=True).order_by("-updated_at")
    return render(
        request, "pages/simulation-list-proxy.html", {"simulations": simulations}
    )


def simulation_list(request):
    simulations = _user_runs().order_by("-updated_at")
    return render(request, "pages/simulation-list.html", {"simulations": simulations})


def _scenario_page_context(**extra: Any) -> dict[str, Any]:
    """Return the template context shared by the scenario create/update pages.

    Args:
        **extra: Additional context entries.

    Returns:
        Context with ``resources`` and ``allowed_energy_types``.
    """
    return {
        "resources": Resource.objects.all(),
        "allowed_energy_types": allowed_energy_types_by_agent_type(),
        **extra,
    }


@csrf_exempt
def create_simulation(request: HttpRequest) -> HttpResponse:
    """Render the scenario form (GET) or create a proxy scenario (POST, JSON).

    Agent entries follow ``_parse_scenario_agents`` (modes ``new`` and
    ``library``). Everything is written in one transaction.

    Args:
        request: The request.

    Returns:
        The page, ``{"status": "success", "simulation_id": id}``, or HTTP 400
        ``{"errors": {...}}``.
    """
    if request.method != "POST":
        return render(request, "pages/simulation-create.html", _scenario_page_context())
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)
    simulation_name = data.get("simulation_name")
    simulationMode = int(data["simulationMode"])
    periodNumber = int(data["periodNumber"])
    periodType = int(data["periodType"])

    marketStrategy = int(data["strategy"])
    lowerBound = int(data["lowerBound"])
    upperBound = int(data["upperBound"])

    # Validate every agent before the first write (PLAN 2.2, 2.3).
    try:
        entries = _parse_scenario_agents(data.get("agents", []), market=None)
    except ValidationError as exc:
        return _validation_error_response(exc)

    created: dict[str, Simulation] = {}

    def write() -> None:
        simulation = SimulationFactory.create(
            name=simulation_name,
            mode=enums.SimulationMode(simulationMode),
            periodType=enums.PeriodType(periodType),
            periodNumber=periodNumber,
            proxy=True,
        )
        market = MarketFactory.create(
            sim=simulation,
            lowerBound=lowerBound,
            upperBound=upperBound,
            strategy=enums.MarketStrategy(marketStrategy),
        )
        _write_scenario_agents(market, entries)
        created["simulation"] = simulation

    error_response = _save_scenario(write, entries)
    if error_response is not None:
        return error_response
    return JsonResponse(
        {"status": "success", "simulation_id": created["simulation"].id}, status=200
    )


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
    """Render the scenario edit page (GET) or update a proxy scenario (POST, JSON).

    Agent entries follow ``_parse_scenario_agents`` (modes ``new``,
    ``library`` and ``scenario``). Agents of the scenario that the payload
    does not keep are deleted. Everything is written in one transaction.

    Args:
        request: The request.
        simulation_id: Id of the proxy simulation.

    Returns:
        The page, ``{"status": "success"}``, or HTTP 400.
    """
    if request.method == "GET":
        simulation = get_object_or_404(Simulation, id=simulation_id)
        agents = list(
            simulation.market.agent_set.select_related("source").order_by("id")
        )
        return render(
            request,
            "pages/simulation-update.html",
            _scenario_page_context(simulation=simulation, agents=agents),
        )

    if request.method != "POST":
        return HttpResponseBadRequest()
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)

    simulation_name = data.get("simulation_name")
    simulationMode = int(data["simulationMode"])
    periodNumber = int(data["periodNumber"])
    periodType = int(data["periodType"])

    marketStrategy = int(data["strategy"])
    lowerBound = int(data["lowerBound"])
    upperBound = int(data["upperBound"])

    simulation = get_object_or_404(Simulation, id=simulation_id)
    if not simulation.proxy:
        return HttpResponseBadRequest()

    # Validate every agent before the first write (PLAN 2.2, 2.3).
    try:
        entries = _parse_scenario_agents(
            data.get("agents", []), market=simulation.market
        )
    except ValidationError as exc:
        return _validation_error_response(exc)

    def write() -> None:
        simulation.name = simulation_name
        simulation.mode = enums.SimulationMode(simulationMode)
        simulation.periodNumber = periodNumber
        simulation.periodType = enums.PeriodType(periodType)
        simulation.state = enums.SimulationState.CREATED
        simulation.currentPeriod = -1
        simulation.proxy = True
        simulation.save()

        market = simulation.market
        market.lowerBidBound = lowerBound
        market.upperBidBound = upperBound
        market.strategy = enums.MarketStrategy(marketStrategy)
        market.save()

        _write_scenario_agents(market, entries)

    error_response = _save_scenario(write, entries)
    if error_response is not None:
        return error_response
    return JsonResponse({"status": "success"}, status=200)


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
    """Delete a library agent (POST only). Its copies keep working with ``source`` NULL.

    Args:
        request: The request.
        agent_id: Id of the library agent.

    Returns:
        Redirect to the agent list (JSON ``{"status": "success"}`` for a JSON
        client), HTTP 400 for other methods or a non-library agent.
    """
    agent = get_object_or_404(Agent, id=agent_id)
    if request.method != "POST" or not agent.is_library:
        return HttpResponseBadRequest()
    agent.delete()
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
