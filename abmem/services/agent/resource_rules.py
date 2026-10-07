"""Agent-type rule for which resources an agent may own (PLAN 2.2).

An agent of type RENEWABLE, FOSSIL or NUCLEAR may only own plants whose
resource has the same energy type. A HYBRID agent may own plants of every
energy type. ``allowed_energy_types`` is the single definition of this rule.
Views, forms, templates (through ``allowed_energy_types_by_agent_type``) and
the ``check_resource_rules`` management command all derive from it.
"""

from __future__ import annotations

from typing import Sequence

from django.core.exceptions import ValidationError

from ...models.agent import Agent
from ...models.enums import AgentType, EnergyType
from ...models.models import Resource

_ALLOWED_ENERGY_TYPES: dict[AgentType, frozenset[EnergyType]] = {
    AgentType.RENEWABLE: frozenset({EnergyType.RENEWABLE}),
    AgentType.FOSSIL: frozenset({EnergyType.FOSSIL}),
    AgentType.NUCLEAR: frozenset({EnergyType.NUCLEAR}),
    AgentType.HYBRID: frozenset(EnergyType),
}


def allowed_energy_types(agent_type: AgentType) -> frozenset[EnergyType]:
    """Return the energy types an agent of the given type may own.

    Args:
        agent_type: Agent type as an ``AgentType`` member or its integer value.

    Returns:
        The set of allowed ``EnergyType`` members.

    Raises:
        ValueError: If ``agent_type`` is not a valid ``AgentType`` value.
    """
    if isinstance(agent_type, bool) or not isinstance(agent_type, int):
        raise ValueError(f"Unknown agent type: {agent_type!r}")
    try:
        member = AgentType(agent_type)
    except ValueError as exc:
        raise ValueError(f"Unknown agent type: {agent_type!r}") from exc
    return _ALLOWED_ENERGY_TYPES[member]


def allowed_energy_types_by_agent_type() -> dict[str, list[int]]:
    """Serialise the rule for the frontend (``json_script`` in templates).

    Returns:
        A mapping from the agent type value (as a string key, the JSON object
        key type) to the sorted list of allowed energy type values.
    """
    return {
        str(agent_type.value): sorted(e.value for e in allowed_energy_types(agent_type))
        for agent_type in AgentType
    }


def validate_agent_resources(
    agent_type: AgentType, resources: Sequence[Resource]
) -> None:
    """Check that every resource is allowed for the agent type.

    Args:
        agent_type: Agent type as an ``AgentType`` member or its integer value.
        resources: Resources of the agent's plants (duplicates are allowed).

    Raises:
        ValidationError: ``'At least one plant is required'`` when
            ``resources`` is empty, otherwise ``{'plants': [...]}`` with one
            message naming the agent type and all offending resources.
        ValueError: If ``agent_type`` is not a valid ``AgentType`` value.
    """
    allowed = allowed_energy_types(agent_type)
    if len(resources) == 0:
        raise ValidationError("At least one plant is required", code="no_plants")
    offending: list[str] = []
    for resource in resources:
        if (
            EnergyType(resource.energyType) not in allowed
            and resource.name not in offending
        ):
            offending.append(resource.name)
    if offending:
        type_name = AgentType(agent_type).name
        allowed_names = ", ".join(sorted(e.name for e in allowed))
        raise ValidationError(
            {
                "plants": [
                    ValidationError(
                        f"A {type_name} agent may only own {allowed_names} resources. "
                        f"Not allowed: {', '.join(offending)}.",
                        code="resource_not_allowed",
                    )
                ]
            }
        )


def find_violations() -> list[tuple[int, str, str, list[str]]]:
    """Scan all agents for plants that break the rule.

    Agents without a type are reported with type name ``"UNSET"`` and all of
    their resources, because no resource is allowed without a type.

    Returns:
        One ``(agent_id, agent_name, agent_type_name, offending_resource_names)``
        tuple per violating agent, ordered by agent id.
    """
    violations: list[tuple[int, str, str, list[str]]] = []
    agents = Agent.objects.order_by("id").prefetch_related(
        "portfolio__plant_set__resource"
    )
    for agent in agents:
        try:
            plants = list(agent.portfolio.plant_set.all())
        except Agent.portfolio.RelatedObjectDoesNotExist:
            continue
        if agent.type is None:
            allowed: frozenset[EnergyType] = frozenset()
            type_name = "UNSET"
        else:
            allowed = allowed_energy_types(agent.type)
            type_name = AgentType(agent.type).name
        offending: list[str] = []
        for plant in plants:
            name = plant.resource.name
            if (
                EnergyType(plant.resource.energyType) not in allowed
                and name not in offending
            ):
                offending.append(name)
        if offending:
            violations.append((agent.id, agent.name, type_name, offending))
    return violations
