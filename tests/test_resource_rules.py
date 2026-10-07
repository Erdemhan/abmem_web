"""Tests for the agent-type resource rule (PLAN 2.2, spec resource_filter)."""

from __future__ import annotations

import importlib
from io import StringIO

import pytest
from django.core.exceptions import ValidationError
from django.core.management import call_command
from django.core.management.base import CommandError
from django.db import connection
from django.db.migrations.loader import MigrationLoader

from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent.resource_rules import (
    allowed_energy_types,
    allowed_energy_types_by_agent_type,
    find_violations,
    validate_agent_resources,
)
from tests.factories import make_agent_with_plants, make_market, make_resource

EXPECTED_ALLOWED: dict[AgentType, set[EnergyType]] = {
    AgentType.RENEWABLE: {EnergyType.RENEWABLE},
    AgentType.FOSSIL: {EnergyType.FOSSIL},
    AgentType.NUCLEAR: {EnergyType.NUCLEAR},
    AgentType.HYBRID: {EnergyType.RENEWABLE, EnergyType.FOSSIL, EnergyType.NUCLEAR},
}

MATRIX = [
    (agent_type, energy_type, energy_type in EXPECTED_ALLOWED[agent_type])
    for agent_type in AgentType
    for energy_type in EnergyType
]


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_allowed_energy_types_matrix(
    agent_type: AgentType, energy_type: EnergyType, allowed: bool
) -> None:
    assert (energy_type in allowed_energy_types(agent_type)) is allowed


@pytest.mark.parametrize("agent_type", list(AgentType))
def test_allowed_energy_types_accepts_int_value(agent_type: AgentType) -> None:
    assert allowed_energy_types(agent_type.value) == frozenset(
        EXPECTED_ALLOWED[agent_type]
    )
    assert isinstance(allowed_energy_types(agent_type), frozenset)


@pytest.mark.parametrize("bad", [4, -1, None, "1", True, 1.0])
def test_allowed_energy_types_rejects_unknown(bad: object) -> None:
    with pytest.raises(ValueError):
        allowed_energy_types(bad)  # type: ignore[arg-type]


def test_serialised_mapping_matches_rule() -> None:
    mapping = allowed_energy_types_by_agent_type()
    assert mapping == {
        str(t.value): sorted(e.value for e in EXPECTED_ALLOWED[t]) for t in AgentType
    }


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_validate_agent_resources_matrix(
    agent_type: AgentType, energy_type: EnergyType, allowed: bool
) -> None:
    resource = make_resource(
        name=f"r-{energy_type.name}", energy_type=energy_type, save=False
    )
    if allowed:
        validate_agent_resources(agent_type, [resource])
    else:
        with pytest.raises(ValidationError) as info:
            validate_agent_resources(agent_type, [resource])
        messages = info.value.message_dict["plants"]
        assert len(messages) == 1
        assert agent_type.name in messages[0]
        assert resource.name in messages[0]


def test_validate_agent_resources_empty() -> None:
    with pytest.raises(ValidationError) as info:
        validate_agent_resources(AgentType.HYBRID, [])
    assert info.value.messages == ["At least one plant is required"]


def test_validate_agent_resources_lists_every_offender_once() -> None:
    solar = make_resource(name="Solar", energy_type=EnergyType.RENEWABLE, save=False)
    uranium = make_resource(name="Uranium", energy_type=EnergyType.NUCLEAR, save=False)
    coal = make_resource(name="Coal", energy_type=EnergyType.FOSSIL, save=False)
    with pytest.raises(ValidationError) as info:
        validate_agent_resources(AgentType.FOSSIL, [solar, coal, uranium, solar])
    (message,) = info.value.message_dict["plants"]
    assert "FOSSIL" in message
    assert "Solar" in message and "Uranium" in message
    assert "Not allowed: Solar, Uranium." in message


def test_validate_hybrid_accepts_all_energy_types_and_duplicates() -> None:
    resources = [
        make_resource(name=f"h-{e.name}", energy_type=e, save=False) for e in EnergyType
    ]
    validate_agent_resources(AgentType.HYBRID, resources + resources)


@pytest.mark.django_db
def test_find_violations_and_command() -> None:
    market = make_market()
    solar = make_resource(name="SolarX", energy_type=EnergyType.RENEWABLE)
    coal = make_resource(name="CoalX", energy_type=EnergyType.FOSSIL)
    make_agent_with_plants(
        market, [(coal, 10)], name="ok-fossil", agent_type=AgentType.FOSSIL
    )
    make_agent_with_plants(
        market, [(solar, 5), (coal, 5)], name="hyb", agent_type=AgentType.HYBRID
    )
    bad = make_agent_with_plants(
        market, [(coal, 10), (solar, 5)], name="bad-fossil", agent_type=AgentType.FOSSIL
    )

    assert find_violations() == [(bad.id, "bad-fossil", "FOSSIL", ["SolarX"])]

    out = StringIO()
    call_command("check_resource_rules", stdout=out)
    assert "bad-fossil" in out.getvalue()
    assert "1 violation(s) found." in out.getvalue()

    with pytest.raises(CommandError) as info:
        call_command("check_resource_rules", "--fail-on-violation", stdout=StringIO())
    assert info.value.returncode == 1


@pytest.mark.django_db
def test_command_without_violations_succeeds() -> None:
    market = make_market()
    coal = make_resource(name="CoalY", energy_type=EnergyType.FOSSIL)
    make_agent_with_plants(market, [(coal, 1)], agent_type=AgentType.FOSSIL)
    out = StringIO()
    call_command("check_resource_rules", "--fail-on-violation", stdout=out)
    assert "0 violation(s) found." in out.getvalue()


MIGRATION = importlib.import_module("abmem.migrations.0004_resource_ccgt_fossil")


def _historical_apps(node: str):
    """Return the historical app registry at the given abmem migration."""
    loader = MigrationLoader(connection)
    return loader.project_state(("abmem", node)).apps


def test_migration_depends_on_0003() -> None:
    assert MIGRATION.Migration.dependencies == [("abmem", "0003_remove_agent_seed")]


@pytest.mark.django_db
def test_migration_ccgt_forward_and_reverse() -> None:
    ccgt = make_resource(name="CCGT", energy_type=EnergyType.RENEWABLE)
    wind = make_resource(name="Wind", energy_type=EnergyType.RENEWABLE)

    MIGRATION.ccgt_to_fossil(_historical_apps("0003_remove_agent_seed"), None)
    ccgt.refresh_from_db()
    wind.refresh_from_db()
    assert ccgt.energyType == EnergyType.FOSSIL
    assert wind.energyType == EnergyType.RENEWABLE

    MIGRATION.ccgt_to_renewable(_historical_apps("0004_resource_ccgt_fossil"), None)
    ccgt.refresh_from_db()
    wind.refresh_from_db()
    assert ccgt.energyType == EnergyType.RENEWABLE
    assert wind.energyType == EnergyType.RENEWABLE


@pytest.mark.django_db
def test_migration_is_noop_without_ccgt() -> None:
    make_resource(name="Coal", energy_type=EnergyType.FOSSIL)
    MIGRATION.ccgt_to_fossil(_historical_apps("0003_remove_agent_seed"), None)
    assert list(Resource.objects.values_list("name", "energyType")) == [
        ("Coal", EnergyType.FOSSIL)
    ]
