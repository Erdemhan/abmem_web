"""View tests for the agent-type resource rule (PLAN 2.2, spec resource_filter).

Every AgentType x EnergyType combination is sent through the Django test
client to the four write paths: create_simulation, update_simulation,
create_agent_and_related and update_agent.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Plant, Portfolio, Resource
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_resource,
    make_simulation,
)

pytestmark = pytest.mark.django_db

ALLOWED: dict[AgentType, set[EnergyType]] = {
    AgentType.RENEWABLE: {EnergyType.RENEWABLE},
    AgentType.FOSSIL: {EnergyType.FOSSIL},
    AgentType.NUCLEAR: {EnergyType.NUCLEAR},
    AgentType.HYBRID: set(EnergyType),
}
MATRIX = [
    (agent_type, energy_type, energy_type in ALLOWED[agent_type])
    for agent_type in AgentType
    for energy_type in EnergyType
]


@pytest.fixture
def resources() -> dict[EnergyType, Resource]:
    """One saved resource per energy type."""
    return {e: make_resource(name=f"view-{e.name}", energy_type=e) for e in EnergyType}


def _simulation_payload(agents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "simulation_name": "rf-sim",
        "simulationMode": "1",
        "periodNumber": "5",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "3000",
        "agents": agents,
    }


def _agent(
    agent_type: Any, resource_ids: list[Any], name: str = "a1"
) -> dict[str, Any]:
    return {
        "name": name,
        "type": agent_type,
        "budget": "100",
        "plants": [{"resource": rid, "capacity": "10"} for rid in resource_ids],
    }


def _post_json(client: Client, url: str, payload: dict[str, Any]):
    return client.post(url, data=json.dumps(payload), content_type="application/json")


def _proxy_simulation() -> Simulation:
    simulation = make_simulation(name="proxy-sim")
    simulation.proxy = True
    simulation.save()
    market = make_market(simulation=simulation)
    market.proxy = True
    market.save()
    return simulation


# create_simulation


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_create_simulation_matrix(
    client: Client,
    resources: dict[EnergyType, Resource],
    agent_type: AgentType,
    energy_type: EnergyType,
    allowed: bool,
) -> None:
    payload = _simulation_payload(
        [_agent(str(agent_type.value), [resources[energy_type].id])]
    )
    response = _post_json(client, reverse("create_simulation"), payload)
    if allowed:
        assert response.status_code == 200
        assert Simulation.objects.count() == 1
        agent = Agent.objects.get()
        assert agent.type == agent_type
        assert list(
            agent.portfolio.plant_set.values_list("resource_id", flat=True)
        ) == [resources[energy_type].id]
    else:
        assert response.status_code == 400
        messages = response.json()["errors"]["plants"]
        assert any(
            resources[energy_type].name in m and agent_type.name in m for m in messages
        )
        assert Simulation.objects.count() == 0
        assert Agent.objects.count() == 0
        assert Portfolio.objects.count() == 0
        assert Plant.objects.count() == 0


def test_create_simulation_is_atomic_across_agents(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    payload = _simulation_payload(
        [
            _agent(1, [resources[EnergyType.FOSSIL].id], name="good"),
            _agent(1, [resources[EnergyType.RENEWABLE].id], name="bad"),
        ]
    )
    response = _post_json(client, reverse("create_simulation"), payload)
    assert response.status_code == 400
    assert "bad" in response.json()["errors"]["plants"][0]
    assert Simulation.objects.count() == 0
    assert Agent.objects.count() == 0


def test_create_simulation_hybrid_with_all_types(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    ids = [r.id for r in resources.values()]
    response = _post_json(
        client,
        reverse("create_simulation"),
        _simulation_payload([_agent(3, ids + ids[:1])]),
    )
    assert response.status_code == 200
    assert Plant.objects.count() == 4


@pytest.mark.parametrize(
    "agent_type",
    [None, "", "abc", 1.5, True, 9, [1]],
)
def test_create_simulation_bad_agent_type(
    client: Client, resources: dict[EnergyType, Resource], agent_type: Any
) -> None:
    payload = _simulation_payload(
        [_agent(agent_type, [resources[EnergyType.FOSSIL].id])]
    )
    response = _post_json(client, reverse("create_simulation"), payload)
    assert response.status_code == 400
    assert "type" in response.json()["errors"]
    assert Simulation.objects.count() == 0


def test_create_simulation_missing_type_key(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    agent = _agent(1, [resources[EnergyType.FOSSIL].id])
    del agent["type"]
    response = _post_json(
        client, reverse("create_simulation"), _simulation_payload([agent])
    )
    assert response.status_code == 400
    assert "type" in response.json()["errors"]


def test_create_simulation_unknown_resource(client: Client) -> None:
    response = _post_json(
        client, reverse("create_simulation"), _simulation_payload([_agent(1, [999999])])
    )
    assert response.status_code == 400
    assert "999999" in response.json()["errors"]["plants"][0]
    assert Simulation.objects.count() == 0


@pytest.mark.parametrize("bad_id", ["", None, "x"])
def test_create_simulation_unselected_resource(client: Client, bad_id: Any) -> None:
    response = _post_json(
        client, reverse("create_simulation"), _simulation_payload([_agent(1, [bad_id])])
    )
    assert response.status_code == 400
    assert "plants" in response.json()["errors"]


def test_create_simulation_empty_plants(client: Client) -> None:
    response = _post_json(
        client, reverse("create_simulation"), _simulation_payload([_agent(1, [])])
    )
    assert response.status_code == 400
    assert "At least one plant is required" in response.json()["errors"]["plants"][0]
    assert Simulation.objects.count() == 0


def test_create_simulation_get_renders_rule_and_energy_types(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    response = client.get(reverse("create_simulation"))
    assert response.status_code == 200
    content = response.content.decode()
    assert 'id="allowed-energy-types"' in content
    assert '"3": [0, 1, 2]' in content
    for energy_type, resource in resources.items():
        assert (
            f'value="{resource.id}" data-energy-type="{energy_type.value}"' in content
        )


# update_simulation


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_update_simulation_matrix(
    client: Client,
    resources: dict[EnergyType, Resource],
    agent_type: AgentType,
    energy_type: EnergyType,
    allowed: bool,
) -> None:
    simulation = _proxy_simulation()
    original = make_agent_with_plants(
        simulation.market,
        [(resources[EnergyType.FOSSIL], 7)],
        name="original",
        agent_type=AgentType.FOSSIL,
    )
    payload = _simulation_payload(
        [_agent(agent_type.value, [resources[energy_type].id], "new")]
    )
    response = _post_json(
        client, reverse("update_simulation", args=[simulation.id]), payload
    )
    simulation.refresh_from_db()
    if allowed:
        assert response.status_code == 200
        assert simulation.name == "rf-sim"
        assert list(Agent.objects.values_list("name", flat=True)) == ["new"]
    else:
        assert response.status_code == 400
        assert agent_type.name in response.json()["errors"]["plants"][0]
        assert simulation.name == "proxy-sim"
        assert list(Agent.objects.values_list("id", flat=True)) == [original.id]
        assert Plant.objects.get().resource == resources[EnergyType.FOSSIL]


def test_update_simulation_get_renders_rule(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    simulation = _proxy_simulation()
    make_agent_with_plants(
        simulation.market,
        [(resources[EnergyType.NUCLEAR], 1)],
        agent_type=AgentType.NUCLEAR,
    )
    response = client.get(reverse("update_simulation", args=[simulation.id]))
    assert response.status_code == 200
    content = response.content.decode()
    assert 'id="allowed-energy-types"' in content
    assert f'data-energy-type="{EnergyType.NUCLEAR.value}"' in content


def test_update_simulation_non_proxy_is_rejected(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    simulation = make_simulation()
    make_market(simulation=simulation)
    payload = _simulation_payload([_agent(1, [resources[EnergyType.FOSSIL].id])])
    response = _post_json(
        client, reverse("update_simulation", args=[simulation.id]), payload
    )
    assert response.status_code == 400


# create_agent_and_related


def _agent_form_data(
    agent_type: Any, resource_ids: list[Any], name: str = "lib"
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": name,
        "budget": "100",
        "proxy": "True",
        "resources[]": [str(rid) for rid in resource_ids],
        "capacities[]": ["10"] * len(resource_ids),
    }
    if agent_type is not None:
        data["type"] = str(agent_type)
    return data


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_create_agent_matrix(
    client: Client,
    resources: dict[EnergyType, Resource],
    agent_type: AgentType,
    energy_type: EnergyType,
    allowed: bool,
) -> None:
    response = client.post(
        reverse("agent_create"),
        _agent_form_data(agent_type.value, [resources[energy_type].id]),
    )
    if allowed:
        assert response.status_code == 302
        agent = Agent.objects.get()
        assert agent.type == agent_type
        assert agent.portfolio.plant_set.get().resource == resources[energy_type]
    else:
        assert response.status_code == 400
        messages = response.json()["errors"]["plants"]
        assert (
            agent_type.name in messages[0]
            and resources[energy_type].name in messages[0]
        )
        assert Agent.objects.count() == 0
        assert Portfolio.objects.count() == 0
        assert Plant.objects.count() == 0


def test_create_agent_missing_type(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    response = client.post(
        reverse("agent_create"),
        _agent_form_data(None, [resources[EnergyType.FOSSIL].id]),
    )
    assert response.status_code == 400
    assert "type" in response.json()["errors"]
    assert Agent.objects.count() == 0


def test_create_agent_unknown_resource(client: Client) -> None:
    response = client.post(reverse("agent_create"), _agent_form_data(1, [999999]))
    assert response.status_code == 400
    assert "plants" in response.json()["errors"]
    assert Agent.objects.count() == 0


def test_create_agent_without_plants(client: Client) -> None:
    response = client.post(reverse("agent_create"), _agent_form_data(1, []))
    assert response.status_code == 400
    assert "At least one plant is required" in response.json()["errors"]["plants"][0]
    assert Agent.objects.count() == 0


def test_create_agent_get_is_bad_request(client: Client) -> None:
    assert client.get(reverse("agent_create")).status_code == 400


# update_agent


@pytest.mark.parametrize("agent_type,energy_type,allowed", MATRIX)
def test_update_agent_matrix(
    client: Client,
    resources: dict[EnergyType, Resource],
    agent_type: AgentType,
    energy_type: EnergyType,
    allowed: bool,
) -> None:
    agent = make_agent_with_plants(
        make_market(),
        [(resources[EnergyType.FOSSIL], 7)],
        name="lib-agent",
        agent_type=AgentType.FOSSIL,
    )
    agent.proxy = True
    agent.save()
    response = client.post(
        reverse("agent_update", args=[agent.id]),
        _agent_form_data(agent_type.value, [resources[energy_type].id], name="renamed"),
    )
    agent.refresh_from_db()
    if allowed:
        assert response.status_code == 302
        assert agent.type == agent_type
        assert agent.name == "renamed"
        assert agent.portfolio.plant_set.get().resource == resources[energy_type]
    else:
        assert response.status_code == 400
        assert agent_type.name in response.json()["errors"]["plants"][0]
        assert agent.type == AgentType.FOSSIL
        assert agent.name == "lib-agent"
        plant = agent.portfolio.plant_set.get()
        assert (plant.resource, plant.capacity) == (resources[EnergyType.FOSSIL], 7)


def test_agent_list_renders_rule(
    client: Client, resources: dict[EnergyType, Resource]
) -> None:
    agent = make_agent_with_plants(
        make_market(),
        [(resources[EnergyType.RENEWABLE], 1)],
        agent_type=AgentType.RENEWABLE,
    )
    agent.proxy = True
    agent.save()
    response = client.get(reverse("agent_list"))
    assert response.status_code == 200
    content = response.content.decode()
    assert content.count('id="allowed-energy-types"') == 1
    assert f'data-energy-type="{EnergyType.RENEWABLE.value}"' in content
