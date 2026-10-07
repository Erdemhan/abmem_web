"""View tests of the agent library page and endpoints (spec agent_library, package ui).

Covers the agent list (library agents only), the live name check endpoint,
the library JSON for the "select from library" dialog, and the create /
update / delete flows of library agents, including the database constraint
race mapped to HTTP 400.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem import forms as abmem_forms
from abmem.models import Agent, Market
from abmem.models.enums import AgentType, AlgorithmType, EnergyType
from abmem.models.models import Plant, Resource
from abmem.services.agent import library_service
from abmem.services.agent.library_service import AgentInput, PlantInput
from tests.factories import make_market, make_resource

pytestmark = pytest.mark.django_db

JSON_HEADERS: dict[str, str] = {"HTTP_ACCEPT": "application/json"}


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="lib-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


@pytest.fixture
def renewable() -> Resource:
    """One saved renewable resource."""
    return make_resource(name="lib-wind", energy_type=EnergyType.RENEWABLE)


def _library_agent(resource: Resource, name: str = "Alpha", **overrides: Any) -> Agent:
    values: dict[str, Any] = {
        "name": name,
        "budget": Decimal("100"),
        "type": AgentType.FOSSIL,
        "algorithm_type": AlgorithmType.TD3,
        "failstack_enabled": True,
        "failstack_limit": 3,
        "failstack_delta": Decimal("5"),
        "failstack_min_margin": Decimal("0.1"),
        "plants": (PlantInput(resource.id, 50),),
    }
    values.update(overrides)
    return library_service.create_library_agent(AgentInput(**values))


def _scenario_market() -> Market:
    market = make_market()
    market.simulation.proxy = True
    market.simulation.save()
    market.proxy = True
    market.save()
    return market


def _form_data(resource: Resource, name: str = "Beta", **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": name,
        "budget": "250",
        "type": str(AgentType.FOSSIL.value),
        "algorithm_type": str(AlgorithmType.SAC.value),
        "failstack_enabled": "on",
        "failstack_limit": "4",
        "failstack_delta": "7.5",
        "failstack_min_margin": "0.2",
        "resources[]": [str(resource.id)],
        "capacities[]": ["120"],
    }
    data.update(extra)
    return data


def _plants(agent: Agent) -> list[tuple[int, int]]:
    return sorted(
        Plant.objects.filter(portfolio__agent=agent).values_list(
            "resource_id", "capacity"
        )
    )


# agent list


def test_agent_list_shows_only_library_agents(client: Client, fossil: Resource) -> None:
    library_agent = _library_agent(fossil, name="LibOnlyAgent")
    copy = library_service.copy_agent(
        library_agent, _scenario_market(), proxy=True, name="ScenarioCopyXYZ"
    )
    response = client.get(reverse("agent_list"))
    assert response.status_code == 200
    assert list(response.context["agents"]) == [library_agent]
    html = response.content.decode()
    assert "LibOnlyAgent" in html
    assert copy.name not in html
    assert f'id="pen-{library_agent.id}"' in html
    assert f'id="pen-{copy.id}"' not in html


def test_agent_list_renders_form_urls(client: Client, fossil: Resource) -> None:
    agent = _library_agent(fossil, name="O'Brien")
    html = client.get(reverse("agent_list")).content.decode()
    assert f'data-check-name-url="{reverse("agent_check_name")}"' in html
    assert f'action="{reverse("agent_create")}"' in html
    assert reverse("agent_update", args=[agent.id]) in html
    assert reverse("agent_delete", args=[agent.id]) in html
    # names are escaped for the inline JavaScript of the edit button
    assert "O\\u0027Brien" in html
    assert "localStorage" not in html


# check-name


@pytest.mark.parametrize("variant", ["Alpha", "alpha", "ALPHA", "  alpha  "])
def test_check_name_rejects_taken_name_case_insensitively(
    client: Client, fossil: Resource, variant: str
) -> None:
    _library_agent(fossil, name="Alpha")
    response = client.get(reverse("agent_check_name"), {"name": variant})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert "already exists" in body["message"]


def test_check_name_free_name(client: Client, fossil: Resource) -> None:
    _library_agent(fossil, name="Alpha")
    body = client.get(reverse("agent_check_name"), {"name": "Gamma"}).json()
    assert body == {"available": True, "message": ""}


def test_check_name_ignores_scenario_agents(client: Client, fossil: Resource) -> None:
    source = _library_agent(fossil, name="Alpha")
    library_service.copy_agent(source, _scenario_market(), proxy=True, name="Delta")
    body = client.get(reverse("agent_check_name"), {"name": "delta"}).json()
    assert body["available"] is True


@pytest.mark.parametrize("name", ["", "   ", None])
def test_check_name_empty(client: Client, name: Any) -> None:
    params = {} if name is None else {"name": name}
    body = client.get(reverse("agent_check_name"), params).json()
    assert body["available"] is False
    assert body["message"] == "Agent name is required."


def test_check_name_too_long(client: Client) -> None:
    body = client.get(reverse("agent_check_name"), {"name": "x" * 21}).json()
    assert body["available"] is False
    assert "at most 20 characters" in body["message"]
    assert client.get(reverse("agent_check_name"), {"name": "x" * 20}).json()[
        "available"
    ]


def test_check_name_exclude_id(client: Client, fossil: Resource) -> None:
    agent = _library_agent(fossil, name="Alpha")
    body = client.get(
        reverse("agent_check_name"), {"name": "ALPHA", "exclude_id": agent.id}
    ).json()
    assert body["available"] is True


@pytest.mark.parametrize("exclude_id", ["abc", "-1", "0", "1.5"])
def test_check_name_bad_exclude_id(client: Client, exclude_id: str) -> None:
    response = client.get(
        reverse("agent_check_name"), {"name": "x", "exclude_id": exclude_id}
    )
    assert response.status_code == 400
    assert response.json()["available"] is False


def test_check_name_rejects_post(client: Client) -> None:
    assert client.post(reverse("agent_check_name"), {"name": "x"}).status_code == 405


# library.json


def test_library_json_lists_library_agents_only(
    client: Client, fossil: Resource
) -> None:
    beta = _library_agent(fossil, name="beta", algorithm_type=AlgorithmType.RANDOM)
    alpha = _library_agent(fossil, name="Alpha")
    library_service.copy_agent(alpha, _scenario_market(), proxy=True, name="Copy")
    response = client.get(reverse("agent_library_json"))
    assert response.status_code == 200
    body = response.json()
    assert [item["id"] for item in body] == [alpha.id, beta.id]
    assert body[0] == {
        "id": alpha.id,
        "name": "Alpha",
        "type": AgentType.FOSSIL.value,
        "type_name": "FOSSIL",
        "budget": "100.00",
        "algorithm_type": AlgorithmType.TD3.value,
        "algorithm_name": "TD3",
        "failstack": {
            "enabled": True,
            "active": True,
            "limit": 3,
            "delta": "5.00",
            "min_margin": "0.1000",
        },
        "plants": [
            {
                "resource_id": fossil.id,
                "resource_name": "lib-gas",
                "energy_type": EnergyType.FOSSIL.value,
                "capacity": 50,
            }
        ],
    }
    # a baseline keeps the stored switch but the rule is not active
    assert body[1]["failstack"]["enabled"] is True
    assert body[1]["failstack"]["active"] is False


def test_library_json_empty(client: Client) -> None:
    assert client.get(reverse("agent_library_json")).json() == []


# create


def test_create_library_agent_form_post(client: Client, fossil: Resource) -> None:
    response = client.post(reverse("agent_create"), _form_data(fossil, name="  Beta "))
    assert response.status_code == 302
    assert response.url == reverse("agent_list")
    agent = Agent.objects.get()
    assert agent.is_library
    assert agent.name == "Beta"
    assert agent.algorithm_type == AlgorithmType.SAC
    assert (agent.failstack_enabled, agent.failstack_limit) == (True, 4)
    assert agent.failstack_delta == Decimal("7.5")
    assert _plants(agent) == [(fossil.id, 120)]


def test_create_library_agent_json_client(client: Client, fossil: Resource) -> None:
    response = client.post(reverse("agent_create"), _form_data(fossil), **JSON_HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    agent = Agent.objects.get()
    assert body["agent"]["id"] == agent.id
    assert body["agent"]["plants"][0]["capacity"] == 120


@pytest.mark.parametrize("variant", ["Alpha", "alpha", " ALPHA "])
def test_create_duplicate_name_rejected(
    client: Client, fossil: Resource, variant: str
) -> None:
    _library_agent(fossil, name="Alpha")
    response = client.post(reverse("agent_create"), _form_data(fossil, name=variant))
    assert response.status_code == 400
    assert "already exists" in response.json()["errors"]["name"][0]
    assert Agent.objects.count() == 1


def test_create_name_too_long_rejected(client: Client, fossil: Resource) -> None:
    response = client.post(reverse("agent_create"), _form_data(fossil, name="x" * 21))
    assert response.status_code == 400
    assert "name" in response.json()["errors"]
    assert not Agent.objects.exists()


def test_create_name_race_maps_integrity_error_to_400(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both pre-checks miss the existing name, the DB constraint catches it."""
    _library_agent(fossil, name="Alpha")
    real_check = library_service.check_name_available
    calls = {"n": 0}

    def stale_check(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            return True  # the service pre-check runs before the competing insert
        return real_check(*args, **kwargs)

    monkeypatch.setattr(abmem_forms, "check_name_available", lambda *a, **k: True)
    monkeypatch.setattr(library_service, "check_name_available", stale_check)
    response = client.post(reverse("agent_create"), _form_data(fossil, name="alpha"))
    assert response.status_code == 400
    assert "already exists" in response.json()["errors"]["name"][0]
    assert calls["n"] == 2  # pre-check plus the conversion after IntegrityError
    assert Agent.objects.count() == 1


def test_create_rejects_disallowed_resource(
    client: Client, fossil: Resource, renewable: Resource
) -> None:
    response = client.post(
        reverse("agent_create"),
        _form_data(fossil, **{"resources[]": [str(renewable.id)]}),
    )
    assert response.status_code == 400
    assert "plants" in response.json()["errors"]
    assert not Agent.objects.exists()


def test_create_rejects_zero_capacity(client: Client, fossil: Resource) -> None:
    response = client.post(
        reverse("agent_create"), _form_data(fossil, **{"capacities[]": ["0"]})
    )
    assert response.status_code == 400
    assert "plants" in response.json()["errors"]
    assert not Agent.objects.exists()


# update


def test_update_library_agent_leaves_copies_unchanged(
    client: Client, fossil: Resource, renewable: Resource
) -> None:
    agent = _library_agent(fossil, name="Alpha", type=AgentType.HYBRID)
    copy = library_service.copy_agent(agent, _scenario_market(), proxy=True)
    response = client.post(
        reverse("agent_update", args=[agent.id]),
        _form_data(
            fossil,
            name="Alpha2",
            type=str(AgentType.HYBRID.value),
            algorithm_type=str(AlgorithmType.PPO.value),
            **{"resources[]": [str(renewable.id)], "capacities[]": ["9"]},
        ),
        **JSON_HEADERS,
    )
    assert response.status_code == 200, response.content
    agent.refresh_from_db()
    copy.refresh_from_db()
    assert (agent.name, agent.algorithm_type) == ("Alpha2", AlgorithmType.PPO)
    assert _plants(agent) == [(renewable.id, 9)]
    assert (copy.name, copy.algorithm_type) == ("Alpha", AlgorithmType.TD3)
    assert _plants(copy) == [(fossil.id, 50)]
    assert copy.source_id == agent.id


def test_update_may_change_case_of_own_name(client: Client, fossil: Resource) -> None:
    agent = _library_agent(fossil, name="Alpha")
    response = client.post(
        reverse("agent_update", args=[agent.id]), _form_data(fossil, name="ALPHA")
    )
    assert response.status_code == 302
    agent.refresh_from_db()
    assert agent.name == "ALPHA"


def test_update_to_other_library_name_rejected(
    client: Client, fossil: Resource
) -> None:
    _library_agent(fossil, name="Alpha")
    beta = _library_agent(fossil, name="Beta")
    response = client.post(
        reverse("agent_update", args=[beta.id]), _form_data(fossil, name="alpha")
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["errors"]["name"][0]
    beta.refresh_from_db()
    assert beta.name == "Beta"


def test_update_non_library_agent_rejected(client: Client, fossil: Resource) -> None:
    source = _library_agent(fossil, name="Alpha")
    copy = library_service.copy_agent(source, _scenario_market(), proxy=True)
    response = client.post(
        reverse("agent_update", args=[copy.id]), _form_data(fossil, name="New")
    )
    assert response.status_code == 400
    copy.refresh_from_db()
    assert copy.name == "Alpha"


def test_update_get_is_bad_request(client: Client, fossil: Resource) -> None:
    agent = _library_agent(fossil)
    assert client.get(reverse("agent_update", args=[agent.id])).status_code == 400


# delete


def test_delete_library_agent_used_only_by_runs_keeps_run_copies(
    client: Client, fossil: Resource
) -> None:
    agent = _library_agent(fossil, name="Alpha")
    run_copy = library_service.copy_agent(agent, make_market(), proxy=False)
    response = client.post(reverse("agent_delete", args=[agent.id]))
    assert response.status_code == 302
    assert not Agent.objects.filter(id=agent.id).exists()
    run_copy.refresh_from_db()
    assert run_copy.source_id is None
    assert _plants(run_copy) == [(fossil.id, 50)]


def test_delete_library_agent_used_by_a_scenario_is_blocked(
    client: Client, fossil: Resource
) -> None:
    agent = _library_agent(fossil, name="Alpha")
    market = _scenario_market()
    market.simulation.name = "Uses Alpha"
    market.simulation.save()
    copy = library_service.copy_agent(agent, market, proxy=True)

    response = client.post(
        reverse("agent_delete", args=[agent.id]), HTTP_ACCEPT="application/json"
    )
    assert response.status_code == 400
    messages = response.json()["errors"]["agent"]
    assert any("Uses Alpha" in message for message in messages)
    assert Agent.objects.filter(id=agent.id).exists()
    copy.refresh_from_db()
    assert copy.source_id == agent.id

    form_response = client.post(reverse("agent_delete", args=[agent.id]))
    assert form_response.status_code == 400
    assert Agent.objects.filter(id=agent.id).exists()

    market.simulation.delete()
    assert client.post(reverse("agent_delete", args=[agent.id])).status_code == 302
    assert not Agent.objects.filter(id=agent.id).exists()


def test_scenarios_using_lists_each_scenario_once(fossil: Resource) -> None:
    agent = _library_agent(fossil, name="Alpha")
    first, second = _scenario_market(), _scenario_market()
    library_service.copy_agent(agent, first, proxy=True)
    library_service.copy_agent(agent, second, proxy=True)
    library_service.copy_agent(agent, make_market(), proxy=False)
    used = library_service.scenarios_using(agent)
    assert sorted(s.id for s in used) == sorted([first.simulation_id, second.simulation_id])


def test_delete_non_library_agent_rejected(client: Client, fossil: Resource) -> None:
    source = _library_agent(fossil, name="Alpha")
    copy = library_service.copy_agent(source, _scenario_market(), proxy=True)
    assert client.post(reverse("agent_delete", args=[copy.id])).status_code == 400
    assert Agent.objects.filter(id=copy.id).exists()


def test_delete_get_is_bad_request(client: Client, fossil: Resource) -> None:
    agent = _library_agent(fossil)
    assert client.get(reverse("agent_delete", args=[agent.id])).status_code == 400
    assert Agent.objects.filter(id=agent.id).exists()
