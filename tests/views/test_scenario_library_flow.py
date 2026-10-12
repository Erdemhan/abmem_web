"""Scenario create/update/run flows with the agent library (spec agent_library, package ui).

Payload contract: ``agents[]`` entries with ``mode`` "new" (added to the
library, then copied), "library" (copy of a library agent) and, on update,
"scenario" (agent already in the scenario, edited in place). Errors are
HTTP 400 ``{"errors": {...}}``.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem import views
from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Plant, Resource
from abmem.services.agent import library_service
from abmem.services.agent.library_service import (
    COPIED_AGENT_FIELDS,
    AgentInput,
    PlantInput,
)
from tests.factories import make_agent_with_plants, make_market, make_resource
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="flow-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


@pytest.fixture
def coal() -> Resource:
    """A second saved fossil resource."""
    return make_resource(name="flow-coal", energy_type=EnergyType.FOSSIL, fuel_cost=30)


def _library_agent(resource: Resource, name: str = "LibA", **overrides: Any) -> Agent:
    values: dict[str, Any] = {
        "name": name,
        "budget": Decimal("500"),
        "type": AgentType.FOSSIL,
        "algorithm_type": "PPO",
        "failstack_enabled": True,
        "failstack_limit": 3,
        "failstack_delta": Decimal("10"),
        "failstack_min_margin": Decimal("0.05"),
        "plants": (PlantInput(resource.id, 300), PlantInput(resource.id, 120)),
        "algorithm_params": {"actor_lr": 0.001},
    }
    values.update(overrides)
    return library_service.create_library_agent(AgentInput(**values))


def _new_agent(resource: Resource, name: str = "NewB", **extra: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "mode": "new",
        "name": name,
        "type": AgentType.FOSSIL.value,
        "budget": "250",
        "algorithm_type": "TD3",
        "failstack": {"enabled": True, "limit": 5, "delta": "2.5", "min_margin": "0"},
        "plants": [{"resource": resource.id, "capacity": "200"}],
    }
    data.update(extra)
    return data


def _library_entry(agent: Agent) -> dict[str, Any]:
    return {"mode": "library", "library_agent_id": agent.id}


def _payload(agents: list[dict[str, Any]], name: str = "flow-sim") -> dict[str, Any]:
    return {
        "simulation_name": name,
        "simulationMode": "1",
        "periodNumber": "5",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": agents,
    }


def _post(client: Client, url: str, agents: list[dict[str, Any]], **kw: Any) -> Any:
    return client.post(
        url, json.dumps(_payload(agents, **kw)), content_type="application/json"
    )


def _create(client: Client, agents: list[dict[str, Any]]) -> Any:
    return _post(client, reverse("create_simulation"), agents)


def _plants(agent: Agent) -> list[tuple[int, int]]:
    return sorted(
        Plant.objects.filter(portfolio__agent=agent).values_list(
            "resource_id", "capacity"
        )
    )


def _scenario(client: Client, agents: list[dict[str, Any]]) -> Simulation:
    response = _create(client, agents)
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


def _assert_copy(copy: Agent, source: Agent) -> None:
    for field in COPIED_AGENT_FIELDS:
        assert getattr(copy, field) == getattr(source, field), field
    assert _plants(copy) == _plants(source)


# create


def test_create_with_new_and_library_agent(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(coal)
    simulation = _scenario(client, [_new_agent(fossil), _library_entry(lib_a)])

    assert simulation.proxy
    new_library = Agent.objects.get(name="NewB", market__isnull=True)
    assert new_library.is_library
    assert new_library.algorithm_type == "TD3"
    assert (new_library.failstack_enabled, new_library.failstack_limit) == (True, 5)
    assert _plants(new_library) == [(fossil.id, 200)]

    copies = list(simulation.market.agent_set.order_by("id"))
    assert [agent.name for agent in copies] == ["NewB", "LibA"]
    assert all(agent.proxy for agent in copies)
    assert [agent.source_id for agent in copies] == [new_library.id, lib_a.id]
    _assert_copy(copies[0], new_library)
    _assert_copy(copies[1], lib_a)
    assert Agent.objects.filter(market__isnull=True).count() == 2


def test_legacy_payload_without_mode_is_new(client: Client, fossil: Resource) -> None:
    entry = _new_agent(fossil, name="Legacy")
    del entry["mode"]
    simulation = _scenario(client, [entry])
    assert Agent.objects.filter(name="Legacy", market__isnull=True).count() == 1
    assert simulation.market.agent_set.get().source.name == "Legacy"


def test_same_library_agent_twice_gets_free_names(
    client: Client, fossil: Resource
) -> None:
    """OQ-14 (2026-10-12): repeated library entries without a name are saved
    as separate copies named NAME, NAME_2, ..."""
    lib_a = _library_agent(fossil)
    simulation = _scenario(
        client, [_library_entry(lib_a), _library_entry(lib_a), _library_entry(lib_a)]
    )
    copies = list(simulation.market.agent_set.order_by("id"))
    assert [agent.name for agent in copies] == ["LibA", "LibA_2", "LibA_3"]
    for agent in copies:
        assert agent.source_id == lib_a.id
        for field in COPIED_AGENT_FIELDS:
            if field != "name":
                assert getattr(agent, field) == getattr(lib_a, field), field
        assert _plants(agent) == _plants(lib_a)
    assert Agent.objects.filter(market__isnull=True).count() == 1


def test_auto_name_skips_names_given_in_the_payload(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(
        client,
        [
            _library_entry(lib_a),
            {**_library_entry(lib_a), "name": "liba_2"},
            _library_entry(lib_a),
        ],
    )
    names = [agent.name for agent in simulation.market.agent_set.order_by("id")]
    assert names == ["LibA", "liba_2", "LibA_3"]


def test_same_library_agent_twice_with_wizard_names(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(
        client,
        [
            {**_library_entry(lib_a), "name": "LibA"},
            {**_library_entry(lib_a), "name": "LibA_2"},
        ],
    )
    copies = list(simulation.market.agent_set.order_by("id"))
    assert [agent.name for agent in copies] == ["LibA", "LibA_2"]
    assert [agent.source_id for agent in copies] == [lib_a.id, lib_a.id]


def test_explicit_duplicate_library_copy_names_are_rejected(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    response = _create(
        client,
        [
            {**_library_entry(lib_a), "name": "Twin"},
            {**_library_entry(lib_a), "name": "Twin"},
        ],
    )
    assert response.status_code == 400
    assert "agents[1].name" in response.json()["errors"]
    assert not Simulation.objects.exists()


@pytest.mark.parametrize("name", ["", "   ", "x" * 21, 5])
def test_invalid_library_copy_name_is_rejected(
    client: Client, fossil: Resource, name: Any
) -> None:
    lib_a = _library_agent(fossil)
    response = _create(client, [{**_library_entry(lib_a), "name": name}])
    assert response.status_code == 400
    assert "agents[0].name" in response.json()["errors"]
    assert not Simulation.objects.exists()


def _edited_library_entry(agent: Agent, resource: Resource, **extra: Any) -> dict[str, Any]:
    entry = _new_agent(resource, name="Edited", mode="library", budget="999")
    entry["library_agent_id"] = agent.id
    entry.update(extra)
    return entry


def test_edited_library_entry_is_saved_with_the_edited_values(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(coal)
    library_before = (
        {field: getattr(lib_a, field) for field in COPIED_AGENT_FIELDS},
        _plants(lib_a),
    )
    simulation = _scenario(
        client, [_library_entry(lib_a), _edited_library_entry(lib_a, fossil)]
    )
    unedited, edited = simulation.market.agent_set.order_by("id")
    _assert_copy(unedited, lib_a)
    assert edited.name == "Edited"
    assert edited.source_id == lib_a.id
    assert edited.proxy is True
    assert edited.budget == Decimal("999")
    assert edited.algorithm_type == "TD3"
    assert (edited.failstack_limit, edited.failstack_delta) == (5, Decimal("2.5"))
    assert _plants(edited) == [(fossil.id, 200)]
    # PPO parameters of the library agent do not apply to TD3, they are cleared.
    assert edited.algorithm_params == {}
    lib_a.refresh_from_db()
    assert (
        {field: getattr(lib_a, field) for field in COPIED_AGENT_FIELDS},
        _plants(lib_a),
    ) == library_before
    assert Agent.objects.filter(market__isnull=True).count() == 1


def test_edited_library_entry_keeps_params_for_the_same_algorithm(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(coal)
    simulation = _scenario(
        client, [_edited_library_entry(lib_a, fossil, algorithm_type="PPO")]
    )
    (edited,) = simulation.market.agent_set.all()
    assert edited.algorithm_type == "PPO"
    assert edited.algorithm_params == lib_a.algorithm_params


@pytest.mark.parametrize(
    ("extra", "key"),
    [
        ({"budget": "abc"}, "budget"),
        ({"plants": []}, "plants"),
        ({"algorithm_type": "NOPE"}, "algorithm_type"),
        ({"type": None}, "type"),
    ],
)
def test_invalid_edited_library_entry_is_rejected(
    client: Client, fossil: Resource, extra: dict[str, Any], key: str
) -> None:
    lib_a = _library_agent(fossil)
    response = _create(client, [_edited_library_entry(lib_a, fossil, **extra)])
    assert response.status_code == 400, response.content
    assert key in response.json()["errors"]
    assert not Simulation.objects.exists()


def test_run_of_scenario_with_two_copies_builds(
    client: Client, fossil: Resource
) -> None:
    from abmem.services.simulation.context import SimulationContext
    from tests.runtime.conftest import constant_demand

    lib_a = _library_agent(fossil, algorithm_params={})
    simulation = _scenario(
        client,
        [
            _library_entry(lib_a),
            _library_entry(lib_a),
            _edited_library_entry(lib_a, fossil),
        ],
    )
    run = library_service.clone_scenario_for_run(simulation)
    names = sorted(agent.name for agent in run.market.agent_set.all())
    assert names == ["Edited", "LibA", "LibA_2"]
    assert {agent.source_id for agent in run.market.agent_set.all()} == {lib_a.id}
    ctx = SimulationContext.build(run, demand=constant_demand(), export_json=False)
    assert len(ctx.agents) == 3


@pytest.mark.parametrize("raw_id", [999999, "abc", None, True, 0])
def test_unknown_library_agent_is_rejected(
    client: Client, fossil: Resource, raw_id: Any
) -> None:
    response = _create(client, [{"mode": "library", "library_agent_id": raw_id}])
    assert response.status_code == 400
    assert "agents[0].library_agent_id" in response.json()["errors"]
    assert not Simulation.objects.exists()


def test_scenario_agent_cannot_be_used_as_library_agent(
    client: Client, fossil: Resource
) -> None:
    simulation = _scenario(client, [_new_agent(fossil)])
    copy = simulation.market.agent_set.get()
    response = _create(client, [_library_entry(copy)])
    assert response.status_code == 400
    assert "agents[0].library_agent_id" in response.json()["errors"]
    assert Simulation.objects.count() == 1


@pytest.mark.parametrize("variant", ["LibA", "liba", "  LIBA "])
def test_new_agent_with_taken_library_name_is_rejected(
    client: Client, fossil: Resource, variant: str
) -> None:
    _library_agent(fossil)
    response = _create(client, [_new_agent(fossil, name=variant)])
    assert response.status_code == 400
    messages = response.json()["errors"]["agents[0].name"]
    assert "already exists" in messages[0]
    assert "Agent 1" in messages[0]
    assert not Simulation.objects.exists()
    assert Agent.objects.count() == 1


def test_two_new_agents_with_same_name_are_rejected(
    client: Client, fossil: Resource
) -> None:
    response = _create(
        client, [_new_agent(fossil, name="Twin"), _new_agent(fossil, name="twin")]
    )
    assert response.status_code == 400
    assert "agents[1].name" in response.json()["errors"]
    assert not Agent.objects.exists()


def test_new_agent_name_equal_to_library_pick_is_rejected(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    response = _create(client, [_library_entry(lib_a), _new_agent(fossil, name="LibA")])
    assert response.status_code == 400
    assert "agents[1].name" in response.json()["errors"]
    assert not Simulation.objects.exists()


@pytest.mark.parametrize("name", ["", "   ", None, "x" * 21])
def test_invalid_new_agent_name_is_rejected(
    client: Client, fossil: Resource, name: Any
) -> None:
    response = _create(client, [_new_agent(fossil, name=name)])
    assert response.status_code == 400
    assert "agents[0].name" in response.json()["errors"]
    assert not Agent.objects.exists()


@pytest.mark.parametrize("capacity", ["0", "-5", "1.5", "abc", None])
def test_invalid_capacity_is_rejected(
    client: Client, fossil: Resource, capacity: Any
) -> None:
    entry = _new_agent(fossil, plants=[{"resource": fossil.id, "capacity": capacity}])
    response = _create(client, [entry])
    assert response.status_code == 400
    assert (
        "capacity must be a positive integer" in response.json()["errors"]["plants"][0]
    )
    assert not Agent.objects.exists()


@pytest.mark.parametrize("budget", ["", None, "abc", "NaN"])
def test_invalid_budget_is_rejected(
    client: Client, fossil: Resource, budget: Any
) -> None:
    response = _create(client, [_new_agent(fossil, budget=budget)])
    assert response.status_code == 400
    assert "budget" in response.json()["errors"]
    assert not Agent.objects.exists()


@pytest.mark.parametrize("mode", ["scenario", "other"])
def test_unknown_mode_on_create_is_rejected(
    client: Client, fossil: Resource, mode: str
) -> None:
    response = _create(client, [_new_agent(fossil, mode=mode)])
    assert response.status_code == 400
    assert "agents[0].mode" in response.json()["errors"]


def test_write_race_is_mapped_to_400(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A name taken between validation and write rolls everything back."""

    def racing_add(data: AgentInput, market: Any) -> Agent:
        _library_agent(fossil, name=data.name)  # competing insert
        return library_service.add_new_agent_to_scenario(data, market)

    monkeypatch.setattr(views.library_service, "add_new_agent_to_scenario", racing_add)
    response = _create(client, [_new_agent(fossil, name="Racy")])
    assert response.status_code == 400
    assert "already exists" in response.json()["errors"]["agents[0].name"][0]
    assert not Simulation.objects.exists()
    assert not Agent.objects.exists()


# library edits do not change scenarios


def test_editing_library_agent_leaves_scenario_unchanged(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a)])
    copy = simulation.market.agent_set.get()
    response = client.post(
        reverse("agent_update", args=[lib_a.id]),
        {
            "name": "LibA-v2",
            "budget": "1",
            "type": str(AgentType.FOSSIL.value),
            "algorithm_type": "RANDOM",
            "resources[]": [str(coal.id)],
            "capacities[]": ["7"],
        },
    )
    assert response.status_code == 302, response.content
    copy.refresh_from_db()
    assert copy.name == "LibA"
    assert copy.algorithm_type == "PPO"
    assert copy.budget == Decimal("500")
    assert _plants(copy) == [(fossil.id, 120), (fossil.id, 300)]


# update


def test_update_keeps_edits_adds_and_removes(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(fossil)
    lib_c = _library_agent(coal, name="LibC", algorithm_type="SAC")
    simulation = _scenario(
        client, [_library_entry(lib_a), _new_agent(fossil, name="Gone")]
    )
    kept = simulation.market.agent_set.get(name="LibA")
    removed = simulation.market.agent_set.get(name="Gone")

    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [
            {
                "mode": "scenario",
                "agent_id": kept.id,
                "name": "LibA-edited",
                "type": AgentType.FOSSIL.value,
                "budget": "42",
                "algorithm_type": "RDPG",
                "failstack": {
                    "enabled": False,
                    "limit": 2,
                    "delta": 1,
                    "min_margin": 0,
                },
                "plants": [{"resource": coal.id, "capacity": 11}],
            },
            _library_entry(lib_c),
            _new_agent(fossil, name="Fresh"),
        ],
        name="flow-sim-v2",
    )
    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert simulation.name == "flow-sim-v2"

    agents = {agent.name: agent for agent in simulation.market.agent_set.all()}
    assert set(agents) == {"LibA-edited", "LibC", "Fresh"}
    edited = agents["LibA-edited"]
    assert edited.id == kept.id
    assert edited.source_id == lib_a.id
    assert edited.algorithm_type == "RDPG"
    assert (edited.failstack_enabled, edited.failstack_limit) == (False, 2)
    assert edited.budget == Decimal("42")
    assert _plants(edited) == [(coal.id, 11)]
    # The algorithm changed from PPO to RDPG, so the PPO parameters are cleared.
    assert edited.algorithm_params == {}
    _assert_copy(agents["LibC"], lib_c)
    assert agents["Fresh"].source == Agent.objects.get(
        name="Fresh", market__isnull=True
    )
    assert not Agent.objects.filter(id=removed.id).exists()
    # the library agent the "Gone" copy came from stays in the library
    assert Agent.objects.filter(name="Gone", market__isnull=True).exists()

    # scenario edits never change the library
    lib_a.refresh_from_db()
    assert (lib_a.name, lib_a.algorithm_type) == ("LibA", "PPO")
    assert _plants(lib_a) == [(fossil.id, 120), (fossil.id, 300)]


def test_update_adds_a_second_copy_next_to_a_kept_one(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a)])
    kept = simulation.market.agent_set.get()
    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [
            _new_agent(fossil, name=kept.name, mode="scenario", agent_id=kept.id),
            _library_entry(lib_a),
        ],
    )
    assert response.status_code == 200, response.content
    agents = list(simulation.market.agent_set.order_by("id"))
    assert [agent.name for agent in agents] == ["LibA", "LibA_2"]
    assert agents[0].id == kept.id
    assert [agent.source_id for agent in agents] == [lib_a.id, lib_a.id]


def test_update_with_same_library_agent_twice(client: Client, fossil: Resource) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_new_agent(fossil)])
    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [
            {**_library_entry(lib_a), "name": "LibA"},
            {**_library_entry(lib_a), "name": "LibA_2"},
        ],
    )
    assert response.status_code == 200, response.content
    agents = list(simulation.market.agent_set.order_by("id"))
    assert [agent.name for agent in agents] == ["LibA", "LibA_2"]
    assert [agent.source_id for agent in agents] == [lib_a.id, lib_a.id]


def test_update_can_replace_a_copy_by_a_fresh_library_copy(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a)])
    old_copy = simulation.market.agent_set.get()
    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [_library_entry(lib_a)],
    )
    assert response.status_code == 200, response.content
    new_copy = simulation.market.agent_set.get()
    assert new_copy.id != old_copy.id
    _assert_copy(new_copy, lib_a)


def test_update_rejects_agent_of_another_scenario(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    other = _scenario(client, [_library_entry(lib_a)])
    simulation = _scenario(client, [_new_agent(fossil)])
    foreign = other.market.agent_set.get()
    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [_new_agent(fossil, name="X", mode="scenario", agent_id=foreign.id)],
    )
    assert response.status_code == 400
    assert "agents[0].agent_id" in response.json()["errors"]
    assert simulation.market.agent_set.get().name == "NewB"


def test_update_rejects_duplicate_names_in_scenario(
    client: Client, fossil: Resource
) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a), _new_agent(fossil)])
    a = simulation.market.agent_set.get(name="LibA")
    b = simulation.market.agent_set.get(name="NewB")
    response = _post(
        client,
        reverse("update_simulation", args=[simulation.id]),
        [
            _new_agent(fossil, name="Same", mode="scenario", agent_id=a.id),
            _new_agent(fossil, name="Same", mode="scenario", agent_id=b.id),
        ],
    )
    assert response.status_code == 400
    assert "agents[1].name" in response.json()["errors"]
    assert sorted(simulation.market.agent_set.values_list("name", flat=True)) == [
        "LibA",
        "NewB",
    ]


def test_update_rejects_non_proxy(client: Client, fossil: Resource) -> None:
    market = make_market()
    make_agent_with_plants(market, [(fossil, 1)], name="run-agent")
    response = _post(
        client, reverse("update_simulation", args=[market.simulation.id]), []
    )
    assert response.status_code == 400


def test_update_page_renders_scenario_rows(client: Client, fossil: Resource) -> None:
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a)])
    copy = simulation.market.agent_set.get()
    html = client.get(
        reverse("update_simulation", args=[simulation.id])
    ).content.decode()
    # The wizard renders scenario agents from JSON as "scenario" rows (PLAN 2.6).
    config = wizard_config(html)
    (row,) = config["initial"]["agents"]
    assert (row["mode"], row["agent_id"], row["source_id"], row["source_name"]) == (
        "scenario",
        copy.id,
        lib_a.id,
        "LibA",
    )
    assert "Copy of library agent " in html
    assert 'id="library-cards"' in html
    assert config["library_url"] == reverse("agent_library_json")
    assert reverse("agent_check_name") in html


def test_create_page_renders_library_dialog(client: Client) -> None:
    html = client.get(reverse("create_simulation")).content.decode()
    assert 'id="library-cards"' in html
    assert 'id="library-search"' in html and 'id="library-type-filter"' in html
    assert "addAgent()" not in html
    assert wizard_config(html)["library_url"] == reverse("agent_library_json")
    assert reverse("agent_check_name") in html


# run


def test_run_uses_clone_scenario_for_run(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[int] = []

    def fake_start(simulation: Simulation) -> int:
        started.append(simulation.id)
        return simulation.id

    monkeypatch.setattr(views.StarterService, "start", fake_start)
    lib_a = _library_agent(fossil)
    simulation = _scenario(client, [_library_entry(lib_a), _new_agent(fossil)])

    response = client.get(reverse("run_simulation", args=[simulation.id]))

    run = Simulation.objects.get(proxy=False)
    assert response.status_code == 302
    assert response.url == reverse("watch_simulation", kwargs={"simulation_id": run.id})
    assert started == [run.id]
    assert run.currentPeriod == -1
    scenario_agents = {a.name: a for a in simulation.market.agent_set.all()}
    run_agents = {a.name: a for a in run.market.agent_set.all()}
    assert set(run_agents) == set(scenario_agents)
    for name, run_agent in run_agents.items():
        assert run_agent.proxy is False
        assert run_agent.source_id == scenario_agents[name].source_id
        _assert_copy(run_agent, scenario_agents[name])
    assert run_agents["LibA"].algorithm_type == "PPO"
    assert run_agents["NewB"].failstack_limit == 5


def test_run_of_a_run_redirects_to_its_watch_page(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(views.StarterService, "start", lambda sim: sim.id)
    market = make_market()
    response = client.get(reverse("run_simulation", args=[market.simulation.id]))
    assert response.status_code == 302
    assert response.url == reverse(
        "watch_simulation", kwargs={"simulation_id": market.simulation.id}
    )
    assert Simulation.objects.count() == 1


def test_run_clone_error_returns_400(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation = _scenario(client, [_new_agent(fossil)])

    def failing_clone(*args: Any, **kwargs: Any) -> Simulation:
        raise ValueError("boom")

    monkeypatch.setattr(views.library_service, "clone_scenario_for_run", failing_clone)
    response = client.get(reverse("run_simulation", args=[simulation.id]))
    assert response.status_code == 400
    assert Simulation.objects.filter(proxy=False).count() == 0


def test_run_start_error_deletes_the_run(
    client: Client, fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation = _scenario(client, [_new_agent(fossil)])

    def failing_start(simulation: Simulation) -> int:
        raise OSError("cannot spawn")

    monkeypatch.setattr(views.StarterService, "start", failing_start)
    response = client.get(reverse("run_simulation", args=[simulation.id]))
    assert response.status_code == 500
    assert Simulation.objects.filter(proxy=False).count() == 0
    assert Agent.objects.filter(proxy=False).count() == 0
    assert simulation.market.agent_set.count() == 1


def test_differs_from_source_flags_only_configuration_changes(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    from abmem.services.agent.library_service import differs_from_source

    lib_a = _library_agent(coal)
    simulation = _scenario(
        client,
        [
            _library_entry(lib_a),
            _edited_library_entry(lib_a, fossil),
        ],
    )
    unedited, edited = simulation.market.agent_set.order_by("id")
    assert differs_from_source(unedited) is False
    assert differs_from_source(edited) is True
    unedited.name = "Renamed"
    unedited.save(update_fields=["name"])
    assert differs_from_source(unedited) is False
    standalone = Agent.objects.filter(market__isnull=True).first()
    assert differs_from_source(standalone) is False


def test_wizard_marks_saved_edited_copies(
    client: Client, fossil: Resource, coal: Resource
) -> None:
    lib_a = _library_agent(coal)
    simulation = _scenario(
        client, [_library_entry(lib_a), _edited_library_entry(lib_a, fossil)]
    )
    html = client.get(reverse("update_simulation", args=[simulation.id])).content.decode()
    match = re.search(r'<script id="wizard-config" type="application/json">(.*?)</script>', html, re.S)
    assert match
    agents = json.loads(match.group(1))["initial"]["agents"]
    flags = {agent["name"]: agent["edited"] for agent in agents}
    assert flags == {"LibA": False, "Edited": True}
