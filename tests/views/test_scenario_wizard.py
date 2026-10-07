"""Agent library entry points and the scenario wizard (PLAN 2.6).

Covers the dashboard counter and "New Agent" link, the sidebar link, the
wizard pages (create and prefilled update), the payload round trip, the
server-side header validation, the demand statistics endpoint, the
"Duplicate" endpoint and a syntax check of the inline page scripts.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from decimal import Decimal
from typing import Any, Iterator

import numpy as np
import pytest
from django.test import Client
from django.urls import reverse

from abmem import views_scenario
from abmem.models import Agent, Simulation
from abmem.models.enums import (
    AgentType,
    EnergyType,
    MarketStrategy,
    SimulationMode,
    SimulationState,
)
from abmem.models.models import Plant, Resource
from abmem.models.simulation import DEFAULT_SEED
from abmem.services.agent import library_service
from abmem.services.agent.library_service import AgentInput, PlantInput
from abmem.services.market.demand import DemandSeries
from tests.factories import make_agent_with_plants, make_market, make_resource
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

NODE = shutil.which("node")


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="wiz-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


@pytest.fixture(autouse=True)
def _fresh_demand_cache() -> Iterator[None]:
    """Clear the per-process demand cache around every test."""
    views_scenario.load_demand_series.cache_clear()
    yield
    views_scenario.load_demand_series.cache_clear()


def _library_agent(resource: Resource, name: str = "WizLib", capacity: int = 300) -> Agent:
    return library_service.create_library_agent(
        AgentInput(
            name=name,
            budget=Decimal("500"),
            type=AgentType.FOSSIL,
            algorithm_type="TD3",
            failstack_enabled=True,
            failstack_limit=3,
            failstack_delta=Decimal("10"),
            failstack_min_margin=Decimal("0.05"),
            plants=(PlantInput(resource.id, capacity),),
        )
    )


def _new_agent(resource: Resource, name: str = "WizNew") -> dict[str, Any]:
    return {
        "mode": "new",
        "name": name,
        "type": AgentType.FOSSIL.value,
        "budget": "250",
        "algorithm_type": "DDPG",
        "failstack": {"enabled": False, "limit": "6", "delta": "15", "min_margin": "0"},
        "plants": [{"resource": str(resource.id), "capacity": "120"}],
    }


def _payload(agents: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "wiz-sim",
        "simulationMode": "1",
        "periodType": "0",
        "periodNumber": "168",
        "strategy": "1",
        "lowerBound": "0",
        "upperBound": "200",
        "seed": "42",
        "evaluation_periods": "24",
        "observation_lags": [1, 24],
        "agents": agents,
    }
    payload.update(fields)
    return payload


def _post(client: Client, url: str, payload: Any) -> Any:
    return client.post(url, json.dumps(payload), content_type="application/json")


def _create(client: Client, payload: dict[str, Any]) -> Simulation:
    response = _post(client, reverse("create_simulation"), payload)
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


def _plants(agent: Agent) -> list[tuple[int, int]]:
    return sorted(
        agent.portfolio.plant_set.values_list("resource_id", "capacity")
    )


# dashboard and sidebar -------------------------------------------------


def test_dashboard_counts_only_library_agents(client: Client, fossil: Resource) -> None:
    library = _library_agent(fossil)
    scenario = _create(client, _payload([{"mode": "library", "library_agent_id": library.id}]))
    make_agent_with_plants(make_market(), [(fossil, 5)], name="run-agent")
    assert Agent.objects.count() == 3

    response = client.get(reverse("dashboard"))

    assert response.status_code == 200
    assert response.context["num_library_agents"] == 1
    html = response.content.decode()
    assert '<h5 class="font-weight-bolder mb-0" id="num-library-agents">1</h5>' in html
    assert "Library Agents" in html
    assert f'href="{reverse("agent_list")}" id="dashboard-library-agents"' in html
    assert f'href="{reverse("agent_list")}?new=1" id="dashboard-new-agent"' in html
    assert scenario.market.agent_set.count() == 1


def test_sidebar_links_to_the_agent_library(client: Client) -> None:
    dashboard = client.get(reverse("dashboard")).content.decode()
    library = client.get(reverse("agent_list")).content.decode()

    link = f'href="{reverse("agent_list")}" id="sidebar-agents-link"'
    assert link in dashboard and link in library
    assert re.search(r'class="nav-link active" href="[^"]*" id="sidebar-agents-link"', library)
    assert not re.search(r'class="nav-link active" href="[^"]*" id="sidebar-agents-link"', dashboard)
    assert re.search(r'class="nav-link active" href="%s"' % re.escape(reverse("dashboard")), dashboard)


def test_agent_list_opens_the_create_dialog_for_new_1(client: Client) -> None:
    html = client.get(reverse("agent_list") + "?new=1").content.decode()

    assert 'params.get("new") !== "1"' in html
    assert '$("#create-agent").trigger("click")' in html
    assert "localStorage" not in html


# wizard pages ------------------------------------------------------------


def test_create_wizard_renders_defaults(client: Client, fossil: Resource) -> None:
    response = client.get(reverse("create_simulation"))
    assert response.status_code == 200
    html = response.content.decode()
    config = wizard_config(html)

    assert config["is_update"] is False
    assert config["save_url"] == reverse("create_simulation")
    assert config["demand_stats_url"] == reverse("demand_stats")
    assert config["draft_key"] == "abmem.scenarioWizard.create"
    initial = config["initial"]
    assert initial["periodNumber"] == 168
    assert initial["simulationMode"] == SimulationMode.ONLYRESULT.value
    assert initial["strategy"] == MarketStrategy.PAYASPTF.value
    assert (initial["lowerBound"], initial["upperBound"]) == (0, 200)
    assert initial["seed"] == DEFAULT_SEED
    assert initial["agents"] == []
    assert config["resources"][str(fossil.id)] == {
        "name": "wiz-gas",
        "energy_type": EnergyType.FOSSIL.value,
    }
    for step in range(4):
        assert f'id="wizard-step-{step}"' in html
    for preset in ("24", "168", "336", "custom"):
        assert f'id="duration-{preset}" value="{preset}"' in html
    # The same agent form component as the library page, rendered once.
    assert 'id="agent-form"' in html and 'onclick="agentFormSubmit()"' in html
    assert html.count('id="allowed-energy-types"') == 1
    assert html.count('data-bs-toggle="tooltip"') >= 8
    assert "alert(" not in html and "confirm(" not in html and "prompt(" not in html


def test_update_wizard_is_prefilled(client: Client, fossil: Resource) -> None:
    library = _library_agent(fossil)
    scenario = _create(
        client,
        _payload(
            [{"mode": "library", "library_agent_id": library.id}, _new_agent(fossil)],
            simulation_name="prefill",
            simulationMode="0",
            strategy="0",
            lowerBound="5",
            upperBound="150",
            periodNumber="48",
            seed="7",
            evaluation_periods="6",
            observation_lags=[1, 24, 168],
        ),
    )

    html = client.get(reverse("update_simulation", args=[scenario.id])).content.decode()
    config = wizard_config(html)

    assert config["is_update"] is True
    assert config["simulation_id"] == scenario.id
    assert config["save_url"] == reverse("update_simulation", args=[scenario.id])
    assert config["draft_key"] == f"abmem.scenarioWizard.update.{scenario.id}"
    initial = config["initial"]
    assert {key: initial[key] for key in (
        "simulation_name", "simulationMode", "strategy", "lowerBound", "upperBound",
        "periodNumber", "seed", "evaluation_periods", "observation_lags",
    )} == {
        "simulation_name": "prefill",
        "simulationMode": 0,
        "strategy": 0,
        "lowerBound": 5,
        "upperBound": 150,
        "periodNumber": 48,
        "seed": 7,
        "evaluation_periods": 6,
        "observation_lags": [1, 24, 168],
    }
    rows = initial["agents"]
    assert [row["mode"] for row in rows] == ["scenario", "scenario"]
    assert [row["name"] for row in rows] == ["WizLib", "WizNew"]
    assert rows[0]["source_id"] == library.id and rows[0]["source_name"] == "WizLib"
    assert rows[0]["plants"] == [
        {
            "resource_id": fossil.id,
            "resource_name": "wiz-gas",
            "energy_type": EnergyType.FOSSIL.value,
            "capacity": 300,
        }
    ]
    assert "Edit scenario: prefill" in html
    assert re.search(r'id="seed"[^>]*value="7"', html, re.S)


def test_update_wizard_rejects_runs(client: Client) -> None:
    run = make_market().simulation
    assert client.get(reverse("update_simulation", args=[run.id])).status_code == 400


def test_payload_round_trip_keeps_the_scenario(client: Client, fossil: Resource) -> None:
    library = _library_agent(fossil)
    scenario = _create(
        client,
        _payload([{"mode": "library", "library_agent_id": library.id}, _new_agent(fossil)]),
    )
    before = {
        agent.name: (agent.id, agent.source_id, _plants(agent))
        for agent in scenario.market.agent_set.all()
    }
    initial = wizard_config(
        client.get(reverse("update_simulation", args=[scenario.id])).content.decode()
    )["initial"]

    # The wizard sends scenario rows like this (failstack enabled = active).
    agents = [
        {
            "mode": "scenario",
            "agent_id": row["agent_id"],
            "name": row["name"],
            "type": str(row["type"]),
            "budget": row["budget"],
            "algorithm_type": str(row["algorithm_type"]),
            "failstack": {
                "enabled": row["failstack"]["active"],
                "limit": row["failstack"]["limit"],
                "delta": row["failstack"]["delta"],
                "min_margin": row["failstack"]["min_margin"],
            },
            "plants": [
                {"resource": str(plant["resource_id"]), "capacity": str(plant["capacity"])}
                for plant in row["plants"]
            ],
        }
        for row in initial["agents"]
    ]
    payload = {key: initial[key] for key in initial if key != "agents"}
    payload["agents"] = agents
    response = _post(client, reverse("update_simulation", args=[scenario.id]), payload)

    assert response.status_code == 200, response.content
    body = response.json()
    assert body["status"] == "success"
    assert body["simulation_id"] == scenario.id
    assert body["edit_url"] == reverse("update_simulation", args=[scenario.id])
    assert body["run_url"] == reverse("run_simulation", kwargs={"simulation_id": scenario.id})
    after = {
        agent.name: (agent.id, agent.source_id, _plants(agent))
        for agent in scenario.market.agent_set.all()
    }
    assert after == before
    scenario.refresh_from_db()
    assert (scenario.seed, scenario.evaluation_periods, scenario.periodNumber) == (42, 24, 168)


def test_create_response_links_edit_and_run(client: Client, fossil: Resource) -> None:
    response = _post(client, reverse("create_simulation"), _payload([_new_agent(fossil)]))

    assert response.status_code == 200, response.content
    body = response.json()
    sim_id = body["simulation_id"]
    assert body["edit_url"] == reverse("update_simulation", args=[sim_id])
    assert body["run_url"] == reverse("run_simulation", kwargs={"simulation_id": sim_id})
    simulation = Simulation.objects.get(id=sim_id)
    assert simulation.name == "wiz-sim"
    assert simulation.market.strategy == MarketStrategy.PAYASPTF


@pytest.mark.parametrize(
    "fields,key",
    [
        ({"simulation_name": "  "}, "simulation_name"),
        ({"simulation_name": "x" * 21}, "simulation_name"),
        ({"simulationMode": "7"}, "simulationMode"),
        ({"strategy": ""}, "strategy"),
        ({"lowerBound": "abc"}, "lowerBound"),
        ({"lowerBound": "200", "upperBound": "200"}, "upperBound"),
        ({"upperBound": None}, "upperBound"),
    ],
)
def test_header_errors_are_field_keyed(
    client: Client, fossil: Resource, fields: dict[str, Any], key: str
) -> None:
    response = _post(client, reverse("create_simulation"), _payload([_new_agent(fossil)], **fields))

    assert response.status_code == 400
    assert key in response.json()["errors"]
    assert not Simulation.objects.exists()
    assert not Agent.objects.exists()


def test_header_errors_are_merged_with_agent_errors(client: Client, fossil: Resource) -> None:
    bad_agent = _new_agent(fossil)
    bad_agent["budget"] = ""
    response = _post(
        client, reverse("create_simulation"), _payload([bad_agent], simulation_name="")
    )

    assert response.status_code == 400
    errors = response.json()["errors"]
    assert "simulation_name" in errors and "budget" in errors
    assert errors["budget"][0].startswith("Agent 1 (WizNew): ")


def test_update_header_error_keeps_scenario(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)]))
    response = _post(
        client,
        reverse("update_simulation", args=[scenario.id]),
        _payload([], lowerBound="300", upperBound="100", simulation_name="changed"),
    )

    assert response.status_code == 400
    assert "upperBound" in response.json()["errors"]
    scenario.refresh_from_db()
    assert scenario.name == "wiz-sim"
    assert scenario.market.agent_set.count() == 1


# demand statistics -----------------------------------------------------


def _fake_demand(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    series = DemandSeries(values=np.array(values), source="test")
    monkeypatch.setattr(DemandSeries, "from_excel", staticmethod(lambda *a, **k: series))


def test_demand_stats_of_the_horizon(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_demand(monkeypatch, [10.0, 30.0, 20.0, 1000.0])

    response = client.get(reverse("demand_stats"), {"periods": "3"})

    assert response.status_code == 200
    assert response.json() == {
        "periods": 3,
        "available_periods": 4,
        "min": 10.0,
        "mean": 20.0,
        "max": 30.0,
    }


@pytest.mark.parametrize("periods", ["", "0", "-2", "abc", "1.5"])
def test_demand_stats_rejects_bad_periods(
    client: Client, monkeypatch: pytest.MonkeyPatch, periods: str
) -> None:
    _fake_demand(monkeypatch, [10.0])
    response = client.get(reverse("demand_stats"), {"periods": periods})
    assert response.status_code == 400
    assert "periods" in response.json()["errors"]


def test_demand_stats_rejects_a_horizon_beyond_the_data(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_demand(monkeypatch, [10.0, 20.0])
    response = client.get(reverse("demand_stats"), {"periods": "3"})
    assert response.status_code == 400
    body = response.json()
    assert body["available_periods"] == 2
    assert "covers 2 periods" in body["errors"]["periods"][0]


def test_demand_stats_reads_the_run_demand_file(client: Client) -> None:
    response = client.get(reverse("demand_stats"), {"periods": "24"})
    assert response.status_code == 200
    body = response.json()
    assert body["available_periods"] == 8808
    assert 0 < body["min"] <= body["mean"] <= body["max"]


def test_demand_stats_is_get_only(client: Client) -> None:
    assert client.post(reverse("demand_stats")).status_code == 405


# duplicate -------------------------------------------------------------


def test_duplicate_copies_settings_agents_and_plants(client: Client, fossil: Resource) -> None:
    library = _library_agent(fossil)
    original = _create(
        client,
        _payload(
            [{"mode": "library", "library_agent_id": library.id}, _new_agent(fossil)],
            simulation_name="base",
            simulationMode="0",
            strategy="0",
            lowerBound="3",
            upperBound="90",
            observation_lags=[1, 24, 168],
        ),
    )
    original.state = SimulationState.FINISHED
    original.save()
    library_count = Agent.objects.filter(market__isnull=True).count()

    response = client.post(reverse("duplicate_scenario", args=[original.id]))

    copy = Simulation.objects.exclude(id=original.id).get()
    assert response.status_code == 302
    assert response["Location"] == reverse("update_simulation", args=[copy.id])
    assert copy.name == "base (copy)"
    assert copy.proxy is True and copy.market.proxy is True
    assert copy.state == SimulationState.CREATED and copy.currentPeriod == -1
    for field in ("mode", "periodType", "periodNumber", "seed", "evaluation_periods", "observation_lags"):
        assert getattr(copy, field) == getattr(original, field), field
    for field in ("strategy", "lowerBidBound", "upperBidBound"):
        assert getattr(copy.market, field) == getattr(original.market, field), field
    originals = {agent.name: agent for agent in original.market.agent_set.all()}
    copies = {agent.name: agent for agent in copy.market.agent_set.all()}
    assert set(copies) == set(originals) == {"WizLib", "WizNew"}
    for name, agent in copies.items():
        source = originals[name]
        assert agent.id != source.id
        assert agent.proxy is True
        assert agent.source_id == source.source_id
        assert (agent.algorithm_type, agent.failstack_limit, agent.budget) == (
            source.algorithm_type,
            source.failstack_limit,
            source.budget,
        )
        assert _plants(agent) == _plants(source)
        assert agent.portfolio.proxy is True
    assert Plant.objects.filter(portfolio__agent__market=copy.market, proxy=True).count() == 2
    # The library and the original are unchanged.
    assert Agent.objects.filter(market__isnull=True).count() == library_count
    assert original.market.agent_set.count() == 2


def test_duplicate_shortens_long_names(client: Client, fossil: Resource) -> None:
    original = _create(client, _payload([_new_agent(fossil)], simulation_name="x" * 20))
    response = client.post(
        reverse("duplicate_scenario", args=[original.id]), HTTP_ACCEPT="application/json"
    )
    assert response.status_code == 200
    copy = Simulation.objects.get(id=response.json()["simulation_id"])
    assert copy.name == "x" * 13 + " (copy)"
    assert len(copy.name) == 20
    assert response.json()["edit_url"] == reverse("update_simulation", args=[copy.id])


def test_duplicate_rejects_runs_and_get(client: Client) -> None:
    run = make_market().simulation
    assert client.post(reverse("duplicate_scenario", args=[run.id])).status_code == 400
    assert client.get(reverse("duplicate_scenario", args=[run.id])).status_code == 405
    assert Simulation.objects.count() == 1


def test_templates_list_has_duplicate_button(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)]))
    html = client.get(reverse("simulation_list_proxy")).content.decode()
    assert f'action="{reverse("duplicate_scenario", kwargs={"simulation_id": scenario.id})}" method="POST"' in html
    assert 'aria-label="Duplicate wiz-sim"' in html


def test_duplicate_requires_csrf(fossil: Resource) -> None:
    client = Client(enforce_csrf_checks=True)
    scenario = _create(client, _payload([_new_agent(fossil)]))
    response = client.post(reverse("duplicate_scenario", args=[scenario.id]))
    assert response.status_code == 403
    assert Simulation.objects.count() == 1


# inline scripts --------------------------------------------------------


def _inline_scripts(html: str) -> list[str]:
    return [
        body
        for attrs, body in re.findall(r"<script([^>]*)>(.*?)</script>", html, flags=re.S)
        if "src=" not in attrs and "application/json" not in attrs and body.strip()
    ]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_inline_scripts_pass_node_check(
    client: Client, fossil: Resource, tmp_path: pathlib.Path
) -> None:
    library = _library_agent(fossil)
    scenario = _create(client, _payload([{"mode": "library", "library_agent_id": library.id}]))
    pages = {
        "create": reverse("create_simulation"),
        "update": reverse("update_simulation", args=[scenario.id]),
        "dashboard": reverse("dashboard"),
        "agent_list": reverse("agent_list"),
        "list_proxy": reverse("simulation_list_proxy"),
    }
    for name, url in pages.items():
        scripts = _inline_scripts(client.get(url).content.decode())
        assert scripts, name
        for index, body in enumerate(scripts):
            path = tmp_path / f"{name}_{index}.js"
            path.write_text(body, encoding="utf-8")
            result = subprocess.run(
                [NODE, "--check", str(path)], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, f"{name} script {index}: {result.stderr}"
