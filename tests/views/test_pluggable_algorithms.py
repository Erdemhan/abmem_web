"""Views, forms and services with discovered agent and clearing algorithms (PLAN 2.10)."""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.forms import AgentForm
from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from abmem.services.agent.library_service import (
    AgentInput,
    LibraryValidationError,
    PlantInput,
)
from abmem.templatetags.abmem_choices import algorithm_choices
from tests.factories import make_resource
from tests.plugin_helpers import temporary_agent_algorithm, temporary_clearing_algorithm
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

JSON_HEADERS: dict[str, str] = {"HTTP_ACCEPT": "application/json"}


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="plug-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _new_agent(resource: Resource, algorithm: str = "DDPG") -> dict[str, Any]:
    return {
        "mode": "new",
        "name": "PlugNew",
        "type": AgentType.FOSSIL.value,
        "budget": "250",
        "algorithm_type": algorithm,
        "failstack": {"enabled": False, "limit": "6", "delta": "15", "min_margin": "0"},
        "plants": [{"resource": str(resource.id), "capacity": "120"}],
    }


def _payload(resource: Resource, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "plug-sim",
        "simulationMode": "1",
        "periodType": "0",
        "periodNumber": "48",
        "strategy": "1",
        "lowerBound": "0",
        "upperBound": "200",
        "seed": "42",
        "evaluation_periods": "0",
        "observation_lags": [1, 24],
        "agents": [_new_agent(resource)],
    }
    payload.update(fields)
    return payload


def _post(client: Client, url: str, payload: Any) -> Any:
    return client.post(url, json.dumps(payload), content_type="application/json")


def _create(client: Client, payload: dict[str, Any]) -> Simulation:
    response = _post(client, reverse("create_simulation"), payload)
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


# Scenario wizard: clearing algorithm --------------------------------------------


def test_wizard_saves_the_default_clearing_algorithm(
    client: Client, fossil: Resource
) -> None:
    scenario = _create(client, _payload(fossil))
    assert scenario.market.clearing_algorithm == "MERIT_ORDER"


def test_wizard_saves_and_updates_a_discovered_clearing_algorithm(
    client: Client, fossil: Resource
) -> None:
    with temporary_clearing_algorithm("TEST_CLEARING", "Test clearing"):
        scenario = _create(client, _payload(fossil, clearing_algorithm="TEST_CLEARING"))
        assert scenario.market.clearing_algorithm == "TEST_CLEARING"

        html = client.get(reverse("update_simulation", args=[scenario.id])).content
        page = html.decode()
        assert wizard_config(page)["initial"]["clearing_algorithm"] == "TEST_CLEARING"
        assert re.search(
            r'<option value="TEST_CLEARING"[^>]*selected>Test clearing<', page
        )

        payload = _payload(fossil, clearing_algorithm="MERIT_ORDER")
        payload["agents"] = []
        response = _post(
            client, reverse("update_simulation", args=[scenario.id]), payload
        )
        assert response.status_code == 200, response.content
        scenario.market.refresh_from_db()
        assert scenario.market.clearing_algorithm == "MERIT_ORDER"


@pytest.mark.parametrize("bad", ["NOT_THERE", "merit order", 5, ["MERIT_ORDER"]])
def test_wizard_rejects_an_unknown_clearing_algorithm(
    client: Client, fossil: Resource, bad: object
) -> None:
    response = _post(
        client, reverse("create_simulation"), _payload(fossil, clearing_algorithm=bad)
    )
    assert response.status_code == 400
    messages = response.json()["errors"]["clearing_algorithm"]
    assert "is not available" in messages[0] and "MERIT_ORDER" in messages[0]
    assert not Simulation.objects.exists()


def test_wizard_page_lists_clearing_algorithms_with_descriptions(
    client: Client,
) -> None:
    with temporary_clearing_algorithm("TEST_CLEARING", "Test clearing"):
        page = client.get(reverse("create_simulation")).content.decode()
    assert 'id="clearing_algorithm"' in page
    assert re.search(
        r'<option value="MERIT_ORDER" data-description="Accepts bids[^"]*" selected>'
        r"Uniform-price merit order</option>",
        page,
    )
    assert '<option value="TEST_CLEARING"' in page
    config = wizard_config(page)
    assert config["initial"]["clearing_algorithm"] == "MERIT_ORDER"
    assert [c["key"] for c in config["clearing_algorithms"]] == [
        "MERIT_ORDER",
        "TEST_CLEARING",
    ]
    # The settlement rule stays a separate choice.
    assert 'name="strategy"' in page


def test_editing_a_scenario_with_a_removed_clearing_key_shows_it_unavailable(
    client: Client, fossil: Resource
) -> None:
    scenario = _create(client, _payload(fossil))
    scenario.market.clearing_algorithm = "REMOVED_CLEARING"
    scenario.market.save()
    page = client.get(reverse("update_simulation", args=[scenario.id])).content.decode()
    assert re.search(
        r'<option value="REMOVED_CLEARING" data-unavailable="1"[^>]*selected>'
        r"REMOVED_CLEARING \(unavailable\)</option>",
        page,
    )
    payload = _payload(fossil, clearing_algorithm="REMOVED_CLEARING")
    payload["agents"] = []
    response = _post(client, reverse("update_simulation", args=[scenario.id]), payload)
    assert response.status_code == 400
    assert "clearing_algorithm" in response.json()["errors"]


def test_lists_show_the_clearing_label(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload(fossil))
    page = client.get(reverse("simulation_list_proxy")).content.decode()
    assert "Uniform-price merit order" in page
    scenario.market.clearing_algorithm = "REMOVED_CLEARING"
    scenario.market.save()
    page = client.get(reverse("simulation_list_proxy")).content.decode()
    assert "REMOVED_CLEARING" in page and "unavailable" in page


def test_duplicate_and_run_copy_the_clearing_algorithm(
    client: Client, fossil: Resource
) -> None:
    with temporary_clearing_algorithm("TEST_CLEARING"):
        scenario = _create(client, _payload(fossil, clearing_algorithm="TEST_CLEARING"))
        copy = library_service.duplicate_scenario(scenario)
        run = library_service.clone_scenario_for_run(scenario)
    assert copy.market.clearing_algorithm == "TEST_CLEARING"
    assert run.market.clearing_algorithm == "TEST_CLEARING"


# Agent algorithms ---------------------------------------------------------------


def test_discovered_agent_algorithm_is_selectable_everywhere(
    client: Client, fossil: Resource
) -> None:
    with temporary_agent_algorithm("TEST_AGENT_ALGO", "Test algorithm"):
        assert ("TEST_AGENT_ALGO", "Test algorithm", True) in algorithm_choices()
        page = client.get(reverse("create_simulation")).content.decode()
        assert wizard_config(page)["algorithms"]["TEST_AGENT_ALGO"] == {
            "name": "Test algorithm",
            "baseline": True,
        }
        agent_page = client.get(reverse("agent_list")).content.decode()
        assert '<option value="TEST_AGENT_ALGO" data-baseline="1">' in agent_page
        scenario = _create(
            client, _payload(fossil, agents=[_new_agent(fossil, "TEST_AGENT_ALGO")])
        )
        agent = scenario.market.agent_set.get()
        assert agent.algorithm_type == "TEST_AGENT_ALGO"
    page = client.get(reverse("create_simulation")).content.decode()
    assert "TEST_AGENT_ALGO" not in wizard_config(page)["algorithms"]


def test_wizard_rejects_an_unknown_agent_algorithm(
    client: Client, fossil: Resource
) -> None:
    response = _post(
        client,
        reverse("create_simulation"),
        _payload(fossil, agents=[_new_agent(fossil, "QLEARN")]),
    )
    assert response.status_code == 400
    assert "is not available" in json.dumps(response.json()["errors"])
    assert not Simulation.objects.exists()


def _library_agent(resource: Resource, algorithm: str = "TD3") -> Agent:
    return library_service.create_library_agent(
        AgentInput(
            name="PlugLib",
            budget=Decimal("100"),
            type=AgentType.FOSSIL,
            algorithm_type=algorithm,
            failstack_enabled=False,
            failstack_limit=3,
            failstack_delta=Decimal("5"),
            failstack_min_margin=Decimal("0"),
            plants=(PlantInput(resource.id, 50),),
        )
    )


def test_library_service_rejects_unknown_keys(fossil: Resource) -> None:
    with pytest.raises(LibraryValidationError) as info:
        _library_agent(fossil, "QLEARN")
    assert "algorithm_type" in info.value.message_dict
    scenario = Simulation.objects.create(
        name="s",
        mode=1,
        periodType=0,
        periodNumber=1,
        currentPeriod=-1,
        day=0,
        proxy=True,
    )
    with pytest.raises(LibraryValidationError):
        library_service.clone_scenario_for_run(scenario, forced_algorithm="QLEARN")


def test_agent_with_a_removed_algorithm_is_listed_unavailable_and_cannot_be_saved(
    client: Client, fossil: Resource
) -> None:
    agent = _library_agent(fossil)
    Agent.objects.filter(pk=agent.pk).update(algorithm_type="REMOVED_ALGO")
    page = client.get(reverse("agent_list")).content.decode()
    assert "REMOVED_ALGO" in page
    assert "agent-algorithm-unavailable" in page

    payload = client.get(reverse("agent_library_json")).json()
    entry = next(item for item in payload if item["id"] == agent.id)
    assert entry["algorithm_type"] == "REMOVED_ALGO"
    assert entry["algorithm_available"] is False

    data = {
        "name": "PlugLib",
        "budget": "100",
        "type": str(AgentType.FOSSIL.value),
        "algorithm_type": "REMOVED_ALGO",
        "failstack_limit": "3",
        "failstack_delta": "5",
        "failstack_min_margin": "0",
        "resources[]": [str(fossil.id)],
        "capacities[]": ["50"],
    }
    response = client.post(
        reverse("agent_update", args=[agent.id]), data, **JSON_HEADERS
    )
    assert response.status_code == 400
    assert "is not available" in response.json()["errors"]["algorithm_type"][0]
    data["algorithm_type"] = "SAC"
    response = client.post(
        reverse("agent_update", args=[agent.id]), data, **JSON_HEADERS
    )
    assert response.status_code == 200, response.content
    agent.refresh_from_db()
    assert agent.algorithm_type == "SAC"


def test_agent_form_rejects_unknown_and_keeps_known_keys(fossil: Resource) -> None:
    base = {
        "name": "FormAgent",
        "budget": "10",
        "type": str(AgentType.FOSSIL.value),
        "failstack_limit": "6",
        "failstack_delta": "15",
        "failstack_min_margin": "0",
    }
    form = AgentForm({**base, "algorithm_type": "QLEARN"}, plant_resources=[fossil])
    assert not form.is_valid()
    assert "is not available" in form.errors["algorithm_type"][0]
    form = AgentForm(
        {**base, "algorithm_type": "MARGINAL_COST"}, plant_resources=[fossil]
    )
    assert form.is_valid(), form.errors
    assert form.cleaned_data["algorithm_type"] == "MARGINAL_COST"
    form = AgentForm(base, plant_resources=[fossil])
    assert form.is_valid(), form.errors
    assert form.cleaned_data["algorithm_type"] == "DDPG"


# HPO --------------------------------------------------------------------------------


def test_hpo_param_space_accepts_keys(client: Client) -> None:
    response = client.get(reverse("hpo_param_space"), {"algorithm": " td3 "})
    assert response.status_code == 200, response.content
    assert response.json()["algorithm"] == "TD3"
    response = client.get(reverse("hpo_param_space"), {"algorithm": "2"})
    assert response.status_code == 400


def test_hpo_create_page_lists_labels(client: Client) -> None:
    page = client.get(reverse("hpo_create")).content.decode()
    assert '<option value="FIXED_MARKUP" >Fixed markup</option>' in page
    assert 'value="RANDOM"' not in page


# No live use of the legacy enum ------------------------------------------------------


def test_no_live_module_uses_the_legacy_algorithm_enum() -> None:
    root = Path(__file__).resolve().parents[2] / "abmem"
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "migrations" not in path.parts
        and path.name != "enums.py"
        and "AlgorithmType" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
    for pattern in ("*.html", "*.js"):
        for path in root.rglob(pattern):
            assert "AlgorithmType" not in path.read_text(encoding="utf-8"), path
