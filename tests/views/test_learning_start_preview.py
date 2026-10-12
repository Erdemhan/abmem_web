"""Learning start preview and duration units of the scenario wizard (PLAN 2.14).

The wizard asks ``learning_start_preview`` for the common learning start of
the unsaved agents and shows the first learning period next to the number of
periods. The number of periods is entered as a count and a unit and is
always stored in hours.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from abmem.services.agent.library_service import AgentInput, PlantInput
from abmem.services.simulation.context import (
    algorithm_defaults,
    first_learning_period,
    learner_config,
)
from abmem.views import PERIOD_UNITS, period_unit_for
from tests.factories import make_resource
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

ERROR_TEXT = "Learning start must be empty (automatic) or a positive integer."


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="lsp-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _new_agent(
    resource: Resource, name: str = "LspNew", algorithm_type: str = "TD3"
) -> dict[str, Any]:
    return {
        "mode": "new",
        "name": name,
        "type": AgentType.FOSSIL.value,
        "budget": "250",
        "algorithm_type": algorithm_type,
        "failstack": {"enabled": False},
        "plants": [{"resource": resource.id, "capacity": "200"}],
    }


def _library_agent(
    resource: Resource, name: str, params: dict[str, Any], algorithm_type: str = "TD3"
) -> Any:
    return library_service.create_library_agent(
        AgentInput(
            name=name,
            budget=Decimal("500"),
            type=AgentType.FOSSIL,
            algorithm_type=algorithm_type,
            failstack_enabled=False,
            failstack_limit=3,
            failstack_delta=Decimal("10"),
            failstack_min_margin=Decimal("0"),
            plants=(PlantInput(resource.id, 300),),
            algorithm_params=params,
        )
    )


def _preview(client: Client, body: Any) -> Any:
    return client.post(
        reverse("learning_start_preview"),
        json.dumps(body),
        content_type="application/json",
    )


def _td3_limit() -> int:
    limit = learner_config("TD3", algorithm_defaults().get("TD3", {})).learning_starts_limit()
    assert limit is not None
    return limit


def _scenario_payload(resource: Resource, **fields: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "lsp-sim",
        "simulationMode": "1",
        "periodNumber": "168",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": [_new_agent(resource)],
    }
    payload.update(fields)
    return payload


def _create(client: Client, payload: dict[str, Any]) -> Simulation:
    response = client.post(
        reverse("create_simulation"), json.dumps(payload), content_type="application/json"
    )
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


# preview endpoint


def test_preview_automatic(client: Client, fossil: Resource) -> None:
    response = _preview(
        client,
        {"learning_starts": "", "observation_lags": [1, 24], "agents": [_new_agent(fossil)]},
    )

    assert response.status_code == 200, response.content
    assert response.json() == {
        "automatic": True,
        "value": 64,
        "first_period": 89,
        "minimum": 64,
        "limit": _td3_limit(),
    }


def test_preview_explicit(client: Client, fossil: Resource) -> None:
    response = _preview(
        client,
        {
            "learning_starts": 100,
            "observation_lags": [1, 24, 168],
            "agents": [_new_agent(fossil)],
        },
    )

    assert response.status_code == 200, response.content
    body = response.json()
    assert (body["automatic"], body["value"], body["first_period"]) == (False, 100, 269)


def test_preview_without_lags_uses_the_default_lags(client: Client, fossil: Resource) -> None:
    response = _preview(client, {"agents": [_new_agent(fossil)]})

    assert response.status_code == 200, response.content
    assert response.json()["first_period"] == 89


def test_preview_below_minimum_has_the_save_message(
    client: Client, fossil: Resource
) -> None:
    response = _preview(
        client, {"learning_starts": "25", "agents": [_new_agent(fossil)]}
    )
    saved = client.post(
        reverse("create_simulation"),
        json.dumps(_scenario_payload(fossil, learning_starts=25)),
        content_type="application/json",
    )

    assert response.status_code == 400
    message = response.json()["errors"]["learning_starts"]
    assert "below the minimum" in message[0]
    assert message == saved.json()["errors"]["learning_starts"]
    assert not Simulation.objects.exists()


def test_preview_above_buffer(client: Client, fossil: Resource) -> None:
    response = _preview(
        client,
        {"learning_starts": _td3_limit() + 1, "agents": [_new_agent(fossil)]},
    )

    assert response.status_code == 400
    assert "exceeds the buffer size" in response.json()["errors"]["learning_starts"][0]


def test_preview_without_learning_agents(client: Client, fossil: Resource) -> None:
    response = _preview(
        client,
        {
            "learning_starts": "5",
            "agents": [_new_agent(fossil, algorithm_type="MARGINAL_COST")],
        },
    )

    assert response.status_code == 200, response.content
    assert response.json() == {
        "automatic": False,
        "value": None,
        "first_period": None,
        "minimum": None,
        "limit": None,
    }


def test_preview_without_agents(client: Client) -> None:
    response = _preview(client, {"learning_starts": "", "agents": []})

    assert response.status_code == 200
    assert response.json()["value"] is None


def test_preview_skips_agents_with_invalid_params(
    client: Client, fossil: Resource
) -> None:
    broken = _library_agent(fossil, "LspBroken", {"batch_size": "many"})
    agents = [{"mode": "library", "library_agent_id": broken.id}, _new_agent(fossil)]

    response = _preview(client, {"learning_starts": "", "agents": agents})

    assert response.status_code == 200, response.content
    assert response.json()["value"] == 64


def test_preview_uses_library_agent_params(client: Client, fossil: Resource) -> None:
    library = _library_agent(fossil, "LspLib", {"learning_starts": 120})
    agents = [{"mode": "library", "library_agent_id": library.id}, _new_agent(fossil)]

    response = _preview(client, {"learning_starts": None, "agents": agents})

    assert response.status_code == 200, response.content
    assert response.json()["value"] == 120


def test_preview_uses_kept_scenario_agent_params(
    client: Client, fossil: Resource
) -> None:
    scenario = _create(client, _scenario_payload(fossil))
    agent = scenario.market.agent_set.get()
    agent.algorithm_params = {"learning_starts": 150}
    agent.save()
    entry = _new_agent(fossil, name=agent.name)
    entry.update(mode="scenario", agent_id=agent.id)

    response = _preview(
        client, {"simulation_id": scenario.id, "learning_starts": "", "agents": [entry]}
    )

    assert response.status_code == 200, response.content
    assert response.json()["value"] == 150


def test_preview_rejects_scenario_entries_without_the_scenario(
    client: Client, fossil: Resource
) -> None:
    entry = _new_agent(fossil)
    entry.update(mode="scenario", agent_id=1)

    response = _preview(client, {"agents": [entry]})

    assert response.status_code == 400
    assert "agents[0].mode" in response.json()["errors"]


def test_preview_rejects_an_unknown_scenario(client: Client) -> None:
    response = _preview(client, {"simulation_id": 999999, "agents": []})

    assert response.status_code == 400
    assert "simulation_id" in response.json()["errors"]


@pytest.mark.parametrize("raw_value", ["0", -3, "abc", 1.5, True])
def test_preview_rejects_an_invalid_format(
    client: Client, fossil: Resource, raw_value: Any
) -> None:
    response = _preview(
        client, {"learning_starts": raw_value, "agents": [_new_agent(fossil)]}
    )

    assert response.status_code == 400
    assert response.json()["errors"] == {"learning_starts": [ERROR_TEXT]}


def test_preview_rejects_invalid_lags_and_body(client: Client) -> None:
    lags = _preview(client, {"observation_lags": [2, 3], "agents": []})
    body = client.post(
        reverse("learning_start_preview"), "[1]", content_type="application/json"
    )

    assert lags.status_code == 400 and "observation_lags" in lags.json()["errors"]
    assert body.status_code == 400 and "body" in body.json()["errors"]


def test_preview_is_post_only(client: Client) -> None:
    assert client.get(reverse("learning_start_preview")).status_code == 405


def test_first_learning_period() -> None:
    assert first_learning_period((1, 24), 64) == 89
    assert first_learning_period([1, 24, 168], 64) == 233


# duration units


def test_period_units_are_hours() -> None:
    assert [(key, hours) for key, _, hours in PERIOD_UNITS] == [
        ("hour", 1),
        ("day", 24),
        ("week", 168),
        ("month", 720),
        ("year", 8760),
    ]


@pytest.mark.parametrize(
    ("hours", "expected"),
    [
        (1, ("hour", 1)),
        (25, ("hour", 25)),
        (24, ("day", 1)),
        (48, ("day", 2)),
        (168, ("week", 1)),
        (336, ("week", 2)),
        (720, ("month", 1)),
        (1440, ("month", 2)),
        (5040, ("month", 7)),
        (8760, ("year", 1)),
        (17520, ("year", 2)),
        (0, ("hour", 0)),
    ],
)
def test_period_unit_for(hours: int, expected: tuple[str, int]) -> None:
    assert period_unit_for(hours) == expected


def _selected_unit(html: str) -> str:
    match = re.search(r'<select[^>]*id="periodUnit".*?</select>', html, flags=re.S)
    assert match, "unit selector missing"
    selected = re.findall(r'<option value="(\w+)"[^>]*selected', match.group(0))
    assert len(selected) == 1
    return selected[0]


def _count(html: str) -> str:
    match = re.search(r'id="periodCount"[^>]*value="([^"]*)"', html, flags=re.S)
    assert match
    return match.group(1)


def test_create_page_renders_the_unit_selector(client: Client) -> None:
    html = client.get(reverse("create_simulation")).content.decode()
    config = wizard_config(html)

    for label in ("Hour", "Day", "Week", "Month", "Year"):
        assert f">{label}</option>" in html
    assert (_selected_unit(html), _count(html)) == ("week", "1")
    assert re.search(r'id="periodNumber"[^>]*value="168"', html)
    assert {unit["key"]: unit["hours"] for unit in config["period_units"]} == {
        "hour": 1,
        "day": 24,
        "week": 168,
        "month": 720,
        "year": 8760,
    }
    assert config["learning_start_preview_url"] == reverse("learning_start_preview")
    assert 'id="learning-start-info"' in html


@pytest.mark.parametrize(
    ("hours", "unit", "count"),
    [("8760", "year", "1"), ("48", "day", "2"), ("1440", "month", "2"), ("25", "hour", "25")],
)
def test_edit_page_picks_the_unit(
    client: Client, fossil: Resource, hours: str, unit: str, count: str
) -> None:
    scenario = _create(client, _scenario_payload(fossil, periodNumber=hours))

    html = client.get(reverse("update_simulation", args=[scenario.id])).content.decode()

    assert (_selected_unit(html), _count(html)) == (unit, count)
    assert wizard_config(html)["initial"]["periodNumber"] == int(hours)


def test_saved_period_number_is_in_hours(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _scenario_payload(fossil, periodNumber="8760"))

    assert scenario.periodNumber == 8760
