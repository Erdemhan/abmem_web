"""Run settings of the scenario create/update pages (decision OQ-16).

``seed``, ``evaluation_periods`` and ``observation_lags`` are part of the
scenario JSON payload, validated server side (HTTP 400 with field errors and
no write on rejection) and copied to every run by
``library_service.clone_scenario_for_run``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.models.simulation import DEFAULT_SEED
from abmem.services.agent import library_service
from tests.factories import make_resource

pytestmark = pytest.mark.django_db


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="rs-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _agent(resource: Resource, name: str = "RsAgent") -> dict[str, Any]:
    return {
        "mode": "new",
        "name": name,
        "type": AgentType.FOSSIL.value,
        "budget": "250",
        "algorithm_type": "TD3",
        "failstack": {"enabled": False},
        "plants": [{"resource": resource.id, "capacity": "200"}],
    }


def _payload(resource: Resource, **settings: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "rs-sim",
        "simulationMode": "1",
        "periodNumber": "10",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": [_agent(resource)],
    }
    payload.update(settings)
    return payload


def _post(client: Client, url: str, payload: dict[str, Any]) -> Any:
    return client.post(url, json.dumps(payload), content_type="application/json")


def _create(client: Client, payload: dict[str, Any]) -> Any:
    return _post(client, reverse("create_simulation"), payload)


def _created(client: Client, payload: dict[str, Any]) -> Simulation:
    response = _create(client, payload)
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


# create


def test_create_stores_run_settings(client: Client, fossil: Resource) -> None:
    simulation = _created(
        client,
        _payload(
            fossil, seed="42", evaluation_periods="3", observation_lags=[1, 24, 168]
        ),
    )

    assert (simulation.seed, simulation.evaluation_periods) == (42, 3)
    assert simulation.observation_lags == [1, 24, 168]


def test_create_without_settings_uses_model_defaults(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil))

    assert simulation.seed == DEFAULT_SEED
    assert simulation.evaluation_periods == 0
    assert simulation.observation_lags == [1, 24]


def test_create_accepts_seed_zero_and_comma_string_lags(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil, seed=0, observation_lags="1,24,168"))

    assert simulation.seed == 0
    assert simulation.observation_lags == [1, 24, 168]


def test_create_accepts_largest_storable_seed(client: Client, fossil: Resource) -> None:
    simulation = _created(client, _payload(fossil, seed=2**31 - 1))

    assert simulation.seed == 2**31 - 1


@pytest.mark.parametrize(
    ("settings", "field"),
    [
        ({"seed": "-1"}, "seed"),
        ({"seed": 2**32}, "seed"),
        ({"seed": 2**31}, "seed"),  # Simulation.seed is a signed 32-bit column
        ({"seed": "abc"}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"evaluation_periods": "10"}, "evaluation_periods"),
        ({"evaluation_periods": "11"}, "evaluation_periods"),
        ({"evaluation_periods": "-1"}, "evaluation_periods"),
        ({"evaluation_periods": ""}, "evaluation_periods"),
        ({"observation_lags": [1, 168]}, "observation_lags"),
        ({"observation_lags": [24, 1]}, "observation_lags"),
        ({"observation_lags": "1,x"}, "observation_lags"),
        ({"observation_lags": 24}, "observation_lags"),
        ({"periodNumber": "0"}, "periodNumber"),
    ],
)
def test_create_rejects_invalid_settings_without_writing(
    client: Client, fossil: Resource, settings: dict[str, Any], field: str
) -> None:
    response = _create(client, _payload(fossil, **settings))

    assert response.status_code == 400
    errors = response.json()["errors"]
    assert list(errors) == [field]
    assert errors[field]
    assert not Simulation.objects.exists()
    assert not Agent.objects.exists()


def test_create_reports_settings_and_agent_errors_together(
    client: Client, fossil: Resource
) -> None:
    payload = _payload(fossil, seed="-5", observation_lags=[1])
    payload["agents"][0]["name"] = ""

    response = _create(client, payload)

    assert response.status_code == 400
    errors = response.json()["errors"]
    assert {"seed", "observation_lags"} <= set(errors)
    assert any(key.startswith("agents[0]") for key in errors)
    assert not Simulation.objects.exists()


def test_evaluation_periods_one_less_than_period_number_is_valid(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil, evaluation_periods=9))

    assert simulation.evaluation_periods == 9


# update


def test_update_changes_run_settings(client: Client, fossil: Resource) -> None:
    simulation = _created(client, _payload(fossil))
    url = reverse("update_simulation", args=[simulation.id])
    payload = _payload(
        fossil, seed=7, evaluation_periods=2, observation_lags=[1, 24, 168]
    )
    payload["agents"] = []

    response = _post(client, url, payload)

    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert (simulation.seed, simulation.evaluation_periods) == (7, 2)
    assert simulation.observation_lags == [1, 24, 168]


def test_update_without_settings_keeps_stored_values(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(
        client,
        _payload(fossil, seed=99, evaluation_periods=4, observation_lags=[1, 24, 168]),
    )
    payload = _payload(fossil)
    payload["agents"] = []

    response = _post(
        client, reverse("update_simulation", args=[simulation.id]), payload
    )

    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert (simulation.seed, simulation.evaluation_periods) == (99, 4)
    assert simulation.observation_lags == [1, 24, 168]


def test_update_rejects_kept_evaluation_periods_above_new_period_number(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil, evaluation_periods=8))
    payload = _payload(fossil, periodNumber="5")
    payload["agents"] = []

    response = _post(
        client, reverse("update_simulation", args=[simulation.id]), payload
    )

    assert response.status_code == 400
    assert list(response.json()["errors"]) == ["evaluation_periods"]
    simulation.refresh_from_db()
    assert (simulation.periodNumber, simulation.evaluation_periods) == (10, 8)


def test_update_rejection_writes_nothing(client: Client, fossil: Resource) -> None:
    simulation = _created(client, _payload(fossil, seed=11))
    agent_names = list(simulation.market.agent_set.values_list("name", flat=True))
    payload = _payload(fossil, seed="bad", simulation_name="changed")
    payload["agents"] = []

    response = _post(
        client, reverse("update_simulation", args=[simulation.id]), payload
    )

    assert response.status_code == 400
    assert list(response.json()["errors"]) == ["seed"]
    simulation.refresh_from_db()
    assert (simulation.name, simulation.seed) == ("rs-sim", 11)
    assert (
        list(simulation.market.agent_set.values_list("name", flat=True)) == agent_names
    )


# pages


def _selected_lags(html: str) -> list[str]:
    select = re.search(r'<select[^>]*id="observation_lags".*?</select>', html, re.S)
    assert select is not None
    return re.findall(r'<option value="([^"]+)"\s*selected', select.group(0))


def test_create_page_renders_default_settings(client: Client) -> None:
    html = client.get(reverse("create_simulation")).content.decode()

    assert re.search(rf'id="seed"[^>]*value="{DEFAULT_SEED}"', html, re.S)
    assert re.search(r'id="evaluation_periods"[^>]*value="0"', html, re.S)
    assert '<option value="1,24,168"' in html
    assert _selected_lags(html) == ["1,24"]


def test_update_page_prefills_settings(client: Client, fossil: Resource) -> None:
    simulation = _created(
        client,
        _payload(fossil, seed=123, evaluation_periods=5, observation_lags=[1, 24, 168]),
    )

    html = client.get(
        reverse("update_simulation", args=[simulation.id])
    ).content.decode()

    assert re.search(r'id="seed"[^>]*value="123"', html, re.S)
    assert re.search(r'id="evaluation_periods"[^>]*value="5"', html, re.S)
    assert _selected_lags(html) == ["1,24,168"]


@pytest.mark.parametrize(
    "url_name",
    ["create_simulation", "agent_list"],
)
def test_pages_have_no_blocking_dialogs(client: Client, url_name: str) -> None:
    html = client.get(reverse(url_name)).content.decode()

    assert "alert(" not in html
    assert "confirm(" not in html


# runs


def test_clone_scenario_for_run_copies_run_settings(
    client: Client, fossil: Resource
) -> None:
    scenario = _created(
        client,
        _payload(fossil, seed=31, evaluation_periods=6, observation_lags=[1, 24, 168]),
    )

    run = library_service.clone_scenario_for_run(scenario)

    assert not run.proxy
    assert (run.seed, run.evaluation_periods, run.periodNumber) == (31, 6, 10)
    assert run.observation_lags == [1, 24, 168]


def test_clone_scenario_for_run_seed_override_keeps_other_settings(
    client: Client, fossil: Resource
) -> None:
    scenario = _created(client, _payload(fossil, seed=31, evaluation_periods=6))

    run = library_service.clone_scenario_for_run(scenario, seed=0)

    assert (run.seed, run.evaluation_periods) == (0, 6)
    assert run.observation_lags == [1, 24]
