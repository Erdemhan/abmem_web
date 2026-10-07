"""Seed selection when a scenario is run (PLAN 2.5, spec run_analysis)."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem import views
from abmem.models import Simulation
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_resource,
    make_simulation,
)

pytestmark = pytest.mark.django_db

SCENARIO_SEED = 4242


@pytest.fixture
def scenario() -> Simulation:
    """A proxy scenario with one agent and one plant."""
    simulation = make_simulation(name="seed-sc", period_number=5)
    simulation.proxy = True
    simulation.seed = SCENARIO_SEED
    simulation.save()
    market = make_market(simulation, upper_bid_bound=200)
    market.proxy = True
    market.save()
    agent = make_agent_with_plants(
        market,
        [(make_resource(name="seed-res"), 100)],
        name="SeedAgent",
        budget=Decimal("10"),
    )
    agent.proxy = True
    agent.save()
    return simulation


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the run process start and record the started run ids."""
    ids: list[int] = []

    def fake_start(simulation: Simulation) -> int:
        ids.append(simulation.id)
        return simulation.id

    monkeypatch.setattr(views.StarterService, "start", fake_start)
    return ids


def _run_url(simulation: Simulation) -> str:
    return reverse("run_simulation", args=[simulation.id])


def _post_json(client: Client, url: str, body: Any) -> Any:
    return client.post(
        url,
        data=json.dumps(body),
        content_type="application/json",
        HTTP_ACCEPT="application/json",
    )


@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1])
def test_post_json_seed_creates_run_with_that_seed(
    client: Client, scenario: Simulation, started: list[int], seed: int
) -> None:
    response = _post_json(client, _run_url(scenario), {"seed": seed})
    assert response.status_code == 200
    run = Simulation.objects.get(proxy=False)
    assert response.json() == {
        "run_id": run.id,
        "watch_url": reverse("watch_simulation", kwargs={"simulation_id": run.id}),
    }
    assert run.seed == seed
    assert started == [run.id]
    scenario.refresh_from_db()
    assert scenario.seed == SCENARIO_SEED


def test_post_form_seed_redirects_to_watch_page(
    client: Client, scenario: Simulation, started: list[int]
) -> None:
    response = client.post(_run_url(scenario), {"seed": "77"})
    run = Simulation.objects.get(proxy=False)
    assert response.status_code == 302
    assert response.url == reverse("watch_simulation", kwargs={"simulation_id": run.id})
    assert run.seed == 77


@pytest.mark.parametrize("body", [{}, {"seed": None}, {"seed": ""}])
def test_post_without_seed_keeps_the_scenario_seed(
    client: Client, scenario: Simulation, started: list[int], body: dict[str, Any]
) -> None:
    response = _post_json(client, _run_url(scenario), body)
    assert response.status_code == 200
    assert Simulation.objects.get(proxy=False).seed == SCENARIO_SEED


@pytest.mark.parametrize(
    "seed", [-1, 2**31, 2**32, "abc", "1.5", 1.5, True, [1], {"a": 1}]
)
def test_invalid_seed_returns_400_and_creates_no_run(
    client: Client, scenario: Simulation, started: list[int], seed: Any
) -> None:
    response = _post_json(client, _run_url(scenario), {"seed": seed})
    assert response.status_code == 400
    assert "seed" in response.json()["errors"]
    assert not Simulation.objects.filter(proxy=False).exists()
    assert started == []


def test_invalid_form_seed_returns_400(
    client: Client, scenario: Simulation, started: list[int]
) -> None:
    response = client.post(_run_url(scenario), {"seed": "-5"})
    assert response.status_code == 400
    assert response["Content-Type"] == "application/json"
    assert not Simulation.objects.filter(proxy=False).exists()


def test_malformed_json_returns_400(
    client: Client, scenario: Simulation, started: list[int]
) -> None:
    response = client.post(
        _run_url(scenario), data="{not json", content_type="application/json"
    )
    assert response.status_code == 400
    assert not Simulation.objects.filter(proxy=False).exists()


def test_get_still_runs_with_the_scenario_seed(
    client: Client, scenario: Simulation, started: list[int]
) -> None:
    response = client.get(_run_url(scenario))
    run = Simulation.objects.get(proxy=False)
    assert response.status_code == 302
    assert run.seed == SCENARIO_SEED


def test_json_start_failure_returns_500_json(
    client: Client, scenario: Simulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_start(simulation: Simulation) -> int:
        raise OSError("cannot spawn")

    monkeypatch.setattr(views.StarterService, "start", failing_start)
    response = _post_json(client, _run_url(scenario), {"seed": 3})
    assert response.status_code == 500
    assert "__all__" in response.json()["errors"]
    assert not Simulation.objects.filter(proxy=False).exists()


def test_scenario_list_has_run_dialog(client: Client, scenario: Simulation) -> None:
    html = client.get(reverse("simulation_list_proxy")).content.decode()
    assert 'id="run-modal"' in html
    assert 'id="run-seed"' in html
    assert 'id="run-random-seed"' in html
    assert f'data-seed="{SCENARIO_SEED}"' in html
    assert f'data-run-url="{_run_url(scenario)}"' in html
    assert "csrfmiddlewaretoken" in html
    assert "alert(" not in html and "confirm(" not in html and "prompt(" not in html
