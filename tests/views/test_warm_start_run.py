"""Warm start chosen in the run dialog (finding A9)."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse

from abmem import views
from abmem.models import PolicyCheckpoint, Simulation
from abmem.models.enums import SimulationState
from abmem.models.models import Plant
from abmem.services.agent import library_service
from abmem.services.market.demand import DemandSeries
from abmem.services.simulation import checkpoints, simulation_service
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_resource,
    make_simulation,
)
from tests.runtime.conftest import SMALL_DDPG, constant_demand

pytestmark = pytest.mark.django_db

PERIODS = 30


@pytest.fixture(autouse=True)
def _fast_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Excel demand file with an in-memory series."""
    series = constant_demand()
    monkeypatch.setattr(
        DemandSeries, "from_excel", staticmethod(lambda *_a, **_k: series)
    )


def _scenario(name: str = "ws-sc") -> Simulation:
    simulation = make_simulation(name=name, period_number=PERIODS)
    simulation.proxy = True
    simulation.save()
    market = make_market(simulation, upper_bid_bound=200)
    market.proxy = True
    market.save()
    learner = make_agent_with_plants(
        market,
        [(make_resource(name=f"{name}-r1"), 100), (make_resource(name=f"{name}-r2"), 50)],
        name="Learner",
        budget=Decimal("10"),
    )
    learner.proxy = True
    learner.algorithm_type = "DDPG"
    learner.algorithm_params = dict(SMALL_DDPG)
    learner.save()
    baseline = make_agent_with_plants(
        market, [(make_resource(name=f"{name}-r3"), 80)], name="Baseline"
    )
    baseline.proxy = True
    baseline.algorithm_type = "MARGINAL_COST"
    baseline.save()
    return simulation


def _finished_run(scenario: Simulation, **options: Any) -> Simulation:
    run = library_service.clone_scenario_for_run(scenario, **options)
    simulation_service.run(run, export_json=False)
    run.refresh_from_db()
    assert run.state == SimulationState.FINISHED
    return run


@pytest.fixture
def scenario() -> Simulation:
    return _scenario()


@pytest.fixture
def source(scenario: Simulation) -> Simulation:
    return _finished_run(scenario)


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the run process start and record the started run ids."""
    ids: list[int] = []

    def fake_start(simulation: Simulation) -> int:
        ids.append(simulation.id)
        return simulation.id

    monkeypatch.setattr(views.StarterService, "start", fake_start)
    return ids


def _post(client: Client, scenario: Simulation, body: dict[str, Any]) -> Any:
    return client.post(
        reverse("run_simulation", args=[scenario.id]),
        data=json.dumps(body),
        content_type="application/json",
        HTTP_ACCEPT="application/json",
    )


def _new_runs(scenario: Simulation, *exclude: Simulation) -> list[Simulation]:
    return list(
        Simulation.objects.filter(scenario=scenario).exclude(
            pk__in=[s.pk for s in exclude]
        )
    )


def test_source_run_has_one_checkpoint_and_records_its_scenario(
    scenario: Simulation, source: Simulation
) -> None:
    assert source.scenario == scenario
    assert source.warm_start_from is None and not source.freeze_policies
    assert PolicyCheckpoint.objects.filter(simulation=source).count() == 1


@pytest.mark.parametrize("freeze", [False, True])
def test_post_with_warm_start_records_the_origin(
    client: Client,
    scenario: Simulation,
    source: Simulation,
    started: list[int],
    freeze: bool,
) -> None:
    response = _post(
        client, scenario, {"seed": 5, "warm_start_from": source.id, "freeze_policies": freeze}
    )
    assert response.status_code == 200, response.content
    run = Simulation.objects.get(pk=response.json()["run_id"])
    assert started == [run.id]
    assert run.scenario == scenario
    assert run.warm_start_from == source
    assert run.freeze_policies is freeze
    assert run.seed == 5


def test_warm_started_run_executes_and_shows_its_origin_in_the_analysis(
    client: Client, scenario: Simulation, source: Simulation
) -> None:
    run = _finished_run(scenario, warm_start_from=source, freeze_policies=True)
    data = client.get(reverse("simulation_analysis_data", args=[run.id])).json()
    settings = data["settings"]
    assert settings["warm_start_from"] == source.id
    assert settings["freeze_policies"] is True
    assert len(settings["checkpoints"]) == 1
    saved = settings["checkpoints"][0]
    assert saved["agent"] == "Learner"
    assert saved["algorithm"] == "DDPG"
    assert saved["parent_run"] == source.id
    assert saved["file_size"] > 0


def test_dry_run_validates_without_creating_a_run(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    response = _post(
        client, scenario, {"warm_start_from": source.id, "dry_run": True}
    )
    assert response.status_code == 200
    assert response.json() == {"warnings": []}
    assert _new_runs(scenario, source) == []
    assert started == []


def _assert_rejected(
    response: Any, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    assert response.status_code == 400, response.content
    assert "warm_start_from" in response.json()["errors"] or (
        "freeze_policies" in response.json()["errors"]
    )
    assert _new_runs(scenario, source) == []
    assert started == []


def test_unknown_source_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    response = _post(client, scenario, {"warm_start_from": source.id + 10_000})
    _assert_rejected(response, scenario, source, started)


@pytest.mark.parametrize("value", ["abc", 0, -3, 1.5, True])
def test_malformed_source_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int], value: Any
) -> None:
    response = _post(client, scenario, {"warm_start_from": value})
    _assert_rejected(response, scenario, source, started)


def test_freeze_without_source_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    response = _post(client, scenario, {"freeze_policies": True})
    _assert_rejected(response, scenario, source, started)


def test_source_of_another_scenario_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    other = _scenario("ws-other")
    foreign = _finished_run(other)
    response = _post(client, scenario, {"warm_start_from": foreign.id})
    _assert_rejected(response, scenario, source, started)
    assert "scenario" in response.json()["errors"]["warm_start_from"][0]


def test_unfinished_source_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    source.state = SimulationState.FAILED
    source.save()
    response = _post(client, scenario, {"warm_start_from": source.id})
    _assert_rejected(response, scenario, source, started)


def test_scenario_as_source_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    response = _post(client, scenario, {"warm_start_from": scenario.id})
    _assert_rejected(response, scenario, source, started)


def test_source_without_checkpoints_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    PolicyCheckpoint.objects.filter(simulation=source).delete()
    response = _post(client, scenario, {"warm_start_from": source.id})
    _assert_rejected(response, scenario, source, started)


def test_corrupted_checkpoint_is_rejected(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    path = checkpoints.checkpoint_path(PolicyCheckpoint.objects.get(simulation=source))
    path.write_bytes(path.read_bytes()[:-10])
    response = _post(client, scenario, {"warm_start_from": source.id})
    _assert_rejected(response, scenario, source, started)
    assert "corrupted" in response.json()["errors"]["warm_start_from"][0]


def test_changed_plants_warn_without_freeze_and_fail_with_freeze(
    client: Client, scenario: Simulation, source: Simulation, started: list[int]
) -> None:
    learner = scenario.market.agent_set.get(name="Learner")
    Plant.objects.create(
        portfolio=learner.portfolio, resource=make_resource(name="ws-extra"), capacity=10
    )
    response = _post(
        client, scenario, {"warm_start_from": source.id, "dry_run": True}
    )
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 1 and "Learner" in warnings[0]

    response = _post(
        client, scenario, {"warm_start_from": source.id, "freeze_policies": True}
    )
    _assert_rejected(response, scenario, source, started)


def test_scenario_list_offers_the_finished_runs_with_policies(
    client: Client, scenario: Simulation, source: Simulation
) -> None:
    html = client.get(reverse("simulation_list_proxy")).content.decode()
    assert 'id="run-warm-start"' in html
    assert 'id="run-freeze"' in html
    assert 'id="run-sources"' in html
    assert f'data-scenario-id="{scenario.id}"' in html
    start = html.index('id="run-sources"')
    payload = html[html.index(">", start) + 1 : html.index("</script>", start)]
    sources = json.loads(payload)
    assert [run["id"] for run in sources[str(scenario.id)]] == [source.id]
    assert sources[str(scenario.id)][0]["checkpoints"] == 1
    assert "alert(" not in html and "confirm(" not in html and "prompt(" not in html
