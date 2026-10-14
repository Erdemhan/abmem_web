"""Hyperparameter search UI: list, create, param space, watch, status, cancel, apply (spec hpo.ui).

The study process is never spawned: ``hpo_service.start`` is replaced in every
test, and tests that create studies assert on the replacement.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from datetime import timedelta
from decimal import Decimal
from typing import Any

import pytest
from django.test import Client
from django.urls import reverse
from django.utils import timezone

from abmem.models import (
    Agent,
    AlgorithmDefault,
    AlgorithmDefaultOrigin,
    HPOStudy,
    HPOTarget,
    HPOTrial,
    Market,
    Simulation,
)
from abmem.models.enums import (
    AgentType,
    HPOPhase,
    HPOStatus,
    HPOTrialStatus,
    MarketStrategy,
    PeriodType,
    SimulationMode,
)
from abmem.models.models import Plant, Portfolio
from abmem.services.hpo import hpo_service
from tests.factories import make_resource, make_simulation

pytestmark = pytest.mark.django_db


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def started(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace the process start. Returns the ids of the studies "started"."""
    calls: list[int] = []
    monkeypatch.setattr(hpo_service, "start", lambda study: calls.append(study.pk))
    return calls


@pytest.fixture
def scenario() -> Simulation:
    """Proxy scenario with a DDPG agent and a FIXED_MARKUP agent."""
    sim = Simulation.objects.create(
        name="scn",
        mode=SimulationMode.ONLYRESULT,
        periodType=PeriodType.HOUR,
        periodNumber=10,
        currentPeriod=-1,
        day=0,
        seed=3,
        proxy=True,
    )
    market = Market.objects.create(
        simulation=sim,
        strategy=MarketStrategy.PAYASBID,
        lowerBidBound=0,
        upperBidBound=200,
        proxy=True,
    )
    for i, algorithm in enumerate(("DDPG", "FIXED_MARKUP")):
        agent = Agent.objects.create(
            name=f"a{i}",
            market=market,
            proxy=True,
            type=AgentType.HYBRID,
            algorithm_type=algorithm,
            algorithm_params={"batch_size": 8} if i == 0 else {"markup": 0.2},
        )
        portfolio = Portfolio.objects.create(agent=agent, proxy=True)
        Plant.objects.create(
            portfolio=portfolio, resource=make_resource(), capacity=700, proxy=True
        )
    return sim


def payload(scenario: Simulation, **changes: Any) -> dict[str, Any]:
    """Valid create payload as sent by the create page."""
    data: dict[str, Any] = {
        "name": "study-1",
        "scenario": str(scenario.pk),
        "algorithm_type": "DDPG",
        "failstack_enabled": False,
        "colony_size": "2",
        "max_iter": "1",
        "limit": "5",
        "search_seeds": "1, 2",
        "evaluation_seeds": "9",
        "weight_agent": "1",
        "weight_market": "0.5",
        "param_bounds": {
            "gamma": {"low": "0.9", "high": "0.99", "scale": "linear"},
            "gradient_steps": {"low": "1", "high": "4", "scale": "int"},
        },
    }
    data.update(changes)
    return data


def post_json(client: Client, url: str, data: Any) -> Any:
    """POST a JSON body."""
    return client.post(url, data=json.dumps(data), content_type="application/json")


def make_study(scenario: Simulation, **fields: Any) -> HPOStudy:
    """Saved study with small settings."""
    values: dict[str, Any] = {
        "name": "s",
        "algorithm_type": "DDPG",
        "scenario": scenario,
        "param_bounds": {"gamma": {"low": 0.9, "high": 0.99, "scale": "linear"}},
        "fixed_params": {"failstack_enabled": False},
        "colony_size": 2,
        "max_iter": 1,
        "limit": 5,
        "search_seeds": [1],
        "evaluation_seeds": [9],
        "objective": {"mode": "weighted", "weights": [1.0, 0.0]},
    }
    values.update(fields)
    return HPOStudy.objects.create(**values)


def add_trial(study: HPOStudy, **fields: Any) -> HPOTrial:
    """Saved trial, FINISHED by default."""
    values: dict[str, Any] = {
        "iteration": 0,
        "phase": HPOPhase.INIT,
        "source_index": 0,
        "params": {"gamma": 0.95},
        "seed": 1,
        "status": HPOTrialStatus.FINISHED,
        "duration_s": 1.5,
    }
    values.update(fields)
    return HPOTrial.objects.create(study=study, **values)


# -------------------------------------------------------------- list / pages
def test_list_page_shows_studies_and_fails_stale(scenario: Simulation) -> None:
    fresh = make_study(
        scenario, name="fresh-one", status=HPOStatus.FINISHED, best_score=1.25
    )
    stale = make_study(
        scenario,
        name="stale-one",
        status=HPOStatus.RUNNING,
        started_at=timezone.now() - timedelta(hours=5),
        heartbeat_at=timezone.now() - timedelta(hours=5),
    )
    response = Client().get(reverse("hpo_list"))
    assert response.status_code == 200
    content = response.content.decode()
    assert "fresh-one" in content and "stale-one" in content
    assert "1.2500" in content
    assert reverse("hpo_watch", args=[fresh.pk]) in content
    stale.refresh_from_db()
    assert stale.status == HPOStatus.FAILED
    assert "heartbeat" in stale.error


def test_list_page_shows_the_expected_total_of_trials(scenario: Simulation) -> None:
    # SN 3, 2 cycles, 2 search seeds, 1 evaluation seed: (3 + 2 x 2 x 3) x 2 + 1 = 31.
    study = make_study(
        scenario, name="totals", status=HPOStatus.RUNNING, colony_size=3, max_iter=2,
        search_seeds=[1, 2], evaluation_seeds=[9],
    )
    add_trial(study)
    add_trial(study, status=HPOTrialStatus.RUNNING)
    content = Client().get(reverse("hpo_list")).content.decode()
    assert "1 / 31" in content
    # Every scout candidate adds one trial per search seed.
    add_trial(study, phase=HPOPhase.SCOUT, iteration=1, status=HPOTrialStatus.RUNNING)
    add_trial(study, phase=HPOPhase.SCOUT, iteration=1, seed=2, status=HPOTrialStatus.RUNNING)
    content = Client().get(reverse("hpo_list")).content.decode()
    assert "1 / 33" in content


def test_sidebar_links_to_hpo_list() -> None:
    response = Client().get(reverse("hpo_list"))
    assert "Hyperparameter Search" in response.content.decode()
    assert reverse("hpo_list") == "/dev/hpo/"


def test_create_page_lists_only_proxy_scenarios(scenario: Simulation) -> None:
    run = make_simulation(name="a-real-run")
    response = Client().get(reverse("hpo_create") + f"?scenario={scenario.pk}")
    assert response.status_code == 200
    content = response.content.decode()
    assert f'value="{scenario.pk}" selected' in content
    assert run.name not in content
    assert 'value="DDPG"' in content
    assert 'value="RANDOM"' not in content  # nothing to search
    assert "csrfmiddlewaretoken" in content


def test_watch_page_renders(scenario: Simulation) -> None:
    study = make_study(scenario)
    response = Client().get(reverse("hpo_watch", args=[study.pk]))
    assert response.status_code == 200
    content = response.content.decode()
    assert reverse("hpo_status", args=[study.pk]) in content
    assert "chartjs.min.js" in content
    assert Client().get(reverse("hpo_watch", args=[999999])).status_code == 404


# ---------------------------------------------------------------- param space
def test_param_space_returns_bounds_and_defaults() -> None:
    response = Client().get(reverse("hpo_param_space") + "?algorithm=DDPG")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "DDPG"
    assert data["bounds"]["gamma"] == {"low": 0.9, "high": 0.999, "scale": "linear"}
    assert "gradient_steps" in data["integer"]
    assert data["defaults"]["gamma"] == pytest.approx(0.99)
    assert data["failstack"] is True
    assert set(data["failstack_defaults"]) == {"failstack_limit", "failstack_delta"}


def test_param_space_baseline_and_errors() -> None:
    client = Client()
    data = client.get(reverse("hpo_param_space") + "?algorithm=FIXED_MARKUP").json()
    assert data["failstack"] is False
    assert list(data["bounds"]) == ["markup"]
    for bad in ("RANDOM", "nope", ""):
        response = client.get(reverse("hpo_param_space") + f"?algorithm={bad}")
        assert response.status_code == 400
        assert "algorithm" in response.json()["errors"]


# --------------------------------------------------------------------- create
def test_create_valid_study_starts_process(
    scenario: Simulation, started: list[int]
) -> None:
    response = post_json(Client(), reverse("hpo_create"), payload(scenario))
    assert response.status_code == 201, response.content
    study = HPOStudy.objects.get(pk=response.json()["id"])
    assert response.json()["watch_url"] == reverse("hpo_watch", args=[study.pk])
    assert started == [study.pk]
    assert study.status == HPOStatus.CREATED
    assert study.algorithm_type == "DDPG"
    assert study.param_bounds == {
        "gamma": {"low": 0.9, "high": 0.99, "scale": "linear"},
        "gradient_steps": {"low": 1, "high": 4, "scale": "int"},
    }
    assert study.search_seeds == [1, 2] and study.evaluation_seeds == [9]
    assert study.objective == {"mode": "weighted", "weights": [1.0, 0.5]}
    assert study.fixed_params == {"failstack_enabled": False}


def test_create_with_failstack_search_and_fixed_values(
    scenario: Simulation, started: list[int]
) -> None:
    bounds = payload(scenario)["param_bounds"]
    bounds["failstack_delta"] = {"low": "0", "high": "30", "scale": "linear"}
    data = payload(
        scenario,
        failstack_enabled=True,
        failstack_limit="4",
        failstack_delta="15",
        param_bounds=bounds,
    )
    response = post_json(Client(), reverse("hpo_create"), data)
    assert response.status_code == 201, response.content
    study = HPOStudy.objects.get(pk=response.json()["id"])
    # The searched delta is not fixed, the limit is.
    assert study.fixed_params == {"failstack_enabled": True, "failstack_limit": 4}
    assert study.param_bounds["failstack_delta"] == {
        "low": 0.0,
        "high": 30.0,
        "scale": "linear",
    }


def _errors(response: Any) -> dict[str, list[str]]:
    assert response.status_code == 400, response.content
    return response.json()["errors"]


@pytest.mark.parametrize(
    ("bounds", "fragment"),
    [
        (
            {"gamma": {"low": "0.99", "high": "0.9", "scale": "linear"}},
            "must be smaller than max",
        ),
        (
            {"gamma": {"low": "0.9", "high": "0.9", "scale": "linear"}},
            "must be smaller than max",
        ),
        (
            {"not_a_param": {"low": "0", "high": "1", "scale": "linear"}},
            "unknown parameter",
        ),
        (
            {"failstack_limit": {"low": "1", "high": "5", "scale": "int"}},
            "enable fail-stack",
        ),
        (
            {"gamma": {"low": "0.9", "high": "0.99", "scale": "int"}},
            "not an integer parameter",
        ),
        (
            {"gradient_steps": {"low": "1", "high": "4", "scale": "log"}},
            "scale must be 'int'",
        ),
        (
            {"tau": {"low": "0", "high": "0.1", "scale": "log"}},
            "log scale needs min > 0",
        ),
        (
            {"gamma": {"low": "abc", "high": "0.99", "scale": "linear"}},
            "must be a number",
        ),
        ({}, "at least one parameter"),
    ],
)
def test_create_rejects_bad_bounds(
    scenario: Simulation, started: list[int], bounds: dict[str, Any], fragment: str
) -> None:
    errors = _errors(
        post_json(
            Client(), reverse("hpo_create"), payload(scenario, param_bounds=bounds)
        )
    )
    assert any(fragment.lower() in m.lower() for m in errors["param_bounds"]), errors
    assert HPOStudy.objects.count() == 0
    assert started == []


@pytest.mark.parametrize(
    ("changes", "field", "fragment"),
    [
        ({"search_seeds": ""}, "search_seeds", "at least one seed"),
        ({"evaluation_seeds": "  "}, "evaluation_seeds", "at least one seed"),
        ({"search_seeds": "1,,2"}, "search_seeds", "empty entry"),
        ({"search_seeds": "1, x"}, "search_seeds", "not an integer"),
        ({"search_seeds": "1, 1"}, "search_seeds", "duplicate"),
        ({"search_seeds": str(2**31)}, "search_seeds", "outside"),
        ({"evaluation_seeds": "2"}, "evaluation_seeds", "disjoint"),
        ({"colony_size": "1"}, "colony_size", ""),
        ({"max_iter": "0"}, "max_iter", ""),
        ({"name": ""}, "name", ""),
        ({"algorithm_type": "RANDOM"}, "algorithm_type", ""),
        ({"weight_agent": "0", "weight_market": "0"}, "weight_market", "non-zero"),
        (
            {
                "failstack_enabled": True,
                "algorithm_type": "FIXED_MARKUP",
                "param_bounds": {"markup": {"low": 0, "high": 0.4, "scale": "linear"}},
            },
            "failstack_enabled",
            "learning algorithms",
        ),
    ],
)
def test_create_rejects_bad_fields(
    scenario: Simulation, changes: dict[str, Any], field: str, fragment: str
) -> None:
    errors = _errors(
        post_json(Client(), reverse("hpo_create"), payload(scenario, **changes))
    )
    assert field in errors, errors
    assert any(fragment.lower() in m.lower() for m in errors[field]), errors
    assert HPOStudy.objects.count() == 0


def test_create_rejects_non_proxy_scenario(scenario: Simulation) -> None:
    run = make_simulation(name="run")
    errors = _errors(
        post_json(Client(), reverse("hpo_create"), {**payload(scenario), "scenario": run.pk})
    )
    assert "scenario" in errors


def test_create_reports_service_validation_errors(scenario: Simulation) -> None:
    # gamma above 1 passes the form checks but violates the config range.
    bounds = {"gamma": {"low": "0.9", "high": "1.5", "scale": "linear"}}
    errors = _errors(
        post_json(
            Client(), reverse("hpo_create"), payload(scenario, param_bounds=bounds)
        )
    )
    assert any("gamma" in m for m in errors["param_bounds"])


def test_create_rejects_malformed_json() -> None:
    response = Client().post(
        reverse("hpo_create"), data="{", content_type="application/json"
    )
    assert "__all__" in _errors(response)
    response = Client().post(
        reverse("hpo_create"), data="[1]", content_type="application/json"
    )
    assert "__all__" in _errors(response)


def test_create_accepts_form_encoded_post(
    scenario: Simulation, started: list[int]
) -> None:
    data = payload(scenario)
    data["param_bounds"] = json.dumps(data["param_bounds"])
    data["failstack_enabled"] = ""
    response = Client().post(reverse("hpo_create"), data=data)
    assert response.status_code == 201, response.content
    assert len(started) == 1


def test_create_marks_study_failed_when_start_raises(
    scenario: Simulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(study: HPOStudy) -> None:
        raise OSError("no processes")

    monkeypatch.setattr(hpo_service, "start", boom)
    response = post_json(Client(), reverse("hpo_create"), payload(scenario))
    assert response.status_code == 500
    study = HPOStudy.objects.get()
    assert study.status == HPOStatus.FAILED
    assert "no processes" in study.error
    assert "no processes" in response.json()["errors"]["__all__"][0]


# ----------------------------------------------------------------------- CSRF
def test_post_views_require_csrf(scenario: Simulation, started: list[int]) -> None:
    client = Client(enforce_csrf_checks=True)
    study = make_study(scenario, status=HPOStatus.FINISHED, best_params={"gamma": 0.95})
    assert (
        post_json(client, reverse("hpo_create"), payload(scenario)).status_code == 403
    )
    assert client.post(reverse("hpo_cancel", args=[study.pk])).status_code == 403
    assert (
        post_json(
            client, reverse("hpo_apply", args=[study.pk]), {"target": "default"}
        ).status_code
        == 403
    )
    assert started == []

    page = client.get(reverse("hpo_create"))
    token = page.cookies["csrftoken"].value
    response = client.post(
        reverse("hpo_create"),
        data=json.dumps(payload(scenario)),
        content_type="application/json",
        HTTP_X_CSRFTOKEN=token,
    )
    assert response.status_code == 201


def test_get_not_allowed_on_post_views(scenario: Simulation) -> None:
    study = make_study(scenario)
    assert Client().get(reverse("hpo_cancel", args=[study.pk])).status_code == 405
    assert Client().get(reverse("hpo_apply", args=[study.pk])).status_code == 405


# --------------------------------------------------------------------- status
def test_status_json_shape_history_scatter_and_progress(scenario: Simulation) -> None:
    study = make_study(
        scenario,
        status=HPOStatus.RUNNING,
        started_at=timezone.now(),
        heartbeat_at=timezone.now(),
        objective={"mode": "weighted", "weights": [1.0, 1.0]},
    )
    add_trial(
        study, source_index=0, params={"gamma": 0.91}, agent_score=1.0, market_score=0.0
    )
    add_trial(
        study, source_index=1, params={"gamma": 0.95}, agent_score=0.0, market_score=2.0
    )
    add_trial(
        study,
        iteration=1,
        phase=HPOPhase.EMPLOYED,
        source_index=0,
        params={"gamma": 0.92},
        agent_score=0.5,
        market_score=-0.1,
    )
    add_trial(
        study,
        iteration=1,
        phase=HPOPhase.EMPLOYED,
        source_index=1,
        params={"gamma": 0.96},
        status=HPOTrialStatus.FAILED,
        error="boom",
    )
    add_trial(
        study,
        iteration=1,
        phase=HPOPhase.ONLOOKER,
        source_index=1,
        params={"gamma": 0.97},
        status=HPOTrialStatus.RUNNING,
    )
    response = Client().get(reverse("hpo_status", args=[study.pk]))
    assert response.status_code == 200
    data = response.json()
    assert set(data) >= {
        "status",
        "started_at",
        "finished_at",
        "progress",
        "best_score",
        "best_params",
        "history",
        "scatter",
        "trials",
        "error",
    }
    assert data["status"] == "RUNNING"
    # (2 + 1 * 2 * 2 + 0 scouts) * 1 search seed + 1 evaluation seed.
    progress = data["progress"]
    assert (progress["done"], progress["total"]) == (4, 7)
    assert set(progress) == {"done", "total", "round", "elapsed_seconds", "remaining_seconds"}
    # Iteration 0 best is 2.0 (0 + 2), iteration 1 adds 0.4 (dominated by the first trial), best stays 2.0.
    assert data["history"] == [[0, 2.0], [1, 2.0]]
    assert data["search_best"] == {"score": 2.0, "params": {"gamma": 0.95}}
    assert data["best_score"] is None and data["best_params"] is None
    scatter = {(p[0], p[1]): (p[2], p[3]) for p in data["scatter"]}
    assert scatter == {
        (1.0, 0.0): ("INIT", True),
        (0.0, 2.0): ("INIT", True),
        (0.5, -0.1): ("EMPLOYED", False),
    }
    assert [t["status"] for t in data["trials"]] == [
        "RUNNING",
        "FAILED",
        "FINISHED",
        "FINISHED",
        "FINISHED",
    ]
    failed = data["trials"][1]
    assert failed["agent_score"] is None and failed["error"] == "boom"
    assert set(failed) == {
        "id",
        "iteration",
        "phase",
        "source_index",
        "seed",
        "params",
        "agent_score",
        "market_score",
        "status",
        "duration_s",
        "simulation_id",
        "error",
    }


def test_status_candidate_needs_all_search_seeds(scenario: Simulation) -> None:
    study = make_study(
        scenario,
        status=HPOStatus.RUNNING,
        heartbeat_at=timezone.now(),
        search_seeds=[1, 2],
    )
    add_trial(study, seed=1, agent_score=3.0, market_score=0.0)
    data = Client().get(reverse("hpo_status", args=[study.pk])).json()
    assert data["history"] == [] and data["search_best"] is None
    add_trial(study, seed=2, agent_score=1.0, market_score=0.0)
    data = Client().get(reverse("hpo_status", args=[study.pk])).json()
    assert data["history"] == [[0, 2.0]]


def test_status_finished_reports_final_eval_and_scout_total(
    scenario: Simulation,
) -> None:
    study = make_study(
        scenario,
        status=HPOStatus.FINISHED,
        best_score=1.5,
        best_params={"gamma": 0.93},
        finished_at=timezone.now(),
    )
    add_trial(
        study,
        iteration=1,
        phase=HPOPhase.SCOUT,
        source_index=0,
        agent_score=0.2,
        market_score=0.0,
    )
    add_trial(
        study,
        iteration=1,
        phase=HPOPhase.FINAL_EVAL,
        source_index=0,
        seed=9,
        params={"gamma": 0.93},
        agent_score=1.5,
        market_score=0.0,
    )
    data = Client().get(reverse("hpo_status", args=[study.pk])).json()
    assert data["best_score"] == 1.5 and data["best_params"] == {"gamma": 0.93}
    assert data["progress"]["total"] == (2 + 4 + 1) * 1 + 1
    final = [p for p in data["scatter"] if p[2] == "FINAL_EVAL"]
    assert final == [[1.5, 0.0, "FINAL_EVAL", False]]


def test_status_fails_stale_study(scenario: Simulation, settings: Any) -> None:
    settings.ABMEM_HPO_HEARTBEAT_TIMEOUT = 60
    study = make_study(
        scenario,
        status=HPOStatus.RUNNING,
        heartbeat_at=timezone.now() - timedelta(seconds=120),
    )
    data = Client().get(reverse("hpo_status", args=[study.pk])).json()
    assert data["status"] == "FAILED"
    assert "heartbeat" in data["error"]


def test_status_unknown_study_is_404() -> None:
    assert Client().get(reverse("hpo_status", args=[424242])).status_code == 404


# --------------------------------------------------------------------- cancel
def test_cancel_running_sets_flag(scenario: Simulation) -> None:
    study = make_study(scenario, status=HPOStatus.RUNNING, heartbeat_at=timezone.now())
    response = Client().post(reverse("hpo_cancel", args=[study.pk]))
    assert response.status_code == 200
    assert response.json() == {"status": "RUNNING", "cancel_requested": True}
    study.refresh_from_db()
    assert study.cancel_requested is True


def test_cancel_created_study_is_canceled_at_once(scenario: Simulation) -> None:
    study = make_study(scenario)
    response = Client().post(reverse("hpo_cancel", args=[study.pk]))
    assert response.json()["status"] == "CANCELED"


def test_cancel_finished_study_is_rejected(scenario: Simulation) -> None:
    study = make_study(scenario, status=HPOStatus.FINISHED)
    errors = _errors(Client().post(reverse("hpo_cancel", args=[study.pk])))
    assert "FINISHED" in errors["status"][0]


# ---------------------------------------------------------------------- apply
def finished_study(scenario: Simulation, **fields: Any) -> HPOStudy:
    """FINISHED DDPG study with best parameters."""
    values: dict[str, Any] = {
        "status": HPOStatus.FINISHED,
        "best_score": 1.0,
        "best_params": {"gamma": 0.95, "gradient_steps": 3},
    }
    values.update(fields)
    return make_study(scenario, **values)


def test_apply_requires_finished_study(scenario: Simulation) -> None:
    study = make_study(scenario, status=HPOStatus.RUNNING, heartbeat_at=timezone.now())
    errors = _errors(
        post_json(
            Client(), reverse("hpo_apply", args=[study.pk]), {"target": "default"}
        )
    )
    assert "status" in errors


def test_apply_rejects_unknown_target(scenario: Simulation) -> None:
    study = finished_study(scenario)
    errors = _errors(
        post_json(Client(), reverse("hpo_apply", args=[study.pk]), {"target": "x"})
    )
    assert "target" in errors


def test_apply_as_algorithm_default_merges(scenario: Simulation) -> None:
    AlgorithmDefault.objects.create(
        algorithm_type="DDPG", params={"tau": 0.01, "gamma": 0.9}
    )
    study = finished_study(
        scenario,
        best_params={"gamma": 0.95, "gradient_steps": 3, "failstack_delta": 12.0},
        fixed_params={"failstack_enabled": True},
    )
    response = post_json(
        Client(), reverse("hpo_apply", args=[study.pk]), {"target": "default"}
    )
    assert response.status_code == 200, response.content
    row = AlgorithmDefault.objects.get(algorithm_type="DDPG")
    assert row.params == {"tau": 0.01, "gamma": 0.95, "gradient_steps": 3}
    assert response.json()["not_applied"] == ["failstack_delta"]
    # Scenario agents are untouched.
    assert Agent.objects.get(name="a0", market=scenario.market).algorithm_params == {
        "batch_size": 8
    }


def test_apply_as_default_creates_row(scenario: Simulation) -> None:
    study = finished_study(scenario)
    response = post_json(
        Client(), reverse("hpo_apply", args=[study.pk]), {"target": "default"}
    )
    assert response.status_code == 200
    assert AlgorithmDefault.objects.get(algorithm_type="DDPG").params == {
        "gamma": 0.95,
        "gradient_steps": 3,
    }


def test_apply_as_default_marks_the_row_optimized(scenario: Simulation) -> None:
    AlgorithmDefault.objects.create(algorithm_type="DDPG", params={"tau": 0.01})
    study = finished_study(scenario)
    response = post_json(
        Client(), reverse("hpo_apply", args=[study.pk]), {"target": "default"}
    )
    assert response.status_code == 200, response.content
    row = AlgorithmDefault.objects.get(algorithm_type="DDPG")
    assert row.origin == AlgorithmDefaultOrigin.OPTIMIZED
    assert row.source_study_id == study.pk
    # A second study replaces the source.
    other = finished_study(scenario, name="second")
    post_json(Client(), reverse("hpo_apply", args=[other.pk]), {"target": "default"})
    row.refresh_from_db()
    assert row.source_study_id == other.pk
    # Applying to the scenario does not touch the default row.
    third = finished_study(scenario, name="third")
    post_json(Client(), reverse("hpo_apply", args=[third.pk]), {"target": "scenario"})
    row.refresh_from_db()
    assert row.source_study_id == other.pk


def test_apply_to_scenario_switches_all_agents(scenario: Simulation) -> None:
    study = finished_study(
        scenario,
        best_params={"gamma": 0.95, "gradient_steps": 3, "failstack_delta": 12.0},
        fixed_params={"failstack_enabled": True, "failstack_limit": 4},
    )
    response = post_json(
        Client(),
        reverse("hpo_apply", args=[study.pk]),
        {"target": "scenario", "switch_algorithm": True},
    )
    assert response.status_code == 200, response.content
    assert response.json()["agents_updated"] == ["a0", "a1"]
    a0, a1 = Agent.objects.filter(market=scenario.market).order_by("id")
    assert a0.algorithm_params == {"batch_size": 8, "gamma": 0.95, "gradient_steps": 3}
    assert a1.algorithm_type == "DDPG"
    assert a1.algorithm_params == {"gamma": 0.95, "gradient_steps": 3}
    for agent in (a0, a1):
        assert agent.failstack_enabled is True
        assert agent.failstack_limit == 4
        assert agent.failstack_delta == Decimal("12.00")


def test_apply_to_scenario_only_matching_agents(scenario: Simulation) -> None:
    study = finished_study(scenario)
    response = post_json(
        Client(),
        reverse("hpo_apply", args=[study.pk]),
        {"target": "scenario", "switch_algorithm": False},
    )
    assert response.status_code == 200
    assert response.json()["agents_updated"] == ["a0"]
    a1 = Agent.objects.get(market=scenario.market, name="a1")
    assert a1.algorithm_type == "FIXED_MARKUP"
    assert a1.algorithm_params == {"markup": 0.2}


def test_apply_to_scenario_without_matching_agent_is_rejected(
    scenario: Simulation,
) -> None:
    study = finished_study(
        scenario, algorithm_type="TD3", best_params={"gamma": 0.95}
    )
    errors = _errors(
        post_json(
            Client(), reverse("hpo_apply", args=[study.pk]), {"target": "scenario"}
        )
    )
    assert "TD3" in errors["target"][0]
    assert not Agent.objects.filter(algorithm_type="TD3").exists()


def test_apply_to_scenario_blocked_while_another_study_runs(
    scenario: Simulation,
) -> None:
    study = finished_study(scenario)
    make_study(scenario, status=HPOStatus.RUNNING, heartbeat_at=timezone.now())
    errors = _errors(
        post_json(
            Client(),
            reverse("hpo_apply", args=[study.pk]),
            {"target": "scenario", "switch_algorithm": True},
        )
    )
    assert "running" in errors["target"][0]


def test_apply_rejects_invalid_params_without_writing(scenario: Simulation) -> None:
    study = finished_study(scenario, best_params={"gamma": 2.0})
    errors = _errors(
        post_json(
            Client(), reverse("hpo_apply", args=[study.pk]), {"target": "default"}
        )
    )
    assert "gamma" in errors["target"][0]
    assert not AlgorithmDefault.objects.exists()


# ------------------------------------------------------------ run list filter
def test_simulation_list_and_dashboard_hide_hpo_runs(scenario: Simulation) -> None:
    user_run = make_simulation(name="user-run-1")
    hpo_run = make_simulation(name="hpo-eval-run")
    study = make_study(scenario)
    add_trial(study, simulation=hpo_run, agent_score=1.0, market_score=0.0)
    content = Client().get(reverse("simulation_list")).content.decode()
    assert user_run.name in content
    assert hpo_run.name not in content
    response = Client().get(reverse("dashboard"))
    assert response.context["num_simulations"] == 1
    assert [s.name for s in response.context["latest_simulations"]] == ["user-run-1"]


# ------------------------------------------------- target mode (PLAN 2.15)
NODE = shutil.which("node")


def _json_script(content: str, element_id: str) -> Any:
    match = re.search(
        rf'<script id="{element_id}" type="application/json">(.*?)</script>',
        content,
        flags=re.S,
    )
    assert match, element_id
    return json.loads(match.group(1))


def test_create_page_embeds_scenario_algorithms(scenario: Simulation) -> None:
    content = Client().get(reverse("hpo_create")).content.decode()
    assert 'id="hpo-target-matching"' in content
    assert 'id="hpo-target-matching" checked' not in content
    assert _json_script(content, "hpo-scenario-algorithms") == {
        str(scenario.pk): {"DDPG": 1}
    }
    options = _json_script(content, "hpo-algorithm-options")
    names = [option["name"] for option in options]
    assert "DDPG" in names and "TD3" in names and "RANDOM" not in names
    checked = Client().get(reverse("hpo_create") + "?target=matching")
    assert 'id="hpo-target-matching" checked' in checked.content.decode()


def test_scenario_algorithms_helper_filters_searchable(scenario: Simulation) -> None:
    from abmem.views_hpo import scenario_algorithms

    assert scenario_algorithms([scenario], ["DDPG", "TD3"]) == {
        str(scenario.pk): {"DDPG": 1}
    }
    assert scenario_algorithms([scenario], ["TD3"]) == {str(scenario.pk): {}}


def test_create_matching_study(scenario: Simulation, started: list[int]) -> None:
    response = post_json(
        Client(), reverse("hpo_create"), payload(scenario, target="MATCHING")
    )
    assert response.status_code == 201, response.content
    study = HPOStudy.objects.get(pk=response.json()["id"])
    assert study.target == HPOTarget.MATCHING
    assert started == [study.pk]


def test_create_without_target_is_all(scenario: Simulation) -> None:
    response = post_json(Client(), reverse("hpo_create"), payload(scenario))
    assert response.status_code == 201, response.content
    assert HPOStudy.objects.get().target == HPOTarget.ALL


@pytest.mark.parametrize(
    ("changes", "text"),
    [
        (
            {
                "algorithm_type": "TD3",
                "param_bounds": {"gamma": {"low": "0.9", "high": "0.99"}},
            },
            "No agent of the scenario uses TD3",
        ),
        (
            {
                "algorithm_type": "FIXED_MARKUP",
                "param_bounds": {"markup": {"low": "0.0", "high": "0.5"}},
            },
            "learning algorithm",
        ),
    ],
)
def test_create_matching_rejects(
    scenario: Simulation, changes: dict[str, Any], text: str
) -> None:
    errors = _errors(
        post_json(
            Client(),
            reverse("hpo_create"),
            payload(scenario, target="MATCHING", **changes),
        )
    )
    assert text in errors["algorithm_type"][0]
    assert not HPOStudy.objects.exists()


def test_form_matching_validation(scenario: Simulation) -> None:
    from abmem.forms_hpo import HPOStudyForm

    data = payload(scenario, target="MATCHING", algorithm_type="TD3")
    data["param_bounds"] = {"gamma": {"low": "0.9", "high": "0.99"}}
    form = HPOStudyForm(data)
    assert not form.is_valid()
    assert "TD3" in form.errors["algorithm_type"][0]
    data["target"] = "ALL"
    form = HPOStudyForm(data)
    assert form.is_valid(), form.errors
    assert form.to_service_input()["target"] == "ALL"
    form = HPOStudyForm(payload(scenario, target="MATCHING"))
    assert form.is_valid(), form.errors
    assert form.to_service_input()["target"] == "MATCHING"


def test_list_watch_and_status_show_target(scenario: Simulation) -> None:
    matching = make_study(scenario, name="m-study", target=HPOTarget.MATCHING)
    all_study = make_study(scenario, name="a-study")
    content = Client().get(reverse("hpo_list")).content.decode()
    assert "Only matching agents" in content and "All agents" in content

    watch = Client().get(reverse("hpo_watch", args=[matching.pk]))
    assert watch.context["tuned_agents"] == ["a0"]
    assert watch.context["matching"] is True
    text = watch.content.decode()
    assert "<strong>Mode:</strong> Only matching agents" in text
    assert 'id="hpo-switch-algorithm" checked' not in text

    watch = Client().get(reverse("hpo_watch", args=[all_study.pk]))
    assert watch.context["tuned_agents"] == ["a0", "a1"]
    assert 'id="hpo-switch-algorithm" checked' in watch.content.decode()

    status = Client().get(reverse("hpo_status", args=[matching.pk])).json()
    assert status["target"] == "MATCHING"


def test_apply_to_scenario_matching_ignores_switch(scenario: Simulation) -> None:
    study = finished_study(
        scenario,
        target=HPOTarget.MATCHING,
        fixed_params={"failstack_enabled": True, "failstack_limit": 3},
    )
    before = Agent.objects.get(market=scenario.market, name="a1")
    response = post_json(
        Client(),
        reverse("hpo_apply", args=[study.pk]),
        {"target": "scenario", "switch_algorithm": True},
    )
    assert response.status_code == 200, response.content
    assert response.json()["agents_updated"] == ["a0"]
    a0, a1 = Agent.objects.filter(market=scenario.market).order_by("id")
    assert a0.algorithm_params == {"batch_size": 8, "gamma": 0.95, "gradient_steps": 3}
    assert a0.failstack_enabled is True and a0.failstack_limit == 3
    assert (a1.algorithm_type, a1.algorithm_params) == ("FIXED_MARKUP", {"markup": 0.2})
    assert (a1.failstack_enabled, a1.failstack_limit) == (
        before.failstack_enabled,
        before.failstack_limit,
    )


def test_apply_to_scenario_matching_without_agent_is_rejected(
    scenario: Simulation,
) -> None:
    study = finished_study(
        scenario,
        target=HPOTarget.MATCHING,
        algorithm_type="TD3",
        best_params={"gamma": 0.95},
    )
    errors = _errors(
        post_json(
            Client(),
            reverse("hpo_apply", args=[study.pk]),
            {"target": "scenario", "switch_algorithm": True},
        )
    )
    assert errors["target"] == ["No agent of the scenario uses TD3."]
    assert not Agent.objects.filter(algorithm_type="TD3").exists()


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_hpo_inline_scripts_pass_node_check(
    scenario: Simulation, tmp_path: pathlib.Path
) -> None:
    study = make_study(scenario, target=HPOTarget.MATCHING)
    pages = {
        "create": reverse("hpo_create"),
        "create_matching": reverse("hpo_create") + "?target=MATCHING",
        "watch": reverse("hpo_watch", args=[study.pk]),
    }
    for name, url in pages.items():
        html = Client().get(url).content.decode()
        scripts = [
            body
            for attrs, body in re.findall(
                r"<script([^>]*)>(.*?)</script>", html, flags=re.S
            )
            if "src=" not in attrs and "application/json" not in attrs and body.strip()
        ]
        assert scripts, name
        for index, body in enumerate(scripts):
            path = tmp_path / f"{name}_{index}.js"
            path.write_text(body, encoding="utf-8")
            result = subprocess.run(
                [NODE, "--check", str(path)], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, f"{name} script {index}: {result.stderr}"
