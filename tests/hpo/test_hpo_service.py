"""HPO service: validation, lifecycle, end-to-end study with serial evaluation (PLAN 2.4)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
import torch
from django.core.exceptions import ValidationError
from django.utils import timezone

from abmem import bootstrap
from abmem.models import Agent, HPOStudy, HPOTrial, Market, Simulation
from abmem.models.enums import (
    AgentType,
    AlgorithmType,
    HPOPhase,
    HPOStatus,
    HPOTrialStatus,
    MarketStrategy,
    PeriodType,
    SimulationMode,
)
from abmem.models.models import Period, Plant, Portfolio
from abmem.services.hpo import hpo_service, worker
from abmem.services.hpo.worker import EvaluationTask, TrialOutcome, split_params
from abmem.services.market.demand import DemandSeries
from abmem.services.simulation import simulation_service
from tests.factories import make_resource

pytestmark = pytest.mark.django_db


@pytest.fixture(autouse=True)
def _restore_torch_determinism() -> Iterator[None]:
    """Undo ``torch.use_deterministic_algorithms(True)`` set by seeding."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


@pytest.fixture(autouse=True)
def fast_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Excel demand file with an in-memory series."""
    t = np.arange(400, dtype=np.float64)
    series = DemandSeries(
        values=np.round(1000.0 + 100.0 * np.sin(2 * np.pi * t / 24.0), 2), source="test"
    )
    monkeypatch.setattr(
        DemandSeries, "from_excel", staticmethod(lambda *a, **k: series)
    )


def make_scenario(period_number: int = 10, evaluation_periods: int = 0) -> Simulation:
    """Proxy scenario with two RANDOM agents (each with its own algorithm params)."""
    sim = Simulation.objects.create(
        name="scn",
        mode=SimulationMode.ONLYRESULT,
        periodType=PeriodType.HOUR,
        periodNumber=period_number,
        currentPeriod=-1,
        day=0,
        seed=3,
        proxy=True,
        evaluation_periods=evaluation_periods,
    )
    market = Market.objects.create(
        simulation=sim,
        strategy=MarketStrategy.PAYASBID,
        lowerBidBound=0,
        upperBidBound=200,
        proxy=True,
    )
    for i in range(2):
        res = make_resource(fuel_cost=Decimal(10 + 5 * i), om_cost=Decimal("1"))
        agent = Agent.objects.create(
            name=f"a{i}",
            market=market,
            proxy=True,
            type=AgentType.HYBRID,
            algorithm_type=AlgorithmType.DDPG if i == 0 else AlgorithmType.FIXED_MARKUP,
            algorithm_params={"batch_size": 8, "learning_starts": 8} if i == 0 else {},
        )
        portfolio = Portfolio.objects.create(agent=agent, proxy=True)
        Plant.objects.create(
            portfolio=portfolio, resource=res, capacity=700, proxy=True
        )
    return sim


def study_input(scenario: Simulation, **changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "tiny",
        "scenario": scenario.pk,
        "algorithm_type": "FIXED_MARKUP",
        "param_bounds": {"markup": {"low": 0.0, "high": 0.5, "scale": "linear"}},
        "colony_size": 2,
        "max_iter": 1,
        "limit": 100,
        "search_seeds": [11],
        "evaluation_seeds": [22],
        "objective": {"mode": "weighted", "weights": [1.0, 0.5]},
    }
    data.update(changes)
    return data


# ----------------------------------------------------------------- validation
def test_validate_accepts_and_normalises() -> None:
    scenario = make_scenario()
    out = hpo_service.validate_study_input(study_input(scenario, search_seeds="1, 2"))
    assert out["algorithm_type"] == AlgorithmType.FIXED_MARKUP
    assert out["scenario"] == scenario
    assert out["search_seeds"] == [1, 2]
    assert out["fixed_params"] == {"failstack_enabled": False}
    assert out["objective"] == {"mode": "weighted", "weights": [1.0, 0.5]}


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"colony_size": 1}, "colony_size"),
        ({"max_iter": 0}, "max_iter"),
        ({"search_seeds": []}, "search_seeds"),
        ({"search_seeds": [1, 1]}, "search_seeds"),
        ({"search_seeds": [2**31]}, "search_seeds"),
        ({"search_seeds": [-1]}, "search_seeds"),
        ({"evaluation_seeds": [11]}, "evaluation_seeds"),
        ({"param_bounds": {"markup": {"low": 0.5, "high": 0.1}}}, "param_bounds"),
        ({"param_bounds": {"markup": {"low": -2.0, "high": 0.1}}}, "param_bounds"),
        ({"param_bounds": {"nope": {"low": 0.0, "high": 1.0}}}, "param_bounds"),
        ({"param_bounds": {}}, "param_bounds"),
        (
            {
                "param_bounds": {
                    "failstack_limit": {"low": 2, "high": 9, "scale": "int"}
                }
            },
            "param_bounds",
        ),
        ({"objective": {"mode": "pareto"}}, "objective"),
        ({"algorithm_type": "XYZ"}, "algorithm_type"),
        ({"name": ""}, "name"),
    ],
)
def test_validate_rejects(changes: dict[str, Any], field: str) -> None:
    scenario = make_scenario()
    with pytest.raises(ValidationError) as info:
        hpo_service.validate_study_input(study_input(scenario, **changes))
    assert field in info.value.message_dict


def test_validate_rejects_non_proxy_scenario_and_log_bound_low_zero() -> None:
    scenario = make_scenario()
    run = Simulation.objects.create(
        name="run",
        mode=SimulationMode.ONLYRESULT,
        periodType=PeriodType.HOUR,
        periodNumber=5,
        currentPeriod=-1,
        day=0,
    )
    with pytest.raises(ValidationError) as info:
        hpo_service.validate_study_input(study_input(run))
    assert "scenario" in info.value.message_dict
    with pytest.raises(ValidationError) as info:
        hpo_service.validate_study_input(
            study_input(
                scenario,
                algorithm_type="DDPG",
                param_bounds={"lr_actor": {"low": 0.0, "high": 1e-3, "scale": "log"}},
            )
        )
    assert "param_bounds" in info.value.message_dict


def test_validate_failstack_bounds_when_enabled() -> None:
    scenario = make_scenario()
    out = hpo_service.validate_study_input(
        study_input(
            scenario,
            algorithm_type=AlgorithmType.DDPG,
            fixed_params={"failstack_enabled": True, "failstack_delta": 10},
            param_bounds={
                "gamma": [0.9, 0.99, "linear"],
                "failstack_limit": {"low": 2, "high": 9, "scale": "int"},
            },
        )
    )
    assert set(out["param_bounds"]) == {"gamma", "failstack_limit"}
    assert out["fixed_params"] == {"failstack_enabled": True, "failstack_delta": 10.0}


def test_split_params() -> None:
    overrides, fields = split_params(
        {"markup": 0.2, "failstack_limit": 4.0},
        {"failstack_enabled": True, "failstack_delta": 12.345},
    )
    assert overrides == {"markup": 0.2}
    assert fields == {
        "failstack_enabled": True,
        "failstack_limit": 4,
        "failstack_delta": Decimal("12.34"),
    }
    assert split_params({"markup": 0.1}, {}) == (
        {"markup": 0.1},
        {"failstack_enabled": False},
    )


def test_param_space_payload() -> None:
    payload = hpo_service.param_space_payload(AlgorithmType.TD3)
    assert payload["algorithm"] == "TD3"
    assert "gradient_steps" in payload["bounds"]
    assert payload["bounds"]["gradient_steps"]["scale"] == "int"
    assert payload["defaults"]["gradient_steps"] == 1
    assert payload["failstack_defaults"] == {
        "failstack_limit": 6,
        "failstack_delta": 15.0,
    }


# ------------------------------------------------------------------ lifecycle
def test_start_spawns_process_with_study_id(monkeypatch: pytest.MonkeyPatch) -> None:
    study = hpo_service.create_study(study_input(make_scenario()))
    started: list[tuple[Any, tuple]] = []

    class FakeProcess:
        def __init__(self, target: Any, args: tuple) -> None:
            self.target, self.args = target, args

        def start(self) -> None:
            started.append((self.target, self.args))

    class FakeContext:
        Process = FakeProcess

    monkeypatch.setattr(
        hpo_service.multiprocessing, "get_context", lambda m: FakeContext()
    )
    hpo_service.start(study)
    assert started == [(bootstrap.run_hpo_process, (study.id,))]


def test_request_cancel_on_created_study() -> None:
    study = hpo_service.create_study(study_input(make_scenario()))
    hpo_service.request_cancel(study)
    assert study.status == HPOStatus.CANCELED and study.cancel_requested
    hpo_service.run_study(study.pk, task_runner=hpo_service.serial_runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.CANCELED
    assert not HPOTrial.objects.filter(study=study).exists()


def test_stale_running_study_becomes_failed() -> None:
    scenario = make_scenario()
    stale = hpo_service.create_study(study_input(scenario))
    fresh = hpo_service.create_study(study_input(scenario, name="fresh"))
    now = timezone.now()
    HPOStudy.objects.filter(pk=stale.pk).update(
        status=HPOStatus.RUNNING, heartbeat_at=now - timedelta(hours=2)
    )
    HPOStudy.objects.filter(pk=fresh.pk).update(
        status=HPOStatus.RUNNING, heartbeat_at=now - timedelta(minutes=1)
    )
    assert hpo_service.fail_stale_studies() == 1
    stale.refresh_from_db()
    fresh.refresh_from_db()
    assert stale.status == HPOStatus.FAILED and "heartbeat" in stale.error
    assert fresh.status == HPOStatus.RUNNING


def test_heartbeat_timeout_setting(settings: Any) -> None:
    settings.ABMEM_HPO_HEARTBEAT_TIMEOUT = 60
    assert hpo_service.heartbeat_timeout() == timedelta(seconds=60)


# ----------------------------------------------------------------- end to end
def recording_runner(log: list[Any]) -> Callable[..., list[TrialOutcome]]:
    def runner(tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]):
        log.append(
            HPOStudy.objects.values_list("status", "heartbeat_at").get(
                pk=tasks[0].study_id
            )
        )
        return hpo_service.serial_runner(tasks, is_canceled)

    return runner


def test_tiny_study_runs_end_to_end() -> None:
    scenario = make_scenario(evaluation_periods=3)
    study = hpo_service.create_study(study_input(scenario))
    assert study.status == HPOStatus.CREATED
    log: list[Any] = []
    hpo_service.run_study(study.pk, task_runner=recording_runner(log))
    study.refresh_from_db()
    assert study.status == HPOStatus.FINISHED, study.error
    assert study.started_at and study.finished_at and study.heartbeat_at
    assert all(status == HPOStatus.RUNNING and hb is not None for status, hb in log)

    trials = list(HPOTrial.objects.filter(study=study).order_by("id"))
    phases = [t.phase for t in trials]
    # INIT 2 + EMPLOYED 2 + ONLOOKER 2 (no scout, limit 100) + FINAL_EVAL 1, one seed each.
    assert len(trials) == 7
    assert phases.count(HPOPhase.FINAL_EVAL) == 1
    assert all(t.status == HPOTrialStatus.FINISHED for t in trials)
    final = trials[-1]
    assert final.phase == HPOPhase.FINAL_EVAL and final.seed == 22
    assert all(t.seed == 11 for t in trials[:-1])
    assert study.best_score == pytest.approx(
        final.agent_score + 0.5 * final.market_score
    )
    assert study.best_params == final.params
    assert 0.0 <= study.best_params["markup"] <= 0.5

    # K7: every agent of every evaluation run uses the study algorithm, and the
    # DDPG-specific params of scenario agent a0 do not reach the FIXED_MARKUP agents.
    for trial in trials:
        agents = trial.simulation.market.agent_set.all()
        assert {a.algorithm_type for a in agents} == {AlgorithmType.FIXED_MARKUP}
        assert all(a.algorithm_params == trial.params for a in agents)
        assert trial.simulation.seed == trial.seed
        assert not trial.simulation.proxy

    # K9: only the best candidate's search trials and the final evaluation keep periods.
    kept = {
        t.pk
        for t in trials
        if Period.objects.filter(market__simulation=t.simulation).exists()
    }
    assert final.pk in kept
    best_search = [t for t in trials[:-1] if t.params == study.best_params]
    assert kept == {final.pk} | {t.pk for t in best_search}


def test_cancel_between_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    study = hpo_service.create_study(study_input(make_scenario()))

    def runner(tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]):
        outcomes = hpo_service.serial_runner(tasks, is_canceled)
        HPOStudy.objects.filter(pk=study.pk).update(cancel_requested=True)
        return outcomes

    hpo_service.run_study(study.pk, task_runner=runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.CANCELED
    assert study.finished_at is not None
    assert HPOTrial.objects.filter(study=study).count() == 2  # INIT batch only


def test_cancel_during_batch_marks_unfinished_trials() -> None:
    study = hpo_service.create_study(study_input(make_scenario()))

    def runner(tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]):
        raise hpo_service.StudyCanceled()

    hpo_service.run_study(study.pk, task_runner=runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.CANCELED
    trials = HPOTrial.objects.filter(study=study)
    assert trials.count() == 2
    assert all(
        t.status == HPOTrialStatus.FAILED and t.error == "canceled" for t in trials
    )


def test_runner_exception_sets_failed_with_traceback() -> None:
    study = hpo_service.create_study(study_input(make_scenario()))

    def runner(tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]):
        raise RuntimeError("pool exploded")

    hpo_service.run_study(study.pk, task_runner=runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.FAILED
    assert "pool exploded" in study.error and "Traceback" in study.error
    assert len(study.error) <= 4000


def test_worker_failure_marks_trial_failed_and_study_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = hpo_service.create_study(study_input(make_scenario()))
    real_run = simulation_service.run
    calls = {"n": 0}

    def flaky_run(sim: Simulation, **kwargs: Any) -> Simulation:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected failure")
        return real_run(sim, **kwargs)

    monkeypatch.setattr(simulation_service, "run", flaky_run)
    hpo_service.run_study(study.pk, task_runner=hpo_service.serial_runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.FINISHED, study.error
    failed = HPOTrial.objects.get(study=study, status=HPOTrialStatus.FAILED)
    assert "injected failure" in failed.error
    assert failed.agent_score is None
    assert failed.duration_s is not None
    assert HPOTrial.objects.filter(study=study).count() == 7


def test_all_evaluations_of_a_batch_failing_fails_the_study(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = hpo_service.create_study(study_input(make_scenario()))

    def broken_run(sim: Simulation, **kwargs: Any) -> Simulation:
        raise RuntimeError("always broken")

    monkeypatch.setattr(simulation_service, "run", broken_run)
    hpo_service.run_study(study.pk, task_runner=hpo_service.serial_runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.FAILED
    assert "all 2 evaluations of the batch failed" in study.error


def test_run_study_ignores_non_created_study() -> None:
    study = hpo_service.create_study(study_input(make_scenario()))
    HPOStudy.objects.filter(pk=study.pk).update(status=HPOStatus.RUNNING)
    hpo_service.run_study(study.pk, task_runner=hpo_service.serial_runner)
    assert not HPOTrial.objects.filter(study=study).exists()


def test_evaluate_task_outcome_objectives() -> None:
    failed = TrialOutcome(trial_id=1, ok=False, error="x")
    assert failed.objectives == (float("-inf"), float("-inf"))
    assert TrialOutcome(
        trial_id=1, ok=True, agent_score=0.3, market_score=0.1
    ).objectives == (
        0.3,
        0.1,
    )
    assert worker.evaluate_task.__module__ == "abmem.services.hpo.worker"


class _FakeAsync:
    def __init__(self, result: list[Any] | None) -> None:
        self.result = result

    def wait(self, timeout: float) -> None:
        return None

    def ready(self) -> bool:
        return self.result is not None

    def get(self) -> list[Any]:
        assert self.result is not None
        return self.result


class _FakePool:
    instances: list["_FakePool"] = []

    def __init__(self, result: list[Any] | None, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.result = result
        self.terminated = False
        _FakePool.instances.append(self)

    def map_async(self, func: Any, items: list[Any], chunksize: int) -> _FakeAsync:
        assert func is worker.evaluate_task and chunksize == 1
        return _FakeAsync(self.result)

    def close(self) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True

    def join(self) -> None:
        pass


@pytest.mark.parametrize("canceled", [False, True])
def test_pool_runner_uses_spawn_pool_and_terminates_on_cancel(
    monkeypatch: pytest.MonkeyPatch, canceled: bool
) -> None:
    tasks = [
        EvaluationTask(study_id=1, trial_id=i, seed=i, params={}) for i in (1, 2, 3)
    ]
    outcomes = [
        TrialOutcome(trial_id=t.trial_id, ok=True, agent_score=0.0, market_score=0.0)
        for t in tasks
    ]
    _FakePool.instances.clear()

    class FakeContext:
        def Pool(self, **kwargs: Any) -> _FakePool:  # noqa: N802
            return _FakePool(None if canceled else outcomes, **kwargs)

    contexts: list[str] = []

    def get_context(method: str) -> FakeContext:
        contexts.append(method)
        return FakeContext()

    monkeypatch.setattr(hpo_service.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(hpo_service.os, "cpu_count", lambda: 2)
    if canceled:
        with pytest.raises(hpo_service.StudyCanceled):
            hpo_service.pool_runner(tasks, lambda: True)
    else:
        assert hpo_service.pool_runner(tasks, lambda: False) == outcomes
    pool = _FakePool.instances[0]
    assert contexts == ["spawn"]
    assert pool.kwargs == {
        "processes": 2,
        "initializer": bootstrap.init_hpo_worker,
        "maxtasksperchild": 1,
    }
    assert pool.terminated


def test_run_hpo_command_creates_and_runs_study(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import json

    from django.core.management import call_command

    monkeypatch.setattr(hpo_service, "pool_runner", hpo_service.serial_runner)
    config = tmp_path / "study.json"
    config.write_text(json.dumps(study_input(make_scenario())), encoding="utf-8")
    call_command("run_hpo", "--config", str(config))
    study = HPOStudy.objects.get(name="tiny")
    assert study.status == HPOStatus.FINISHED, study.error
