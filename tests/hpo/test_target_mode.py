"""HPO target mode: tune all agents or only the agents of the study algorithm (PLAN 2.15)."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
import torch
from django.core.exceptions import ValidationError

from abmem.models import Agent, HPOStudy, HPOTarget, HPOTrial, Market, Simulation
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
from abmem.services.hpo import hpo_service, worker
from abmem.services.hpo.objective import score_simulation
from abmem.services.hpo.worker import EvaluationTask, target_algorithms
from abmem.services.market.demand import DemandSeries
from abmem.services.simulation import simulation_service
from tests.factories import make_resource

pytestmark = pytest.mark.django_db

SMALL_DDPG: dict[str, Any] = {
    "batch_size": 8,
    "learning_starts": 8,
    "buffer_size": 512,
    "hidden_sizes": [16],
}


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


def make_scenario(algorithms: tuple[str, ...] = ("DDPG", "FIXED_MARKUP", "DDPG")) -> Simulation:
    """Proxy scenario with one agent per entry of ``algorithms``."""
    sim = Simulation.objects.create(
        name="scn",
        mode=SimulationMode.ONLYRESULT,
        periodType=PeriodType.HOUR,
        periodNumber=6,
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
    for i, algorithm in enumerate(algorithms):
        res = make_resource(fuel_cost=Decimal(10 + 5 * i), om_cost=Decimal("1"))
        agent = Agent.objects.create(
            name=f"a{i}",
            market=market,
            proxy=True,
            type=AgentType.HYBRID,
            algorithm_type=algorithm,
            algorithm_params=dict(SMALL_DDPG) if algorithm == "DDPG" else {"markup": 0.3},
            failstack_enabled=False,
            failstack_limit=4,
        )
        portfolio = Portfolio.objects.create(agent=agent, proxy=True)
        Plant.objects.create(
            portfolio=portfolio, resource=res, capacity=700, proxy=True
        )
    return sim


def study_input(scenario: Simulation, **changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "matching",
        "scenario": scenario.pk,
        "algorithm_type": "DDPG",
        "target": "MATCHING",
        "param_bounds": {"gamma": {"low": 0.8, "high": 0.99, "scale": "linear"}},
        "fixed_params": {"failstack_enabled": True, "failstack_limit": 2},
        "colony_size": 2,
        "max_iter": 1,
        "limit": 100,
        "search_seeds": [11],
        "evaluation_seeds": [22],
        "objective": {"mode": "weighted", "weights": [1.0, 0.5]},
    }
    data.update(changes)
    return data


# ---------------------------------------------------------------------- model
def test_study_target_defaults_to_all() -> None:
    scenario = make_scenario()
    data = hpo_service.validate_study_input(study_input(scenario))
    data.pop("target")
    study = HPOStudy.objects.create(**data)
    study.refresh_from_db()
    assert study.target == HPOTarget.ALL == "ALL"
    assert study.get_target_display() == "All agents"
    assert HPOTarget.MATCHING.label == "Only matching agents"


# ----------------------------------------------------------------- validation
def test_validate_target_default_and_case() -> None:
    scenario = make_scenario()
    data = study_input(scenario)
    del data["target"]
    assert hpo_service.validate_study_input(data)["target"] == "ALL"
    assert (
        hpo_service.validate_study_input(study_input(scenario, target="matching"))[
            "target"
        ]
        == "MATCHING"
    )


@pytest.mark.parametrize(
    ("changes", "field", "text"),
    [
        ({"target": "SOME"}, "target", "unknown target"),
        ({"algorithm_type": "TD3"}, "algorithm_type", "No agent of the scenario uses TD3"),
        (
            {
                "algorithm_type": "FIXED_MARKUP",
                "param_bounds": {"markup": {"low": 0.0, "high": 0.5}},
            },
            "algorithm_type",
            "learning algorithm",
        ),
    ],
)
def test_validate_matching_rejects(changes: dict[str, Any], field: str, text: str) -> None:
    scenario = make_scenario()
    with pytest.raises(ValidationError) as excinfo:
        hpo_service.validate_study_input(study_input(scenario, **changes))
    assert text in " ".join(excinfo.value.message_dict[field])


def test_validate_all_accepts_algorithm_absent_from_scenario() -> None:
    scenario = make_scenario()
    data = hpo_service.validate_study_input(
        study_input(scenario, target="ALL", algorithm_type="TD3")
    )
    assert data["target"] == "ALL" and data["algorithm_type"] == "TD3"


def test_scenario_algorithm_counts() -> None:
    scenario = make_scenario(("DDPG", "FIXED_MARKUP", "DDPG", "TD3", "RANDOM"))
    counts = hpo_service.scenario_algorithm_counts(scenario)
    assert counts == {"DDPG": 2, "TD3": 1}
    empty = Simulation.objects.create(
        name="empty",
        mode=SimulationMode.ONLYRESULT,
        periodType=PeriodType.HOUR,
        periodNumber=3,
        currentPeriod=-1,
        day=0,
        proxy=True,
    )
    assert hpo_service.scenario_algorithm_counts(empty) == {}


# --------------------------------------------------------------------- worker
def test_target_algorithms() -> None:
    assert target_algorithms("ALL", "DDPG") == ("DDPG", None)
    assert target_algorithms(HPOTarget.MATCHING, "DDPG") == (None, "DDPG")
    with pytest.raises(ValueError):
        target_algorithms("OTHER", "DDPG")


def _pending_trial(study: HPOStudy, params: dict[str, Any]) -> HPOTrial:
    return HPOTrial.objects.create(
        study=study,
        iteration=0,
        phase=HPOPhase.INIT,
        source_index=0,
        params=params,
        seed=11,
    )


def test_matching_trial_tunes_only_matching_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_scenario()
    study = hpo_service.create_study(study_input(scenario))
    trial = _pending_trial(study, {"gamma": 0.85})
    calls: list[dict[str, Any]] = []
    real_run = simulation_service.run

    def spy(simulation: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_run(simulation, **kwargs)

    monkeypatch.setattr(simulation_service, "run", spy)
    outcome = worker.evaluate_task(
        EvaluationTask(study_id=study.pk, trial_id=trial.pk, seed=11, params={"gamma": 0.85})
    )
    assert outcome.ok, outcome.error
    assert calls[0]["forced_algorithm"] is None
    assert calls[0]["override_algorithm"] == "DDPG"

    trial.refresh_from_db()
    assert trial.status == HPOTrialStatus.FINISHED
    run = trial.simulation
    a0, a1, a2 = run.market.agent_set.order_by("id")
    for tuned in (a0, a2):
        assert tuned.algorithm_type == "DDPG"
        assert tuned.algorithm_params == {**SMALL_DDPG, "gamma": 0.85}
        assert tuned.effective_params["gamma"] == 0.85
        assert tuned.failstack_enabled is True and tuned.failstack_limit == 2
    assert a1.algorithm_type == "FIXED_MARKUP"
    assert a1.algorithm_params == {"markup": 0.3}
    assert a1.effective_params["markup"] == 0.3
    assert a1.failstack_enabled is False and a1.failstack_limit == 4

    filtered = score_simulation(run, agent_ids=[a0.pk, a2.pk])
    whole = score_simulation(run)
    assert trial.agent_score == pytest.approx(filtered.agent_score)
    assert trial.market_score == pytest.approx(whole.market_score)


def test_all_trial_still_forces_the_study_algorithm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = make_scenario()
    study = hpo_service.create_study(study_input(scenario, target="ALL"))
    trial = _pending_trial(study, {"gamma": 0.85})
    calls: list[dict[str, Any]] = []
    real_run = simulation_service.run

    def spy(simulation: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_run(simulation, **kwargs)

    monkeypatch.setattr(simulation_service, "run", spy)
    outcome = worker.evaluate_task(
        EvaluationTask(study_id=study.pk, trial_id=trial.pk, seed=11, params={"gamma": 0.85})
    )
    assert outcome.ok, outcome.error
    assert calls[0]["forced_algorithm"] == "DDPG"
    assert calls[0]["override_algorithm"] is None
    trial.refresh_from_db()
    agents = list(trial.simulation.market.agent_set.order_by("id"))
    assert {a.algorithm_type for a in agents} == {"DDPG"}
    # The former FIXED_MARKUP agent gets exactly the overrides (foreign agent).
    assert agents[1].algorithm_params == {"gamma": 0.85}
    assert trial.agent_score == pytest.approx(
        score_simulation(trial.simulation).agent_score
    )


def test_matching_study_runs_end_to_end() -> None:
    scenario = make_scenario(("DDPG", "FIXED_MARKUP"))
    study = hpo_service.create_study(study_input(scenario))
    hpo_service.run_study(study.pk, task_runner=hpo_service.serial_runner)
    study.refresh_from_db()
    assert study.status == HPOStatus.FINISHED, study.error
    trials = list(HPOTrial.objects.filter(study=study).order_by("id"))
    assert trials[-1].phase == HPOPhase.FINAL_EVAL and trials[-1].seed == 22
    for trial in trials:
        a0, a1 = trial.simulation.market.agent_set.order_by("id")
        assert a0.algorithm_type == "DDPG"
        assert a0.algorithm_params == {**SMALL_DDPG, **trial.params}
        assert (a1.algorithm_type, a1.algorithm_params) == (
            "FIXED_MARKUP",
            {"markup": 0.3},
        )
