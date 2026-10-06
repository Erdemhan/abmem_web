"""HPOStudy, HPOTrial and AlgorithmDefault (data_model spec, 0011/0012)."""

from __future__ import annotations

from typing import Any

import pytest
from django.db import IntegrityError, transaction
from django.db.models import ProtectedError

from abmem.models import AlgorithmDefault, HPOStudy, HPOTrial, Simulation
from abmem.models.enums import AlgorithmType, HPOPhase, HPOStatus, HPOTrialStatus
from tests.factories import make_simulation

pytestmark = pytest.mark.django_db


def _scenario() -> Simulation:
    simulation = make_simulation()
    simulation.proxy = True
    simulation.save()
    return simulation


def _study(scenario: Simulation, **fields: Any) -> HPOStudy:
    values: dict[str, Any] = {
        "name": "study",
        "algorithm_type": AlgorithmType.TD3,
        "scenario": scenario,
        "param_bounds": {"lr": {"low": 1e-4, "high": 1e-2, "scale": "log"}},
        "colony_size": 4,
        "max_iter": 2,
        "limit": 3,
        "search_seeds": [1, 2],
        "evaluation_seeds": [3],
        "objective": {"agent": 1.0, "market": 0.0},
    }
    values.update(fields)
    return HPOStudy.objects.create(**values)


def _trial(study: HPOStudy, **fields: Any) -> HPOTrial:
    values: dict[str, Any] = {
        "study": study,
        "iteration": 0,
        "phase": HPOPhase.INIT,
        "source_index": 0,
        "params": {"lr": 1e-3},
        "seed": 1,
    }
    values.update(fields)
    return HPOTrial.objects.create(**values)


def test_study_defaults() -> None:
    study = _study(_scenario())
    study.refresh_from_db()
    assert study.status == HPOStatus.CREATED
    assert study.fixed_params == {}
    assert study.error == ""
    assert study.cancel_requested is False
    assert study.heartbeat_at is None
    assert study.best_params is None and study.best_score is None


def test_trial_defaults() -> None:
    trial = _trial(_study(_scenario()))
    trial.refresh_from_db()
    assert trial.status == HPOTrialStatus.PENDING
    assert trial.simulation is None
    assert trial.error == ""


def test_scenario_delete_is_protected() -> None:
    scenario = _scenario()
    _study(scenario)
    with pytest.raises(ProtectedError):
        scenario.delete()


def test_study_delete_cascades_trials() -> None:
    study = _study(_scenario())
    _trial(study)
    _trial(study, seed=2)
    study.delete()
    assert HPOTrial.objects.count() == 0


def test_simulation_delete_sets_trial_simulation_null() -> None:
    study = _study(_scenario())
    run = make_simulation()
    trial = _trial(study, simulation=run)
    assert Simulation.objects.filter(proxy=False, hpo_trial__isnull=True).count() == 0
    run.delete()
    trial.refresh_from_db()
    assert trial.simulation is None


@pytest.mark.parametrize(
    "fields",
    [{"colony_size": 1}, {"max_iter": 0}, {"limit": 0}],
)
def test_study_check_constraints(fields: dict[str, Any]) -> None:
    scenario = _scenario()
    with pytest.raises(IntegrityError), transaction.atomic():
        _study(scenario, **fields)


def test_final_eval_phase_and_failed_trial_status() -> None:
    trial = _trial(
        _study(_scenario()), phase=HPOPhase.FINAL_EVAL, status=HPOTrialStatus.FAILED
    )
    trial.refresh_from_db()
    assert trial.phase == HPOPhase.FINAL_EVAL
    assert trial.status == HPOTrialStatus.FAILED


def test_algorithm_default_unique_per_type() -> None:
    default = AlgorithmDefault.objects.create(algorithm_type=AlgorithmType.SAC)
    default.refresh_from_db()
    assert default.params == {}
    assert default.updated_at is not None
    with pytest.raises(IntegrityError), transaction.atomic():
        AlgorithmDefault.objects.create(
            algorithm_type=AlgorithmType.SAC, params={"lr": 1}
        )
