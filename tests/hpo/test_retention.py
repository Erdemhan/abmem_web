"""Retention of HPO evaluation data (K9)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from abmem.models import HPOStudy, HPOTrial, Simulation
from abmem.models.enums import AgentType, AlgorithmType, HPOPhase, HPOTrialStatus
from abmem.models.models import Offer, Period
from abmem.services.hpo.retention import apply_retention, purge_simulation_details
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offer,
    make_period,
    make_resource,
    make_simulation,
)

pytestmark = pytest.mark.django_db


def _sim_with_data() -> Simulation:
    sim = make_simulation(period_number=1)
    market = make_market(simulation=sim, upper_bid_bound=100)
    res = make_resource()
    agent = make_agent_with_plants(market, [(res, 10)], agent_type=AgentType.HYBRID)
    period = make_period(market, 1, ptf=Decimal("10"))
    make_offer(
        period,
        agent,
        res,
        10,
        Decimal("10"),
        acceptance=True,
        acceptance_price=Decimal("10"),
        acceptance_amount=Decimal("10"),
        save=True,
    )
    return sim


def _study() -> HPOStudy:
    scenario = make_simulation()
    scenario.proxy = True
    scenario.save()
    return HPOStudy.objects.create(
        name="s",
        algorithm_type=AlgorithmType.FIXED_MARKUP,
        scenario=scenario,
        param_bounds={"markup": {"low": 0.0, "high": 0.5, "scale": "linear"}},
        colony_size=2,
        max_iter=1,
        limit=5,
        search_seeds=[1],
        evaluation_seeds=[2],
        objective={"mode": "weighted", "weights": [1.0, 0.0]},
    )


def _trial(
    study: HPOStudy,
    phase: HPOPhase,
    index: int,
    status: HPOTrialStatus = HPOTrialStatus.FINISHED,
) -> HPOTrial:
    return HPOTrial.objects.create(
        study=study,
        iteration=0,
        phase=phase,
        source_index=index,
        params={},
        seed=1,
        simulation=_sim_with_data(),
        status=status,
    )


def _has_periods(trial: HPOTrial) -> bool:
    return Period.objects.filter(market__simulation=trial.simulation).exists()


def test_purge_keeps_simulation_market_agents() -> None:
    sim = _sim_with_data()
    purge_simulation_details(sim)
    assert not Period.objects.filter(market__simulation=sim).exists()
    assert not Offer.objects.filter(period__market__simulation=sim).exists()
    sim.refresh_from_db()
    assert sim.market.agent_set.count() == 1


def test_only_best_candidate_keeps_periods_and_displaced_best_is_purged() -> None:
    study = _study()
    t0 = _trial(study, HPOPhase.INIT, 0)
    t1 = _trial(study, HPOPhase.INIT, 1)
    apply_retention(study, (0, HPOPhase.INIT.value, 0))
    assert _has_periods(t0) and not _has_periods(t1)

    t2 = _trial(study, HPOPhase.EMPLOYED, 1)
    apply_retention(study, (0, HPOPhase.EMPLOYED.value, 1))
    assert _has_periods(t2) and not _has_periods(t0)
    assert HPOTrial.objects.filter(study=study).count() == 3
    assert all(
        t.simulation_id is not None for t in HPOTrial.objects.filter(study=study)
    )


def test_non_finished_trials_are_not_purged_and_keep_ids_override_key() -> None:
    study = _study()
    running = _trial(study, HPOPhase.INIT, 0, HPOTrialStatus.RUNNING)
    a = _trial(study, HPOPhase.ONLOOKER, 1)
    b = _trial(study, HPOPhase.ONLOOKER, 1)
    apply_retention(study, (0, HPOPhase.ONLOOKER.value, 1), keep_trial_ids=[a.pk])
    assert _has_periods(running)
    assert _has_periods(a) and not _has_periods(b)


def test_no_best_purges_all_finished() -> None:
    study = _study()
    t0 = _trial(study, HPOPhase.INIT, 0)
    apply_retention(study, None)
    assert not _has_periods(t0)
