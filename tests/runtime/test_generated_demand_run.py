"""Runs of a scenario with a synthetic demand profile (PLAN 2.8)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from abmem.models import Simulation
from abmem.models.enums import AlgorithmType, SimulationState
from abmem.models.models import Period
from abmem.services.agent import library_service
from abmem.services.agent.observation import (
    DEMAND_SCALING_MEAN,
    DEMAND_SCALING_STD,
)
from abmem.services.market.demand import DemandSeries
from abmem.services.market.demand_generator import DemandProfile, generate_demand
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
)
from tests.runtime.conftest import make_run, resources

pytestmark = pytest.mark.django_db

PROFILE: dict = {
    "version": 1,
    "seed": 99,
    "day_start_hour": 7,
    "day_end_hour": 22,
    "autocorrelation": 0.85,
    "day": {
        "distribution": "NORMAL",
        "mean": 700.0,
        "std": 80.0,
        "min": 500.0,
        "max": 900.0,
    },
    "night": {
        "distribution": "LOGNORMAL",
        "mean": 450.0,
        "std": 60.0,
        "min": 300.0,
        "max": 600.0,
    },
}


def _scenario(period_number: int, profile: dict | None = PROFILE) -> Simulation:
    r = resources(2)
    simulation = make_run(
        [
            {"plants": [(r[0], 600)], "algorithm": AlgorithmType.RANDOM},
            {"plants": [(r[1], 500)], "algorithm": AlgorithmType.MARGINAL_COST},
        ],
        period_number=period_number,
    )
    simulation.proxy = True
    simulation.demand_profile = profile
    simulation.save()
    market = simulation.market
    market.proxy = True
    market.save()
    return simulation


def _demands(run: Simulation) -> list[Decimal]:
    return list(
        Period.objects.filter(market=run.market)
        .order_by("periodNumber")
        .values_list("demand", flat=True)
    )


def _expected(periods: int) -> list[Decimal]:
    values = generate_demand(DemandProfile.from_dict(PROFILE), periods)
    return [Decimal(repr(float(v))).quantize(Decimal("0.001")) for v in values]


def _fail_if_excel_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def _from_excel(*_args, **_kwargs):
        raise AssertionError("the demand file must not be read for a generated profile")

    monkeypatch.setattr(DemandSeries, "from_excel", staticmethod(_from_excel))


def test_run_uses_the_generated_series_and_ignores_the_run_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_excel_is_read(monkeypatch)
    scenario = _scenario(30)
    runs = [
        library_service.clone_scenario_for_run(scenario, seed=seed) for seed in (1, 2)
    ]
    for run in runs:
        assert run.demand_profile == PROFILE
        result = simulation_service.run(run.id, export_json=False)
        assert result.state == SimulationState.FINISHED
    assert _demands(runs[0]) == _expected(30)
    assert _demands(runs[1]) == _demands(runs[0])


def test_short_and_long_runs_share_the_same_first_periods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_excel_is_read(monkeypatch)
    scenario = _scenario(168)
    short = library_service.clone_scenario_for_run(scenario)
    scenario.periodNumber = 336
    scenario.save()
    long = library_service.clone_scenario_for_run(scenario)
    for run in (short, long):
        assert (
            simulation_service.run(run.id, export_json=False).state
            == SimulationState.FINISHED
        )
    short_demands = _demands(short)
    long_demands = _demands(long)
    assert len(short_demands) == 168 and len(long_demands) == 336
    assert long_demands[:168] == short_demands
    assert long_demands == _expected(336)


def test_build_uses_from_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_if_excel_is_read(monkeypatch)
    run = library_service.clone_scenario_for_run(_scenario(48))
    ctx = SimulationContext.build(run, export_json=False)
    assert ctx.demand.source == "generated"
    assert (
        ctx.demand.values.tobytes()
        == generate_demand(DemandProfile.from_dict(PROFILE), 48).tobytes()
    )


def test_build_scales_demand_with_the_profile_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_excel_is_read(monkeypatch)
    run = library_service.clone_scenario_for_run(_scenario(48))
    ctx = SimulationContext.build(run, export_json=False)
    mean, std = DemandProfile.from_dict(PROFILE).scaling()
    assert ctx.normalizer.mean[0] == mean
    assert ctx.normalizer.scale[0] == std


def test_invalid_stored_profile_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_if_excel_is_read(monkeypatch)
    broken = dict(PROFILE, autocorrelation=1.5)
    run = library_service.clone_scenario_for_run(_scenario(10, broken))
    with pytest.raises(SimulationValidationError, match="invalid demand profile"):
        SimulationContext.build(run, export_json=False)


def test_null_profile_reads_the_file(fast_demand: DemandSeries) -> None:
    run = library_service.clone_scenario_for_run(_scenario(10, None))
    assert run.demand_profile is None
    ctx = SimulationContext.build(run, export_json=False)
    assert ctx.demand is fast_demand
    assert ctx.normalizer.mean[0] == DEMAND_SCALING_MEAN
    assert ctx.normalizer.scale[0] == DEMAND_SCALING_STD


def test_duplicate_copies_the_profile() -> None:
    copy = library_service.duplicate_scenario(_scenario(10))
    assert copy.proxy
    assert copy.demand_profile == PROFILE
