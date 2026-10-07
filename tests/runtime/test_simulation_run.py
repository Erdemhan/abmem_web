"""Tests of ``simulation_service.run`` and ``SimulationContext.build`` (6.3, 5.4)."""

from __future__ import annotations

import ast
import json
import pathlib
from decimal import Decimal

import pytest

from abmem.models import AgentStepLog, AlgorithmDefault
from abmem.models.enums import MarketState, SimulationState
from abmem.models.models import Offer, Period
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
)
from tests.runtime.conftest import SMALL_DDPG, constant_demand, make_run, resources

pytestmark = pytest.mark.django_db

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_MODULES = [
    "abmem/services/agent/agent_service.py",
    "abmem/services/market/market_service.py",
    "abmem/services/simulation/simulation_service.py",
]


def _per_agent_rows(simulation) -> dict[int, int]:
    rows: dict[int, int] = {}
    for agent in simulation.market.agent_set.all():
        rows[agent.id] = AgentStepLog.objects.filter(agent=agent).count()
    return rows


def test_baselines_reach_finished(fast_demand) -> None:
    r = resources(3)
    simulation = make_run(
        [
            {"plants": [(r[0], 400)], "algorithm": "RANDOM"},
            {"plants": [(r[1], 300)], "algorithm": "MARGINAL_COST"},
            {
                "plants": [(r[2], 300)],
                "algorithm": "FIXED_MARKUP",
                "failstack": {"failstack_enabled": True, "failstack_limit": 1},
            },
        ],
        period_number=5,
    )
    result = simulation_service.run(simulation.id, export_json=False)
    assert result.state == SimulationState.FINISHED
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    assert simulation.currentPeriod == 5
    assert simulation.market.state == MarketState.PERIODEND
    assert Period.objects.filter(market=simulation.market).count() == 5
    assert Offer.objects.filter(period__market=simulation.market).count() == 15
    assert set(_per_agent_rows(simulation).values()) == {4}
    assert not AgentStepLog.objects.filter(learning_active=True).exists()
    # the fail-stack rule is never applied to a baseline (2.1)
    for row in AgentStepLog.objects.all():
        assert row.intervened_plants == [False]
    for period in Period.objects.filter(market=simulation.market):
        assert (
            period.ptf is not None and period.ptf > simulation.market.lowerBidBound - 1
        )


def test_ddpg_run_logs_t_minus_one_rows_and_learning_start(fast_demand) -> None:
    r = resources(2)
    simulation = make_run(
        [{"plants": [(r[0], 500), (r[1], 400)], "algorithm": "DDPG"}],
        period_number=200,
    )
    simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    agent = simulation.market.agent_set.get()
    rows = list(
        AgentStepLog.objects.filter(agent=agent)
        .select_related("period")
        .order_by("period__periodNumber")
    )
    assert len(rows) == 199
    active = [row.period.periodNumber for row in rows if row.learning_active]
    # lags {1, 24}: first transition at step 26, learning_starts 64 -> step 89
    assert active[0] == 89
    assert active == list(range(89, 201))
    assert rows[-1].critic_loss is not None
    assert rows[0].critic_loss is None
    assert all(row.exploration_scale is not None for row in rows)


def test_single_period_run_has_no_log_rows(fast_demand) -> None:
    r = resources(1)
    simulation = make_run(
        [{"plants": [(r[0], 400)], "algorithm": "DDPG"}], period_number=1
    )
    simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    assert Period.objects.filter(market=simulation.market).count() == 1
    assert AgentStepLog.objects.count() == 0


def test_short_learning_run_stores_no_transition(fast_demand) -> None:
    r = resources(2)
    simulation = make_run(
        [{"plants": [(r[0], 400)], "algorithm": "DDPG"}],
        period_number=20,
    )
    ctx_holder = {}
    original = SimulationContext.build

    def _capture(*args, **kwargs):
        ctx = original(*args, **kwargs)
        ctx_holder["ctx"] = ctx
        return ctx

    try:
        SimulationContext.build = _capture  # type: ignore[method-assign]
        simulation_service.run(simulation, export_json=False)
    finally:
        SimulationContext.build = original  # type: ignore[method-assign]
    assert len(ctx_holder["ctx"].agents[0].policy.buffer) == 0
    assert not AgentStepLog.objects.filter(learning_active=True).exists()


def test_one_and_five_plant_agents_with_evaluation_window(fast_demand) -> None:
    r = resources(6)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 300)],
                "algorithm": "DDPG",
                "params": SMALL_DDPG,
            },
            {
                "plants": [(r[i], 60 + i) for i in range(1, 6)],
                "algorithm": "TD3",
                "params": SMALL_DDPG,
                "failstack": {"failstack_enabled": True, "failstack_limit": 2},
            },
        ],
        period_number=60,
        evaluation_periods=10,
    )
    simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    rows = AgentStepLog.objects.select_related("period")
    assert rows.count() == 2 * 59
    evaluation = rows.filter(evaluation_phase=True)
    assert evaluation.count() == 2 * 10
    assert {row.period.periodNumber for row in evaluation} == set(range(51, 61))
    assert not evaluation.filter(learning_active=True).exists()
    assert rows.filter(learning_active=True).exists()
    five = simulation.market.agent_set.order_by("id")[1]
    assert all(len(row.intervened_plants) == 5 for row in rows.filter(agent=five))
    assert Offer.objects.filter(period__market=simulation.market).count() == 60 * 6


def test_invalid_bounds_fail_before_period_one(fast_demand) -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}], lower=100, upper=100)
    with pytest.raises(SimulationValidationError, match="upperBidBound"):
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert "upperBidBound" in simulation.error
    assert not Period.objects.filter(market=simulation.market).exists()


def test_short_demand_horizon_fails_before_period_one(monkeypatch) -> None:
    from abmem.services.market.demand import DemandSeries

    series = constant_demand(n=3)
    monkeypatch.setattr(
        DemandSeries, "from_excel", staticmethod(lambda *a, **k: series)
    )
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}], period_number=5)
    with pytest.raises(SimulationValidationError):
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert simulation.error
    assert not Period.objects.filter(market=simulation.market).exists()


@pytest.mark.parametrize(
    "agents",
    [
        [],
        [{"plants": []}],
        [{"plants": "zero"}],
        [{"plants": "unknown-param"}],
    ],
)
def test_build_rejects_invalid_scenarios(agents, fast_demand) -> None:
    r = resources(1)
    specs = []
    for spec in agents:
        if spec["plants"] == "zero":
            specs.append({"plants": [(r[0], 0)]})
        elif spec["plants"] == "unknown-param":
            specs.append(
                {
                    "plants": [(r[0], 10)],
                    "algorithm": "DDPG",
                    "params": {"NOISE_STD": 0.1},
                }
            )
        else:
            specs.append(spec)
    simulation = make_run(specs)
    with pytest.raises(SimulationValidationError):
        SimulationContext.build(simulation)


def test_rerun_of_finished_simulation_raises_value_error(fast_demand) -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}], period_number=2)
    simulation_service.run(simulation, export_json=False)
    with pytest.raises(ValueError, match="CREATED"):
        simulation_service.run(simulation.id, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED


def test_exception_in_policy_update_sets_failed_and_rolls_back(
    fast_demand, monkeypatch
) -> None:
    from abmem.services.algorithms.DDPG import DDPGAlgorithm

    calls = {"n": 0}

    def _boom(self):
        calls["n"] += 1
        raise RuntimeError("update exploded")

    monkeypatch.setattr(DDPGAlgorithm, "update", _boom)
    r = resources(1)
    simulation = make_run(
        [{"plants": [(r[0], 400)], "algorithm": "DDPG"}], period_number=40
    )
    with pytest.raises(RuntimeError, match="update exploded"):
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert "update exploded" in simulation.error
    # first transition (and update) at step 26: periods 1..25 committed
    assert calls["n"] == 1
    assert Period.objects.filter(market=simulation.market).count() == 25


def test_parameter_resolution_oq9(fast_demand) -> None:
    AlgorithmDefault.objects.create(
        algorithm_type="DDPG", params={"batch_size": 16, "gamma": 0.5}
    )
    r = resources(1)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 100)],
                "algorithm": "DDPG",
                "params": {"gamma": 0.8, "learning_starts": 20},
            }
        ]
    )
    ctx = SimulationContext.build(simulation)
    config = ctx.agents[0].policy.config
    assert (config.batch_size, config.gamma, config.learning_starts) == (16, 0.8, 20)

    ctx = SimulationContext.build(simulation, {"gamma": 0.9})
    config = ctx.agents[0].policy.config
    assert (config.batch_size, config.gamma, config.learning_starts) == (16, 0.9, 20)


def test_forced_algorithm_ignores_agent_params(fast_demand) -> None:
    r = resources(1)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 100)],
                "algorithm": "DDPG",
                "params": {"gamma": 0.8},
                "failstack": {"failstack_enabled": True},
            }
        ]
    )
    ctx = SimulationContext.build(simulation, forced_algorithm="RANDOM")
    assert ctx.agents[0].policy.key == "RANDOM"
    assert ctx.agents[0].failstack_params is None
    ctx = SimulationContext.build(
        simulation, {"markup": 0.2}, forced_algorithm="FIXED_MARKUP"
    )
    assert ctx.agents[0].policy.config.markup == 0.2


def test_json_export_written_when_enabled(fast_demand, settings, tmp_path) -> None:
    settings.ABMEM_SIM_DATA_DIR = str(tmp_path)
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}], period_number=3)
    simulation_service.run(simulation)
    data = json.loads(
        (tmp_path / f"simulation_{simulation.id}_offers.json").read_text("utf-8")
    )
    assert data["total_periods"] == 3
    assert sorted(data["offers_by_period"]) == ["1", "2", "3"]
    last = data["offers_by_period"]["3"]
    agent = simulation.market.agent_set.get()
    assert Decimal(last["a0"][0]["budget"]) == agent.budget
    assert (
        Decimal(last["market_price"])
        == Period.objects.get(market=simulation.market, periodNumber=3).ptf
    )


def test_json_export_skipped_when_disabled(fast_demand, settings, tmp_path) -> None:
    settings.ABMEM_SIM_DATA_DIR = str(tmp_path)
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}], period_number=2)
    simulation_service.run(simulation, export_json=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_no_module_level_mutable_state_b10(module: str) -> None:
    tree = ast.parse((REPO_ROOT / module).read_text(encoding="utf-8"))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
            assert not isinstance(
                stmt.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp)
            ), f"module-level mutable container at line {stmt.lineno}"
