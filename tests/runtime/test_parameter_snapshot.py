"""Effective parameter resolution and the per-run parameter snapshot (PLAN 2.12)."""

from __future__ import annotations

import json

import pytest

from abmem.models import Agent, AlgorithmDefault, ClearingDefault, Market
from abmem.services.agent import library_service
from abmem.services.algorithms.algorithm_config import FixedMarkupConfig, RandomConfig
from abmem.services.analysis import exports
from abmem.services.analysis.run_analysis import build_run_analysis
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
    clearing_defaults,
    market_clearing_params,
)
from tests.plugin_helpers import (
    ParamClearingConfig,
    temporary_param_clearing_algorithm,
)
from tests.runtime.conftest import constant_demand, make_run, resources

pytestmark = pytest.mark.django_db

PARAM = "TEST_PARAM_CLEARING"


def _build(simulation: object) -> SimulationContext:
    return SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False  # type: ignore[arg-type]
    )


# Agents ------------------------------------------------------------------------


def test_agent_resolution_order_class_default_agent() -> None:
    r = resources(3)
    AlgorithmDefault.objects.create(
        algorithm_type="FIXED_MARKUP", params={"markup": 0.3, "gamma": 0.9}
    )
    simulation = make_run(
        [
            {"plants": [(r[0], 100)], "algorithm": "FIXED_MARKUP"},
            {
                "plants": [(r[1], 100)],
                "algorithm": "FIXED_MARKUP",
                "params": {"markup": 0.05},
            },
            {"plants": [(r[2], 100)], "algorithm": "RANDOM"},
        ]
    )
    ctx = _build(simulation)
    first, second, third = (state.policy.config for state in ctx.agents)
    assert first == FixedMarkupConfig(markup=0.3, gamma=0.9)
    assert second == FixedMarkupConfig(markup=0.05, gamma=0.9)
    assert third == RandomConfig()


def test_snapshot_is_written_at_build_and_kept_after_a_default_edit() -> None:
    r = resources(2)
    AlgorithmDefault.objects.create(
        algorithm_type="FIXED_MARKUP", params={"markup": 0.2}
    )
    simulation = make_run(
        [
            {"plants": [(r[0], 100)], "algorithm": "FIXED_MARKUP"},
            {"plants": [(r[1], 100)], "algorithm": "RANDOM"},
        ]
    )
    agents = list(simulation.market.agent_set.order_by("id"))
    assert all(agent.effective_params is None for agent in agents)
    assert simulation.market.effective_clearing_params is None

    _build(simulation)

    stored = {
        a.name: a.effective_params
        for a in Agent.objects.filter(market=simulation.market)
    }
    assert stored["a0"] == FixedMarkupConfig(markup=0.2).to_dict()
    assert stored["a1"] == RandomConfig().to_dict()
    json.dumps(stored)  # JSON-serialisable
    assert Market.objects.get(pk=simulation.market.pk).effective_clearing_params == {}

    AlgorithmDefault.objects.filter(algorithm_type="FIXED_MARKUP").update(
        params={"markup": 0.4}
    )
    again = Agent.objects.get(market=simulation.market, name="a0")
    assert again.effective_params["markup"] == 0.2


def test_forced_algorithm_snapshot_records_the_forced_config() -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)], "algorithm": "FIXED_MARKUP"}])
    SimulationContext.build(
        simulation,
        {"markup": 0.25},
        forced_algorithm="FIXED_MARKUP",
        demand=constant_demand(),
        export_json=False,
    )
    agent = Agent.objects.get(market=simulation.market)
    assert agent.effective_params["markup"] == 0.25


# Clearing ----------------------------------------------------------------------


def test_clearing_resolution_order_and_market_params() -> None:
    market = Market(clearing_algorithm=PARAM, clearing_params={"rounds": 4})
    assert market_clearing_params(market, {}) == {"rounds": 4}
    defaults = {PARAM: {"markup": 1.0, "rounds": 2}, "OTHER": {"x": 1}}
    assert market_clearing_params(market, defaults) == {"markup": 1.0, "rounds": 4}
    ClearingDefault.objects.create(algorithm_key=PARAM, params={"strict": True})
    assert clearing_defaults() == {PARAM: {"strict": True}}


def test_param_clearing_algorithm_receives_its_config_in_a_run(
    fast_demand: object,
) -> None:
    r = resources(2)
    with temporary_param_clearing_algorithm(PARAM):
        ClearingDefault.objects.create(
            algorithm_key=PARAM, params={"markup": 1.0, "strict": True}
        )
        simulation = make_run(
            [
                {"plants": [(r[0], 300)], "algorithm": "MARGINAL_COST"},
                {"plants": [(r[1], 300)], "algorithm": "MARGINAL_COST"},
            ],
            period_number=2,
        )
        market = simulation.market
        market.clearing_algorithm = PARAM
        market.clearing_params = {"rounds": 12, "weights": [5]}
        market.save()
        ctx = _build(simulation)
        expected = ParamClearingConfig(markup=1.0, strict=True, rounds=12, weights=(5,))
        assert ctx.clearing.config == expected
        assert Market.objects.get(pk=market.pk).effective_clearing_params == (
            expected.to_dict()
        )

        # A full run reaches FINISHED with the parameterised clearing algorithm.
        run = make_run(
            [{"plants": [(r[0], 300)], "algorithm": "MARGINAL_COST"}], period_number=2
        )
        run.market.clearing_algorithm = PARAM
        run.market.save()
        simulation_service.run(run, export_json=False)
        run.market.refresh_from_db()
        assert (
            run.market.effective_clearing_params
            == ParamClearingConfig(markup=1.0, strict=True).to_dict()
        )


def test_invalid_clearing_params_stop_the_run_at_validation() -> None:
    r = resources(1)
    with temporary_param_clearing_algorithm(PARAM):
        simulation = make_run([{"plants": [(r[0], 100)]}])
        simulation.market.clearing_algorithm = PARAM
        simulation.market.clearing_params = {"rounds": 0}
        simulation.market.save()
        with pytest.raises(SimulationValidationError, match="rounds must be >= 1"):
            _build(simulation)
        simulation.market.clearing_params = {"unknown": 1}
        simulation.market.save()
        with pytest.raises(SimulationValidationError, match="unknown parameter"):
            _build(simulation)
    assert Agent.objects.get(market=simulation.market).effective_params is None


# Copies, analysis and export ---------------------------------------------------


def test_copies_do_not_carry_a_snapshot_but_keep_clearing_params() -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)], "algorithm": "RANDOM"}])
    simulation.proxy = True
    simulation.save()
    market = simulation.market
    market.proxy = True
    market.clearing_params = {"anything": 1}
    market.effective_clearing_params = {"stale": True}
    market.save()
    Agent.objects.filter(market=market).update(effective_params={"stale": True})

    for copy in (
        library_service.duplicate_scenario(simulation),
        library_service.clone_scenario_for_run(simulation),
    ):
        assert copy.market.clearing_params == {"anything": 1}
        assert copy.market.effective_clearing_params is None
        assert all(
            agent.effective_params is None for agent in copy.market.agent_set.all()
        )


def test_analysis_and_export_show_the_snapshot(fast_demand: object) -> None:
    r = resources(2)
    AlgorithmDefault.objects.create(
        algorithm_type="FIXED_MARKUP", params={"markup": 0.2}
    )
    simulation = make_run(
        [
            {"plants": [(r[0], 300)], "algorithm": "FIXED_MARKUP"},
            {"plants": [(r[1], 300)], "algorithm": "MARGINAL_COST"},
        ],
        period_number=2,
    )
    simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()

    analysis = build_run_analysis(simulation)
    first = analysis.settings.agents[0]
    assert first.effective_params == FixedMarkupConfig(markup=0.2).to_dict()
    assert analysis.settings.effective_clearing_params == {}
    data = analysis.to_dict()
    assert data["settings"]["agents"][0]["effective_params"]["markup"] == 0.2

    document = json.loads(exports.parameters_json(simulation))
    assert document["simulation_id"] == simulation.pk
    assert document["clearing_algorithm"] == "MERIT_ORDER"
    assert document["clearing_params"] == {}
    assert [a["name"] for a in document["agents"]] == ["a0", "a1"]
    assert document["agents"][0]["algorithm"] == "FIXED_MARKUP"
    assert document["agents"][0]["params"]["markup"] == 0.2


def test_analysis_of_a_run_without_snapshot(fast_demand: object) -> None:
    r = resources(1)
    simulation = make_run(
        [{"plants": [(r[0], 300)], "algorithm": "RANDOM"}], period_number=2
    )
    simulation_service.run(simulation, export_json=False)
    Agent.objects.filter(market=simulation.market).update(effective_params=None)
    Market.objects.filter(pk=simulation.market.pk).update(
        effective_clearing_params=None
    )
    simulation.refresh_from_db()
    analysis = build_run_analysis(simulation)
    assert analysis.settings.agents[0].effective_params is None
    assert analysis.settings.effective_clearing_params is None
    document = json.loads(exports.parameters_json(simulation))
    assert document["clearing_params"] is None
    assert document["agents"][0]["params"] is None
