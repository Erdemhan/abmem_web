"""Tests of ``build_run_analysis`` (PLAN 2.5, spec run_analysis)."""

from __future__ import annotations

import json
import pathlib
from decimal import Decimal

import pandas as pd
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from abmem.models import Agent, Simulation
from abmem.models.enums import AlgorithmType, SimulationState
from abmem.models.models import Offer
from abmem.services.analysis.run_analysis import (
    RunAnalysisUnavailable,
    build_run_analysis,
    previous_budget,
    reconstruct_budget_series,
    unavailable_reason,
)
from abmem.services.hpo.objective import score_simulation
from abmem.services.simulation import simulation_service
from tests.analysis.legacy_reference import reference_scores
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offer,
    make_period,
    make_resource,
    make_simulation,
)
from tests.runtime.conftest import (  # noqa: F401  (pytest fixtures)
    SMALL_DDPG,
    _restore_torch_determinism,
    fast_demand,
    make_run,
    resources,
)

pytestmark = pytest.mark.django_db

WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE")


# Hand-built run ---------------------------------------------------------


@pytest.fixture
def hand_run() -> Simulation:
    """A FINISHED two-period run with hand-computable numbers.

    r1: static cost 1, variable cost 10. r2: no cost. r3: static cost 2.5.
    Agent A (r1 100 MWh, r2 50 MWh, initial budget 1000), agent B (r3 80 MWh,
    initial budget 500, never accepted). Offer profits:

    * period 1 (PTF 40): A/r1 100 @ 30 accepted 100 at 40 -> 2900, A/r2 50 @ 50
      rejected -> 0, B/r3 80 @ 45 rejected -> -200,
    * period 2 (PTF 55): A/r1 100 @ 55 accepted 60.5 at 55 -> 2622.5, A/r2
      50 @ 20 accepted 50 at 55 -> 2750, B/r3 80 @ 60 rejected -> -200.
    """
    simulation = make_simulation(name="hand", period_number=2, current_period=2)
    simulation.state = SimulationState.FINISHED
    simulation.seed = 123
    simulation.evaluation_periods = 1
    simulation.save()
    market = make_market(simulation, upper_bid_bound=200)
    r1 = make_resource(name="R1", om_cost=Decimal("1"), fuel_cost=Decimal("10"))
    r2 = make_resource(name="R2")
    r3 = make_resource(name="R3", om_cost=Decimal("2.5"))
    a = make_agent_with_plants(market, [(r1, 100), (r2, 50)], name="A")
    b = make_agent_with_plants(market, [(r3, 80)], name="B")
    a.algorithm_type = AlgorithmType.MARGINAL_COST
    b.algorithm_type = AlgorithmType.RANDOM
    a.budget = Decimal("9272.50")
    b.budget = Decimal("100.00")
    a.save()
    b.save()
    p1 = make_period(
        market, 1, demand=120, met_demand=100, market_volume=230, ptf=Decimal("40")
    )
    p2 = make_period(
        market, 2, demand=130, met_demand=110.5, market_volume=230, ptf=Decimal("55")
    )
    specs = [
        (p1, a, r1, 100, "30", True, "40", "100"),
        (p1, a, r2, 50, "50", False, None, "0"),
        (p1, b, r3, 80, "45", False, None, "0"),
        (p2, a, r1, 100, "55", True, "55", "60.5"),
        (p2, a, r2, 50, "20", True, "55", "50"),
        (p2, b, r3, 80, "60", False, None, "0"),
    ]
    for (
        period,
        agent,
        resource,
        amount,
        price,
        accepted,
        acc_price,
        acc_amount,
    ) in specs:
        make_offer(
            period,
            agent,
            resource,
            amount,
            Decimal(price),
            acceptance=accepted,
            acceptance_price=None if acc_price is None else Decimal(acc_price),
            acceptance_amount=Decimal(acc_amount),
            save=True,
        )
    return simulation


def test_hand_run_series_and_summary(hand_run: Simulation) -> None:
    analysis = build_run_analysis(hand_run)

    assert analysis.settings.seed == 123
    assert analysis.settings.evaluation_periods == 1
    assert analysis.settings.upper_bound == 200
    assert [a.name for a in analysis.settings.agents] == ["A", "B"]
    assert analysis.settings.agents[0].plants[0].resource == "R1"
    assert analysis.settings.agents[0].baseline is True

    assert [p.period for p in analysis.periods] == [1, 2]
    assert analysis.periods[1].met_demand == pytest.approx(110.5)
    assert analysis.periods[0].ptf == pytest.approx(40.0)
    assert len(analysis.offers) == 6
    assert [o.profit for o in analysis.offers] == pytest.approx(
        [2900, 0, -200, 2622.5, 2750, -200]
    )

    assert analysis.budgets["A"] == pytest.approx([1000.0, 3900.0, 9272.5])
    assert analysis.budgets["B"] == pytest.approx([500.0, 300.0, 100.0])

    a, b = analysis.agents
    assert a.initial_budget == pytest.approx(1000.0)
    assert a.final_budget == pytest.approx(9272.5)
    assert a.budget_change == pytest.approx(8272.5)
    assert a.mean_budget_change == pytest.approx(4136.25)
    assert a.stats.offers == 4
    assert a.stats.accepted_offers == 3
    assert a.stats.acceptance_rate == pytest.approx(75.0)
    assert a.stats.mean_offer_price == pytest.approx(38.75)
    assert a.stats.mean_accepted_offer_price == pytest.approx(35.0)
    assert a.stats.accepted_energy == pytest.approx(210.5)
    assert a.stats.revenue == pytest.approx(10077.5)
    assert a.stats.profit == pytest.approx(8272.5)
    r1, r2 = a.resources["R1"], a.resources["R2"]
    assert (r1.offers, r1.accepted_offers, r1.mean_offer_price) == (
        2,
        2,
        pytest.approx(42.5),
    )
    assert r1.accepted_energy == pytest.approx(160.5)
    assert r1.revenue == pytest.approx(7327.5)
    assert r1.profit == pytest.approx(5522.5)
    assert r2.acceptance_rate == pytest.approx(50.0)
    assert r2.mean_accepted_offer_price == pytest.approx(20.0)
    assert r2.profit == pytest.approx(2750.0)

    # agent without an accepted offer
    assert b.stats.accepted_offers == 0
    assert b.stats.acceptance_rate == 0.0
    assert b.stats.mean_accepted_offer_price is None
    assert b.stats.revenue == 0.0
    assert b.stats.profit == pytest.approx(-400.0)
    assert b.mean_budget_change == pytest.approx(-200.0)

    # baselines without step logs
    assert analysis.step_logs == {"A": [], "B": []}


def test_hand_run_scores(hand_run: Simulation) -> None:
    analysis = build_run_analysis(hand_run)

    whole = score_simulation(hand_run, range(1, 3))
    window = score_simulation(hand_run)
    assert analysis.corrected.price_cap == 200.0
    assert analysis.corrected.whole_run.agent_score == whole.agent_score
    assert analysis.corrected.whole_run.market_score == whole.market_score
    assert (
        analysis.corrected.whole_run.first_period,
        analysis.corrected.whole_run.last_period,
    ) == (1, 2)
    assert analysis.corrected.evaluation_window is not None
    assert analysis.corrected.evaluation_window.first_period == 2
    assert analysis.corrected.evaluation_window.agent_score == window.agent_score
    assert analysis.corrected.evaluation_window.market_score == window.market_score

    # Legacy rows as in the legacy JSON export: budget after the period.
    rows = [
        (1, "A", 100, 30, True, 3900.0),
        (1, "A", 50, 50, False, 3900.0),
        (1, "B", 80, 45, False, 300.0),
        (2, "A", 100, 55, True, 9272.5),
        (2, "A", 50, 20, True, 9272.5),
        (2, "B", 80, 60, False, 100.0),
    ]
    ptf = {1: 40.0, 2: 55.0}
    df = pd.DataFrame(
        [
            {
                "period": p,
                "agent": ag,
                "amount": amt,
                "offer_price": float(pr),
                "market_price": ptf[p],
                "accepted": acc,
                "budget": bud,
            }
            for p, ag, amt, pr, acc, bud in rows
        ]
    )
    ref_scores, ref_market = reference_scores(df)
    assert analysis.legacy.price_cap == 200.0
    assert analysis.legacy.agent_scores == pytest.approx(
        {k: float(v) for k, v in ref_scores.items()}
    )
    assert analysis.legacy.market_score == pytest.approx(float(ref_market))
    assert analysis.legacy.mean_agent_score == pytest.approx(
        sum(ref_scores.values()) / 2
    )


def test_hand_run_is_read_only_with_bounded_queries(hand_run: Simulation) -> None:
    simulation = Simulation.objects.get(pk=hand_run.pk)
    with CaptureQueriesContext(connection) as captured:
        analysis = build_run_analysis(simulation)
    statements = [q["sql"].lstrip().upper() for q in captured.captured_queries]
    assert not [s for s in statements if s.startswith(WRITE_PREFIXES)]
    assert len(statements) <= 20
    json.dumps(analysis.to_dict())  # JSON serialisable


def test_query_count_does_not_grow_with_the_run(hand_run: Simulation) -> None:
    """A run with more periods and offers uses the same number of queries."""
    small = Simulation.objects.get(pk=hand_run.pk)
    with CaptureQueriesContext(connection) as captured_small:
        build_run_analysis(small)
    market = small.market
    agent = market.agent_set.get(name="A")
    resource = agent.portfolio.plant_set.first().resource
    for number in range(3, 13):
        period = make_period(
            market, number, demand=1, met_demand=1, market_volume=1, ptf=Decimal("10")
        )
        make_offer(
            period,
            agent,
            resource,
            1,
            Decimal("5"),
            acceptance=False,
            acceptance_amount=Decimal("0"),
            save=True,
        )
    for other in market.agent_set.exclude(pk=agent.pk):
        other_resource = other.portfolio.plant_set.first().resource
        for period in market.period_set.filter(periodNumber__gte=3):
            make_offer(
                period,
                other,
                other_resource,
                1,
                Decimal("5"),
                acceptance=False,
                acceptance_amount=Decimal("0"),
                save=True,
            )
    big = Simulation.objects.get(pk=hand_run.pk)
    big.periodNumber = 12
    with CaptureQueriesContext(connection) as captured_big:
        build_run_analysis(big)
    assert len(captured_big.captured_queries) == len(captured_small.captured_queries)


# Unavailable runs -------------------------------------------------------


def test_unfinished_run_is_rejected(hand_run: Simulation) -> None:
    hand_run.state = SimulationState.STARTED
    with pytest.raises(RunAnalysisUnavailable, match="FINISHED"):
        build_run_analysis(hand_run)
    assert "FINISHED" in unavailable_reason(hand_run)


def test_run_without_periods_is_rejected() -> None:
    simulation = make_simulation(name="purged", period_number=3)
    simulation.state = SimulationState.FINISHED
    simulation.save()
    make_market(simulation)
    with pytest.raises(RunAnalysisUnavailable, match="no stored periods"):
        build_run_analysis(simulation)
    assert "no stored periods" in unavailable_reason(simulation)


def test_proxy_scenario_is_rejected(hand_run: Simulation) -> None:
    hand_run.proxy = True
    with pytest.raises(RunAnalysisUnavailable, match="scenario template"):
        build_run_analysis(hand_run)
    assert "scenario template" in unavailable_reason(hand_run)


def test_finished_run_with_data_has_no_reason(hand_run: Simulation) -> None:
    assert unavailable_reason(hand_run) is None


# Budget reconstruction --------------------------------------------------


def test_previous_budget_inverts_the_rounding() -> None:
    assert previous_budget(Decimal("100.00"), Decimal("12.344")) == Decimal("87.66")
    # tie: 87.655 + 12.345 = 100.000 exactly is not a tie, construct one
    before = Decimal("10.01")
    profit = Decimal("0.005")
    after = (before + profit).quantize(Decimal("0.01"))  # half-even -> 10.02
    assert previous_budget(after, profit) in {Decimal("10.01"), Decimal("10.02")}
    assert (previous_budget(after, profit) + profit).quantize(Decimal("0.01")) == after


def test_reconstruct_budget_series_forward_consistency() -> None:
    profits = [Decimal("1.005"), Decimal("-3.3333"), Decimal("0"), Decimal("1234.5678")]
    budget = Decimal("50.00")
    forward = [budget]
    for profit in profits:
        budget = (budget + profit).quantize(Decimal("0.01"))
        forward.append(budget)
    series = reconstruct_budget_series(forward[-1], profits)
    assert series[-1] == forward[-1]
    assert len(series) == len(profits) + 1
    for before, after, profit in zip(series, series[1:], profits):
        assert (before + profit).quantize(Decimal("0.01")) == after


# Real runs --------------------------------------------------------------


@pytest.fixture
def finished_run(
    fast_demand, settings, tmp_path: pathlib.Path
) -> Simulation:  # noqa: F811
    """A real 30-period run with a DDPG agent, two baselines, fail-stack and a JSON export."""
    settings.ABMEM_SIM_DATA_DIR = str(tmp_path)
    r = resources(4)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 400), (r[1], 200)],
                "algorithm": AlgorithmType.DDPG,
                "params": SMALL_DDPG,
                "budget": Decimal("1234.56"),
                "failstack": {"failstack_enabled": True, "failstack_limit": 1},
            },
            {
                "plants": [(r[2], 300)],
                "algorithm": AlgorithmType.MARGINAL_COST,
                "budget": Decimal("-50.25"),
            },
            {"plants": [(r[3], 300)], "algorithm": AlgorithmType.RANDOM},
        ],
        period_number=30,
        evaluation_periods=5,
        seed=0,
    )
    simulation_service.run(simulation.id, export_json=True)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    return simulation


def test_real_run_invariants(finished_run: Simulation) -> None:
    analysis = build_run_analysis(finished_run)
    agents = {a.name: a for a in Agent.objects.filter(market__simulation=finished_run)}

    assert analysis.settings.seed == 0
    initial = {"a0": 1234.56, "a1": -50.25, "a2": 0.0}
    for name, series in analysis.budgets.items():
        assert len(series) == 31
        assert series[-1] == pytest.approx(float(agents[name].budget), abs=0.01)
        assert series[0] == pytest.approx(initial[name], abs=0.01)

    whole = score_simulation(finished_run, range(1, 31))
    window = score_simulation(finished_run)
    assert analysis.corrected.whole_run.agent_score == whole.agent_score
    assert analysis.corrected.whole_run.market_score == whole.market_score
    assert analysis.corrected.evaluation_window.agent_score == window.agent_score
    assert analysis.corrected.evaluation_window.market_score == window.market_score

    assert (
        len(analysis.offers)
        == Offer.objects.filter(period__market__simulation=finished_run).count()
    )
    ddpg_rows = analysis.step_logs["a0"]
    assert ddpg_rows and [r.period for r in ddpg_rows] == sorted(
        r.period for r in ddpg_rows
    )
    assert any(r.evaluation_phase for r in ddpg_rows)
    assert set(analysis.agents[0].resources) == {
        r.resource for r in analysis.settings.agents[0].plants
    }
    json.dumps(analysis.to_dict())


def test_real_run_legacy_scores_reproduce_the_legacy_script(
    finished_run: Simulation, settings
) -> None:
    """The legacy loader + score function on the JSON export give the same scores."""
    path = (
        pathlib.Path(settings.ABMEM_SIM_DATA_DIR)
        / f"simulation_{finished_run.id}_offers.json"
    )
    exported = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    market_prices = {}
    # create_offers_dataframe of abc_log/try/sim/analyze_offers.py. The column
    # acceptance_price is left out: the scores do not use it and the current
    # export writes "None" for rejected offers.
    for period, entry in exported["offers_by_period"].items():
        market_prices[int(period)] = float(entry["market_price"])
        for agent_name, offers in entry.items():
            if agent_name != "market_price":
                for offer in offers:
                    rows.append(
                        {
                            "period": int(period),
                            "agent": offer["agent"],
                            "resource": offer["resource"],
                            "amount": offer["amount"],
                            "offer_price": float(offer["offerPrice"]),
                            "accepted": offer["acceptance"],
                            "budget": float(offer["budget"]),
                        }
                    )
    df = pd.DataFrame(rows)
    df["market_price"] = df["period"].map(market_prices)
    ref_scores, ref_market = reference_scores(df)

    analysis = build_run_analysis(finished_run)
    assert list(analysis.legacy.agent_scores) == list(ref_scores)
    for name, value in ref_scores.items():
        assert analysis.legacy.agent_scores[name] == pytest.approx(
            float(value), rel=1e-9, abs=1e-12
        )
    assert analysis.legacy.market_score == pytest.approx(float(ref_market), rel=1e-9)


def test_one_period_run(fast_demand) -> None:  # noqa: F811
    r = resources(2)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 400)],
                "algorithm": AlgorithmType.RANDOM,
                "budget": Decimal("10"),
            },
            {"plants": [(r[1], 400)], "algorithm": AlgorithmType.MARGINAL_COST},
        ],
        period_number=1,
    )
    simulation_service.run(simulation.id, export_json=False)
    simulation.refresh_from_db()
    analysis = build_run_analysis(simulation)
    assert len(analysis.periods) == 1
    assert analysis.budgets["a0"][0] == pytest.approx(10.0)
    assert analysis.corrected.evaluation_window is None
    assert (
        analysis.corrected.whole_run.first_period
        == analysis.corrected.whole_run.last_period
        == 1
    )
    assert set(analysis.legacy.agent_scores) == {"a0", "a1"}
