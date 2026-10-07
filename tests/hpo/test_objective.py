"""Scores of evaluation simulations computed from hand-built DB data (PLAN 1.11, B9)."""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from abmem.models.enums import AgentType
from abmem.services.hpo.objective import score_simulation, scored_periods
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offer,
    make_period,
    make_resource,
    make_simulation,
)

pytestmark = pytest.mark.django_db


def _build(evaluation_periods: int = 0):
    """Two agents, two periods, P_max = 100, static cost 2, variable cost 10.

    Agent A has two plants (30 + 20 MWh) and a non-zero budget that must not
    enter the score. Agent B has one plant of 40 MWh.
    """
    sim = make_simulation(period_number=2)
    sim.evaluation_periods = evaluation_periods
    sim.save()
    market = make_market(simulation=sim, upper_bid_bound=100)
    res = make_resource(fuel_cost=Decimal("10"), om_cost=Decimal("2"))
    a = make_agent_with_plants(
        market,
        [(res, 30), (res, 20)],
        name="A",
        agent_type=AgentType.HYBRID,
        budget=Decimal("5000"),
    )
    b = make_agent_with_plants(
        market, [(res, 40)], name="B", agent_type=AgentType.HYBRID
    )
    p1 = make_period(market, 1, ptf=Decimal("50"))
    p2 = make_period(market, 2, ptf=Decimal("60"))
    rows = [
        (p1, a, 30, "40", True, "50", "30"),
        (p1, a, 20, "70", False, None, "0"),
        (p1, b, 40, "50", True, "50", "20"),
        (p2, a, 50, "60", True, "60", "50"),
        (p2, b, 40, "80", False, None, "0"),
    ]
    for period, agent, amount, price, acc, acc_price, acc_amount in rows:
        make_offer(
            period,
            agent,
            res,
            amount,
            Decimal(price),
            acceptance=acc,
            acceptance_price=None if acc_price is None else Decimal(acc_price),
            acceptance_amount=Decimal(acc_amount),
            save=True,
        )
    return sim


def test_scores_all_periods_with_b9_corrections() -> None:
    sim = _build()
    scores = score_simulation(sim)
    # Agent A: G = 1140 - 40 + 2400 = 3500, C = 50 (both plants), T = 2, P_max = 100.
    a_budget = 3500 / (50 * 2 * 100)
    a_profit = (3500 / 80) / 100
    a_sim = 1 - math.sqrt((100 + 400 + 0) / 3) / 100
    a = 0.4 * a_budget + 0.4 * a_profit + 0.2 * a_sim
    # Agent B: G = 720 - 80 = 640, C = 40, accepted offered quantity 40.
    b = 0.4 * (640 / 8000) + 0.4 * (640 / 40 / 100) + 0.2 * (1 - math.sqrt(200) / 100)
    assert scores.agent_score == pytest.approx((a + b) / 2)
    assert scores.market_score == pytest.approx(0.6 * (1 - 0.54) + 0.4 * (54 / 60))


def test_evaluation_window_scores_last_periods_only() -> None:
    sim = _build(evaluation_periods=1)
    assert scored_periods(sim) == range(2, 3)
    scores = score_simulation(sim)
    a = 0.4 * 0.48 + 0.4 * 0.48 + 0.2 * 1.0
    b = 0.4 * (-80 / 4000) + 0.0 + 0.2 * 0.8
    assert scores.agent_score == pytest.approx((a + b) / 2)
    assert scores.market_score == pytest.approx(0.6 * 0.4 + 0.4 * 60 / 70)


def test_explicit_periods_argument() -> None:
    sim = _build()
    assert score_simulation(sim, range(2, 3)) == score_simulation(
        _build(evaluation_periods=1)
    )


def test_agent_score_is_clipped_at_zero() -> None:
    sim = make_simulation(period_number=1)
    market = make_market(simulation=sim, upper_bid_bound=100)
    res = make_resource(om_cost=Decimal("90"))
    a = make_agent_with_plants(
        market, [(res, 10)], name="A", agent_type=AgentType.HYBRID
    )
    p1 = make_period(market, 1, ptf=Decimal("10"))
    make_offer(
        p1,
        a,
        res,
        10,
        Decimal("100"),
        acceptance=False,
        acceptance_amount=Decimal("0"),
        save=True,
    )
    assert score_simulation(sim).agent_score == 0.0


def test_no_offers_raises() -> None:
    sim = make_simulation(period_number=1)
    make_market(simulation=sim, upper_bid_bound=100)
    with pytest.raises(ValueError, match="no cleared offers"):
        score_simulation(sim)


def test_non_positive_price_cap_raises() -> None:
    sim = make_simulation(period_number=1)
    make_market(simulation=sim, lower_bid_bound=-10, upper_bid_bound=0)
    with pytest.raises(ValueError, match="upperBidBound"):
        score_simulation(sim)
