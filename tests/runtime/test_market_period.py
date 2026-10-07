"""Tests of ``market_service.run_period`` (ARCHITECTURE.md 6.2)."""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np
import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from abmem.models import AgentStepLog
from abmem.models.enums import MarketState, MarketStrategy
from abmem.models.models import Offer, Period
from abmem.services.agent.action_space import to_price
from abmem.services.algorithms.base import ActionResult
from abmem.services.market import market_service
from abmem.services.market.accounting import offer_profit, outcome_from_offer
from abmem.services.simulation.context import SimulationContext
from tests.runtime.conftest import constant_demand, make_run, resources

pytestmark = pytest.mark.django_db


class FixedPolicy:
    """Deterministic non-learning stub returning a constant action."""

    is_learning = False
    frozen = False
    ready_to_learn = False

    def __init__(self, base, u: float) -> None:
        self.spec = base.spec
        self.u = np.full(base.spec.action_dim, u, dtype=np.float32)

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        return ActionResult(u=self.u.copy())

    def observe(self, transition) -> None:
        raise AssertionError("baseline must not observe")

    def update(self) -> None:
        return None

    def end_period(self) -> None:
        return None

    def exploration_scale(self) -> None:
        return None


def _context(strategy: MarketStrategy = MarketStrategy.PAYASBID) -> SimulationContext:
    r1, r2, r3 = resources(3)
    simulation = make_run(
        [
            {"plants": [(r1, 400)], "budget": Decimal("100.00")},
            {"plants": [(r2, 300), (r3, 500)], "budget": Decimal("-5.50")},
        ],
        period_number=5,
        strategy=strategy,
    )
    ctx = SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False
    )
    ctx.agents[0].policy = FixedPolicy(ctx.agents[0].policy, -0.5)  # type: ignore[assignment]
    ctx.agents[1].policy = FixedPolicy(ctx.agents[1].policy, 0.2)  # type: ignore[assignment]
    return ctx


@pytest.mark.parametrize("strategy", [MarketStrategy.PAYASBID, MarketStrategy.PAYASPTF])
def test_period_writes_offers_budgets_and_state(strategy: MarketStrategy) -> None:
    ctx = _context(strategy)
    initial = [rt.agent.budget for rt in ctx.agents]

    period = market_service.run_period(ctx, 1)

    offers = list(
        Offer.objects.filter(period=period).select_related("resource").order_by("id")
    )
    assert len(offers) == 3
    assert [o.agent_id for o in offers] == [
        ctx.agents[0].agent.id,
        ctx.agents[1].agent.id,
        ctx.agents[1].agent.id,
    ]
    assert [o.amount for o in offers] == [400, 300, 500]
    p_low = to_price(np.array([-0.5], dtype=np.float32), ctx.price_low, ctx.price_high)[
        0
    ]
    p_high = to_price(np.array([0.2], dtype=np.float32), ctx.price_low, ctx.price_high)[
        0
    ]
    assert [o.offerPrice for o in offers] == [p_low, p_high, p_high]

    stored = Period.objects.get(pk=period.pk)
    demand = ctx.demand.at(1)
    assert stored.periodNumber == 1
    assert stored.demand == demand
    assert stored.metDemand == demand
    assert stored.marketVolume == 1200
    assert stored.ptf == p_high
    assert ctx.ptf_history == {1: p_high}

    # 400 at the low price fully, the marginal group of 800 shares the rest.
    accepted = [o.acceptanceAmount for o in offers]
    assert accepted[0] == 400 and sum(accepted) == demand
    for offer in offers:
        assert offer.acceptance is True
        expected_price = (
            offer.offerPrice if strategy == MarketStrategy.PAYASBID else p_high
        )
        assert offer.acceptancePrice == expected_price

    for rt, start in zip(ctx.agents, initial):
        own = [o for o in offers if o.agent_id == rt.agent.id]
        profit = sum((offer_profit(outcome_from_offer(o)) for o in own), Decimal(0))
        expected = (start + profit).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
        rt.agent.refresh_from_db()
        assert rt.agent.budget == expected
        assert len(rt.last_outcomes) == len(rt.plants)

    ctx.market.refresh_from_db()
    assert ctx.market.state == MarketState.PERIODEND
    assert AgentStepLog.objects.count() == 0


def test_second_period_logs_one_row_per_agent() -> None:
    ctx = _context()
    market_service.run_period(ctx, 1)
    period = market_service.run_period(ctx, 2)
    rows = AgentStepLog.objects.filter(period=period).order_by("agent_id")
    assert [r.agent_id for r in rows] == [rt.agent.id for rt in ctx.agents]
    assert all(r.reward is not None for r in rows)
    assert all(r.learning_active is False for r in rows)
    assert [len(r.intervened_plants) for r in rows] == [1, 2]


def test_database_statements_per_period_do_not_grow_with_agents() -> None:
    ctx = _context()
    market_service.run_period(ctx, 1)
    with CaptureQueriesContext(connection) as small:
        market_service.run_period(ctx, 2)

    r = resources(6)
    simulation = make_run(
        [{"plants": [(r[i], 100), (r[(i + 1) % 6], 50)]} for i in range(6)],
        period_number=5,
    )
    big = SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False
    )
    market_service.run_period(big, 1)
    with CaptureQueriesContext(connection) as large:
        market_service.run_period(big, 2)
    assert len(large.captured_queries) == len(small.captured_queries)


def test_policy_error_rolls_back_the_period() -> None:
    ctx = _context()
    market_service.run_period(ctx, 1)

    def _boom(obs, explore):
        raise RuntimeError("policy failure")

    ctx.agents[1].policy.select_action = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="policy failure"):
        market_service.run_period(ctx, 2)
    assert list(
        Period.objects.filter(market=ctx.market).values_list("periodNumber", flat=True)
    ) == [1]
    assert Offer.objects.filter(period__market=ctx.market).count() == 3
    assert AgentStepLog.objects.count() == 0
    assert len(ctx.step_logger) == 0


def test_policy_type_is_from_agent_algorithm_type() -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)], "algorithm": "MARGINAL_COST"}])
    ctx = SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False
    )
    assert ctx.agents[0].policy.key == "MARGINAL_COST"
    assert ctx.agents[0].agent.runtime_policy is ctx.agents[0].policy
