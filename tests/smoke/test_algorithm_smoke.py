"""End-to-end smoke tests of the simulation runtime for every algorithm (PLAN 1.14).

Scenario (OQ-7): 60 periods, lags ``[1, 24]``, one 1-plant and one 3-plant
agent of the same algorithm, PAYASBID and PAYASPTF. With lags ``[1, 24]`` the
first complete observation exists at ``t = 25`` and the first transition is
stored at ``t = 26``, so learning policies act from the agent generator for
``t <= 24`` and never learn under the production defaults (OQ-3) in 60
periods. Learning-path runs use the TEST-ONLY overrides of the conftest.
"""

from __future__ import annotations

import math
from decimal import Decimal

import pytest

from abmem.models.enums import AlgorithmType, MarketStrategy
from abmem.models.models import Offer
from abmem.services.agent.action_space import PRICE_QUANTUM
from abmem.services.agent.failstack import is_failure
from abmem.services.algorithms import registry
from abmem.services.market.accounting import offer_profit, outcome_from_offer
from tests.factories import make_resource
from tests.smoke.conftest import (
    ALL_TYPES,
    BASELINE_TYPES,
    LAGS,
    LEARNING_OVERRIDES,
    LEARNING_TYPES,
    LOWER,
    PERIODS,
    UPPER,
    SmokeRun,
    assert_run_invariants,
    resources,
    run_smoke,
    smoke_agents,
    step_rows,
)

pytestmark = [pytest.mark.smoke, pytest.mark.django_db]

FIRST_COMPLETE_OBS = max(LAGS) + 1
FIRST_TRANSITION = max(LAGS) + 2
EVALUATION_PERIODS = 10
STRATEGIES = (MarketStrategy.PAYASBID, MarketStrategy.PAYASPTF)


def _expected_update_periods(algorithm: AlgorithmType, last_train: int) -> list[int]:
    """Return the periods in which the learning-path run takes gradient steps.

    One transition is stored per period from ``FIRST_TRANSITION`` on.
    Off-policy methods update every period once ``learning_starts``
    transitions are stored, RDPG additionally needs one full window of
    ``seq_len`` steps, PPO updates whenever a rollout of ``rollout_length``
    steps is complete (the rollout buffer is cleared after each update).

    Args:
        algorithm: Learning algorithm.
        last_train: Last period of the training phase.

    Returns:
        Ascending period numbers.
    """
    config = registry.config_class(algorithm).from_dict(LEARNING_OVERRIDES[algorithm])
    if algorithm is AlgorithmType.PPO:
        length = int(config.rollout_length)
        return list(range(FIRST_TRANSITION + length - 1, last_train + 1, length))
    needed = int(config.learning_starts)
    if algorithm is AlgorithmType.RDPG:
        needed = max(needed, int(config.seq_len))
    return list(range(FIRST_TRANSITION + needed - 1, last_train + 1))


def _assert_integer_price_share(run: SmokeRun) -> None:
    """Check that integer prices are as rare as the price quantum implies.

    Prices on the bounds (which are integers) are excluded because clipped
    actions land there. For continuous actions quantised to ``PRICE_QUANTUM``
    the probability of an integer price is ``PRICE_QUANTUM`` (PLAN 1.5). The
    bound is the binomial mean plus four standard deviations plus one.

    Args:
        run: The run result.
    """
    prices = [
        Decimal(p)
        for p in Offer.objects.filter(period__market=run.simulation.market).values_list(
            "offerPrice", flat=True
        )
    ]
    interior = [p for p in prices if LOWER < p < UPPER]
    assert interior
    integers = sum(1 for p in interior if p == p.to_integral_value())
    q = float(PRICE_QUANTUM)
    n = len(interior)
    bound = n * q + 4.0 * math.sqrt(n * q * (1.0 - q)) + 1.0
    assert integers <= bound, (integers, n, bound)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.name)
@pytest.mark.parametrize("algorithm", ALL_TYPES, ids=lambda a: a.name)
def test_acting_smoke(
    algorithm: AlgorithmType, strategy: MarketStrategy, fast_demand
) -> None:
    """Production defaults: acting, offers and logging, no learning in 60 periods."""
    run = run_smoke(smoke_agents(algorithm, resources(4)), strategy=strategy)
    assert_run_invariants(run)
    learning = algorithm in LEARNING_TYPES

    for rt in run.ctx.agents:
        assert rt.failstack_params is None
        trace = run.traces[rt.agent.pk]
        rows = step_rows(run.simulation, rt.agent.pk)
        assert not any(row.learning_active for row in rows)
        assert all(row.critic_loss is None for row in rows)
        assert all(row.actor_loss is None for row in rows)
        assert not any(any(row.intervened_plants) for row in rows)
        assert trace.update_with_stats == []
        if learning:
            # Pre-learning phase: actions from the agent generator for t <= 24.
            assert trace.select_action == list(range(FIRST_COMPLETE_OBS, PERIODS + 1))
            assert trace.observe == list(range(FIRST_TRANSITION, PERIODS + 1))
            assert all(row.exploration_scale is not None for row in rows)
        else:
            assert trace.select_action == list(range(1, PERIODS + 1))
            assert trace.observe == []
            assert all(row.exploration_scale is None for row in rows)

    if algorithm not in (AlgorithmType.MARGINAL_COST, AlgorithmType.FIXED_MARKUP):
        _assert_integer_price_share(run)


@pytest.mark.parametrize("algorithm", LEARNING_TYPES, ids=lambda a: a.name)
def test_learning_smoke(algorithm: AlgorithmType, fast_demand) -> None:
    """TEST-ONLY overrides: gradient updates happen, the evaluation window is frozen."""
    run = run_smoke(
        smoke_agents(algorithm, resources(4), params=LEARNING_OVERRIDES[algorithm]),
        strategy=MarketStrategy.PAYASPTF,
        evaluation_periods=EVALUATION_PERIODS,
    )
    assert_run_invariants(run)
    last_train = PERIODS - EVALUATION_PERIODS
    expected_updates = _expected_update_periods(algorithm, last_train)
    assert expected_updates, "override must lead to an update in the training phase"

    for rt in run.ctx.agents:
        trace = run.traces[rt.agent.pk]
        rows = step_rows(run.simulation, rt.agent.pk)
        by_period = {row.period.periodNumber: row for row in rows}

        assert trace.observe == list(range(FIRST_TRANSITION, last_train + 1))
        assert trace.update_with_stats == expected_updates
        assert trace.select_action == list(range(FIRST_COMPLETE_OBS, PERIODS + 1))

        active = [t for t, row in by_period.items() if row.learning_active]
        if algorithm is AlgorithmType.PPO:
            assert active == expected_updates
        else:
            assert active == list(range(expected_updates[0], last_train + 1))
        for t, row in by_period.items():
            if t in expected_updates:
                assert row.critic_loss is not None, t
            else:
                assert row.critic_loss is None, t
            assert row.evaluation_phase == (t > last_train)
            if row.evaluation_phase:
                assert not row.learning_active
                assert row.actor_loss is None and row.mean_q is None
        assert rt.policy.frozen


def test_failstack_enabled_ddpg_intervenes_only_after_two_failures(
    fast_demand,
) -> None:
    """DDPG with L = 2: intervention at t iff two consecutive failures preceded it."""
    limit = 2
    run = run_smoke(
        smoke_agents(
            AlgorithmType.DDPG,
            resources(4),
            failstack={"failstack_enabled": True, "failstack_limit": limit},
        )
    )
    assert_run_invariants(run)
    total_interventions = 0
    for rt in run.ctx.agents:
        assert rt.failstack_params is not None
        assert rt.failstack_params.limit == limit
        min_margin = rt.failstack_params.min_margin
        offers = (
            Offer.objects.filter(agent=rt.agent)
            .select_related("period", "resource")
            .order_by("period__periodNumber", "id")
        )
        failures: dict[tuple[int, int], bool] = {}
        for offer in offers:
            outcome = outcome_from_offer(offer)
            failures[(offer.period.periodNumber, offer.resource_id)] = is_failure(
                outcome, offer_profit(outcome), min_margin
            )
        logged = {
            row.period.periodNumber: row.intervened_plants
            for row in step_rows(run.simulation, rt.agent.pk)
        }
        counters = [0] * len(rt.plants)
        for t in range(2, PERIODS + 1):
            expected: list[bool] = []
            for i, plant in enumerate(rt.plants):
                failed = failures[(t - 1, plant.resource_id)]
                counters[i] = counters[i] + 1 if failed else 0
                if counters[i] >= limit:
                    expected.append(True)
                    counters[i] = 0
                else:
                    expected.append(False)
            assert logged[t] == expected, t
            total_interventions += sum(expected)
    assert total_interventions > 0, "scenario must trigger the fail-stack rule"


def test_marginal_cost_above_upper_bound_is_clipped(fast_demand) -> None:
    """Variable cost above ``upperBidBound``: the MARGINAL_COST bid is the bound."""
    expensive = make_resource(fuel_cost=Decimal(UPPER + 50), om_cost=Decimal("1.00"))
    cheap = resources(3)
    run = run_smoke(
        [
            {"plants": [(expensive, 400)], "algorithm": AlgorithmType.MARGINAL_COST},
            {
                "plants": list(zip(cheap, (300, 250, 200))),
                "algorithm": AlgorithmType.MARGINAL_COST,
            },
        ]
    )
    assert_run_invariants(run)
    prices = set(
        Offer.objects.filter(resource=expensive).values_list("offerPrice", flat=True)
    )
    assert prices == {Decimal(UPPER)}


@pytest.mark.parametrize(
    "algorithm", BASELINE_TYPES + (AlgorithmType.DDPG,), ids=lambda a: a.name
)
def test_supply_shortfall_every_period(algorithm: AlgorithmType, fast_demand) -> None:
    """Total capacity below demand in every period: the run still finishes."""
    r = resources(4)
    run = run_smoke(
        [
            {"plants": [(r[0], 200)], "algorithm": algorithm},
            {"plants": list(zip(r[1:], (100, 100, 100))), "algorithm": algorithm},
        ]
    )
    assert_run_invariants(run)
    total_capacity = sum(rt.total_capacity for rt in run.ctx.agents)
    for period in run.simulation.market.period_set.all():
        assert period.demand > total_capacity
        assert period.metDemand <= total_capacity
