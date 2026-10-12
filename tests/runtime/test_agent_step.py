"""Unit tests of ``agent_service.step`` with stub policies (ARCHITECTURE.md 6.1).

No database access: the context is assembled from unsaved model instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from abmem.models import Agent
from abmem.models.models import Period, Plant, Resource
from abmem.services.agent import agent_service
from abmem.services.agent.action_space import from_price, to_price
from abmem.services.agent.failstack import FailStackParams, FailStackState
from abmem.services.agent.observation import ObservationNormalizer, build_observation
from abmem.services.agent.reward import compute_reward
from abmem.services.algorithms.base import ActionResult, PolicySpec, Transition
from abmem.services.market.accounting import OfferOutcome
from abmem.services.simulation.context import (
    AgentRuntimeState,
    RunPhase,
    SimulationContext,
)
from tests.runtime.conftest import constant_demand

LOW = Decimal("0")
HIGH = Decimal("200")


class StubPolicy:
    """Policy stub recording every call of the BaseAgentAlgorithm interface."""

    def __init__(self, action_dim: int, obs_dim: int, *, learning: bool = True) -> None:
        self.is_learning = learning
        self.spec = PolicySpec(
            obs_dim=obs_dim,
            action_dim=action_dim,
            price_low=float(LOW),
            price_high=float(HIGH),
            plant_variable_costs=(1.0,) * action_dim,
            plant_static_costs=(1.0,) * action_dim,
        )
        self.frozen = False
        self.selected: list[tuple[int, np.ndarray, bool]] = []
        self.observed: list[Transition] = []
        self.updates = 0
        self.end_periods = 0
        self.u = np.full(action_dim, 0.5, dtype=np.float32)
        self.t = 0

    @property
    def ready_to_learn(self) -> bool:
        return len(self.observed) >= 2

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        self.selected.append((self.t, np.array(obs, copy=True), explore))
        return ActionResult(u=self.u.copy(), log_prob=-1.0, value=0.5)

    def observe(self, transition: Transition) -> None:
        self.observed.append(transition)

    def update(self) -> None:
        self.updates += 1
        return None

    def end_period(self) -> None:
        self.end_periods += 1

    def exploration_scale(self) -> float | None:
        return 0.1 if self.is_learning else None


@dataclass
class RecordingLogger:
    """StepLogger stand-in that keeps the arguments of every ``add`` call."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        agent: Any,
        period: Any,
        reward: Any,
        stats: Any,
        exploration_scale: Any,
        intervened: Any,
        learning_active: Any,
        evaluation_phase: Any,
    ) -> None:
        self.rows.append(
            {
                "period": period,
                "reward": reward,
                "intervened": list(intervened),
                "learning_active": learning_active,
                "evaluation_phase": evaluation_phase,
            }
        )


def _plants(n: int) -> tuple[Plant, ...]:
    resource = Resource(
        name="r",
        fuelCost=Decimal("10"),
        omCost=Decimal("1"),
        investmentCost=0,
        co2Cost=0,
    )
    return tuple(
        Plant(id=i + 1, resource=resource, capacity=100 * (i + 1)) for i in range(n)
    )


def _ctx(
    policy: StubPolicy,
    n_plants: int,
    *,
    lags: tuple[int, ...] = (1, 24),
    evaluation_start: int = 10_000,
    failstack: FailStackParams | None = None,
) -> tuple[SimulationContext, AgentRuntimeState]:
    plants = _plants(n_plants)
    rt = AgentRuntimeState(
        agent=Agent(name="stub"),
        policy=policy,  # type: ignore[arg-type]
        rng=np.random.default_rng(0),
        plants=plants,
        total_capacity=sum((Decimal(p.capacity) for p in plants), Decimal(0)),
        failstack=FailStackState.zeros(n_plants),
        failstack_params=failstack,
    )
    ctx = SimulationContext(
        simulation=None,  # type: ignore[arg-type]
        market=None,  # type: ignore[arg-type]
        seed=0,
        lags=lags,
        evaluation_start=evaluation_start,
        price_low=LOW,
        price_high=HIGH,
        demand=constant_demand(),
        normalizer=ObservationNormalizer.for_market(lags, float(LOW), float(HIGH)),
        agents=[rt],
        ptf_history={},
        step_logger=RecordingLogger(),  # type: ignore[arg-type]
    )
    return ctx, rt


def _outcomes(rt: AgentRuntimeState, accepted: bool, price: Decimal) -> tuple:
    """Outcomes with distinct profits per plant (rejected or fully accepted)."""
    return tuple(
        OfferOutcome(
            quantity=Decimal(p.capacity),
            offer_price=price,
            accepted_quantity=(
                Decimal(p.capacity) if accepted and i % 2 == 0 else Decimal(0)
            ),
            acceptance_price=price if accepted and i % 2 == 0 else None,
            static_cost_rate=Decimal("1"),
            variable_cost_rate=Decimal("10"),
        )
        for i, p in enumerate(rt.plants)
    )


def _run(
    ctx: SimulationContext,
    rt: AgentRuntimeState,
    periods: int,
    *,
    accepted: bool = True,
) -> list[list]:
    """Run ``periods`` steps, simulating the clearing after each step."""
    offers_per_t = []
    for t in range(1, periods + 1):
        ctx.current_period = Period(periodNumber=t)
        rt.policy.t = t  # type: ignore[attr-defined]
        offers = agent_service.step(ctx, rt, t)
        offers_per_t.append(offers)
        ptf = Decimal(50 + (t % 7))
        ctx.ptf_history[t] = ptf
        rt.last_outcomes = _outcomes(rt, accepted, ptf)
    return offers_per_t


def test_first_period_has_no_reward_no_transition_no_log() -> None:
    policy = StubPolicy(2, 3)
    ctx, rt = _ctx(policy, 2)
    _run(ctx, rt, 1)
    assert policy.observed == []
    assert ctx.step_logger.rows == []  # type: ignore[attr-defined]
    assert policy.end_periods == 1


@pytest.mark.parametrize(
    ("lags", "first_complete"), [((1, 24), 25), ((1, 24, 168), 169)]
)
def test_transition_uses_the_decision_observation_c2(lags, first_complete) -> None:
    policy = StubPolicy(2, 1 + len(lags))
    ctx, rt = _ctx(policy, 2, lags=lags)
    _run(ctx, rt, first_complete + 2)

    selected = {t: obs for t, obs, _ in policy.selected}
    assert min(selected) == first_complete
    first = policy.observed[0]
    assert np.array_equal(first.obs, selected[first_complete])
    assert np.array_equal(first.next_obs, selected[first_complete + 1])
    second = policy.observed[1]
    assert np.array_equal(second.obs, selected[first_complete + 1])
    assert len(policy.observed) == 2
    # next_obs equals the freshly built observation of that period
    raw = build_observation(first_complete + 1, ctx.demand, ctx.ptf_history, lags).raw
    assert np.array_equal(first.next_obs, ctx.normalizer.transform(raw))


def test_reward_is_k1_over_all_plants_b2() -> None:
    policy = StubPolicy(3, 3)
    ctx, rt = _ctx(policy, 3)
    _run(ctx, rt, 26)
    expected = compute_reward(
        _outcomes(rt, True, Decimal(50 + (25 % 7))), rt.total_capacity, HIGH
    )
    assert policy.observed[0].reward == pytest.approx(expected, rel=0, abs=0)
    rows = ctx.step_logger.rows  # type: ignore[attr-defined]
    assert len(rows) == 25
    assert rows[-1]["reward"] == expected
    assert rows[-1]["period"].periodNumber == 26


def test_stored_action_is_applied_price_b6() -> None:
    policy = StubPolicy(2, 3)
    policy.u = np.array([0.123456, -0.98765], dtype=np.float32)
    ctx, rt = _ctx(policy, 2)
    offers = _run(ctx, rt, 26)
    prices = [o.offerPrice for o in offers[24]]
    assert prices == to_price(policy.u, LOW, HIGH)
    expected = from_price(prices, LOW, HIGH)
    assert np.array_equal(policy.observed[0].action, expected)
    assert policy.observed[0].behaviour_log_prob == -1.0
    assert [o.amount for o in offers[0]] == [100, 200]


def test_pre_learning_actions_are_random_and_not_stored_oq1() -> None:
    policy = StubPolicy(2, 3)
    ctx, rt = _ctx(policy, 2)
    offers = _run(ctx, rt, 24)
    assert policy.selected == []
    assert policy.observed == []
    prices = {o.offerPrice for t_offers in offers for o in t_offers}
    assert len(prices) > 10
    assert rt.prev is None


def test_baseline_acts_without_complete_observation() -> None:
    policy = StubPolicy(2, 3, learning=False)
    ctx, rt = _ctx(policy, 2)
    _run(ctx, rt, 30)
    assert len(policy.selected) == 30
    assert policy.observed == []
    assert all(not row["learning_active"] for row in ctx.step_logger.rows)  # type: ignore[attr-defined]


def test_evaluation_phase_does_not_learn_or_explore_oq5() -> None:
    policy = StubPolicy(2, 3)
    ctx, rt = _ctx(policy, 2, evaluation_start=28)
    _run(ctx, rt, 30)
    assert ctx.phase_for(27) is RunPhase.TRAIN
    assert ctx.phase_for(28) is RunPhase.EVALUATE
    # transitions stored at steps 26 and 27 only
    assert len(policy.observed) == 2
    explore = {t: e for t, _, e in policy.selected}
    assert explore[27] is True and explore[28] is False and explore[30] is False
    rows = ctx.step_logger.rows  # type: ignore[attr-defined]
    assert [r["evaluation_phase"] for r in rows][-3:] == [True, True, True]


def test_failstack_intervenes_after_limit_only() -> None:
    policy = StubPolicy(2, 3)
    params = FailStackParams(limit=3, delta=Decimal("5"), min_margin=Decimal("0"))
    ctx, rt = _ctx(policy, 2, failstack=params)
    offers = _run(ctx, rt, 10, accepted=False)
    rows = ctx.step_logger.rows  # type: ignore[attr-defined]
    flags = {row["period"].periodNumber: row["intervened"] for row in rows}
    # every step fails: counters reach 3 at the decision of t = 4, reset, then t = 7, 10
    assert flags[2] == [False, False] and flags[3] == [False, False]
    assert flags[4] == [True, True]
    assert flags[5] == [False, False] and flags[6] == [False, False]
    assert flags[7] == [True, True]
    assert flags[10] == [True, True]
    ptf_prev = ctx.ptf_history[3]
    for offer in offers[3]:
        assert ptf_prev - Decimal("5") <= offer.offerPrice < ptf_prev


def test_failstack_is_off_in_the_evaluation_window() -> None:
    # Limit 1 and permanent failure: every training decision from t = 2 on is
    # intervened, the evaluation window from t = 6 on is never intervened.
    policy = StubPolicy(2, 3)
    params = FailStackParams(limit=1, delta=Decimal("5"), min_margin=Decimal("0"))
    ctx, rt = _ctx(policy, 2, failstack=params, evaluation_start=6)
    _run(ctx, rt, 10, accepted=False)
    rows = ctx.step_logger.rows  # type: ignore[attr-defined]
    flags = {row["period"].periodNumber: row["intervened"] for row in rows}
    assert all(flags[t] == [True, True] for t in range(2, 6))
    assert all(flags[t] == [False, False] for t in range(6, 11))


def test_failstack_disabled_never_intervenes() -> None:
    policy = StubPolicy(2, 3)
    ctx, rt = _ctx(policy, 2, failstack=None)
    _run(ctx, rt, 30, accepted=False)
    rows = ctx.step_logger.rows  # type: ignore[attr-defined]
    assert all(row["intervened"] == [False, False] for row in rows)
    assert all(t.intervened is False for t in policy.observed)


def test_intervened_flag_reaches_the_transition() -> None:
    policy = StubPolicy(1, 3)
    params = FailStackParams(limit=1, delta=Decimal("5"), min_margin=Decimal("0"))
    ctx, rt = _ctx(policy, 1, failstack=params)
    _run(ctx, rt, 27, accepted=False)
    # limit 1 and permanent failure: every decision from t = 2 on is intervened
    assert policy.observed and all(t.intervened for t in policy.observed)


def test_step_without_outcomes_after_first_period_raises() -> None:
    policy = StubPolicy(1, 3)
    ctx, rt = _ctx(policy, 1)
    ctx.current_period = Period(periodNumber=2)
    with pytest.raises(ValueError):
        agent_service.step(ctx, rt, 2)
