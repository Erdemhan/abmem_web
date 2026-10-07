"""Fixtures and helpers of the per-algorithm smoke tests (validation.smoke).

The scenario follows OQ-7: 60 periods, observation lags ``[1, 24]``, two agents
(one with 1 plant, one with 3 plants). Learning-path runs use the TEST-ONLY
overrides in :data:`LEARNING_OVERRIDES` so that gradient updates happen within
60 periods. The production defaults (OQ-3) are not changed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final

import pytest
import torch

from abmem.models import AgentStepLog, Simulation
from abmem.models.enums import AlgorithmType, MarketStrategy, SimulationState
from abmem.models.models import Offer, Period, Resource
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import SimulationContext
from tests.runtime.conftest import fast_demand, make_run, resources  # noqa: F401

PERIODS: Final[int] = 60
LAGS: Final[tuple[int, ...]] = (1, 24)
LOWER: Final[int] = 0
UPPER: Final[int] = 200
ONE_PLANT_CAPACITY: Final[int] = 400
THREE_PLANT_CAPACITIES: Final[tuple[int, ...]] = (300, 250, 200)
STEP_LOG_FLOATS: Final[tuple[str, ...]] = (
    "reward",
    "critic_loss",
    "actor_loss",
    "mean_q",
    "exploration_scale",
)

ALL_TYPES: Final[tuple[AlgorithmType, ...]] = tuple(AlgorithmType)
LEARNING_TYPES: Final[tuple[AlgorithmType, ...]] = (
    AlgorithmType.DDPG,
    AlgorithmType.TD3,
    AlgorithmType.SAC,
    AlgorithmType.PPO,
    AlgorithmType.RDPG,
)
BASELINE_TYPES: Final[tuple[AlgorithmType, ...]] = (
    AlgorithmType.RANDOM,
    AlgorithmType.MARGINAL_COST,
    AlgorithmType.FIXED_MARKUP,
)

_SMALL_COMMON: Final[dict[str, Any]] = {
    "batch_size": 8,
    "learning_starts": 8,
    "buffer_size": 512,
    "hidden_sizes": [16],
}

# TEST-ONLY overrides (OQ-7). PPO needs rollout_length >= learning_starts and
# minibatch_size <= rollout_length, RDPG gets a small LSTM. Network sizes are
# reduced only to keep the smoke matrix fast on CPU.
LEARNING_OVERRIDES: Final[Mapping[AlgorithmType, dict[str, Any]]] = {
    AlgorithmType.DDPG: dict(_SMALL_COMMON),
    AlgorithmType.TD3: dict(_SMALL_COMMON),
    AlgorithmType.SAC: dict(_SMALL_COMMON),
    AlgorithmType.PPO: {
        **_SMALL_COMMON,
        "rollout_length": 8,
        "minibatch_size": 8,
        "epochs": 2,
    },
    AlgorithmType.RDPG: {**_SMALL_COMMON, "lstm_hidden": 16},
}


@pytest.fixture(autouse=True)
def _restore_torch_determinism_smoke() -> Iterator[None]:
    """Undo ``torch.use_deterministic_algorithms(True)`` set by seeding."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


@dataclass
class PolicyTrace:
    """Periods in which the runtime called a policy method.

    Attributes:
        select_action: Periods of ``select_action`` calls.
        observe: Periods of ``observe`` calls.
        update_with_stats: Periods in which ``update`` returned statistics.
    """

    select_action: list[int] = field(default_factory=list)
    observe: list[int] = field(default_factory=list)
    update_with_stats: list[int] = field(default_factory=list)


@dataclass
class SmokeRun:
    """Result of one smoke run.

    Attributes:
        simulation: The run, refreshed after the run loop.
        ctx: Context built by the run.
        traces: Policy call traces by agent primary key.
    """

    simulation: Simulation
    ctx: SimulationContext
    traces: dict[int, PolicyTrace]


def _install_spies(ctx: SimulationContext, traces: dict[int, PolicyTrace]) -> None:
    """Wrap the policy methods of every agent to record the calling period.

    Args:
        ctx: Freshly built context, no period run yet.
        traces: Output mapping, filled per agent primary key.
    """

    def _period() -> int:
        assert ctx.current_period is not None
        return int(ctx.current_period.periodNumber)

    for rt in ctx.agents:
        trace = PolicyTrace()
        traces[rt.agent.pk] = trace
        policy = rt.policy
        original_select = policy.select_action
        original_observe = policy.observe
        original_update = policy.update

        def select_action(
            obs: Any,
            explore: bool,
            _orig: Callable[..., Any] = original_select,
            _trace: PolicyTrace = trace,
        ) -> Any:
            _trace.select_action.append(_period())
            return _orig(obs, explore=explore)

        def observe(
            transition: Any,
            _orig: Callable[..., Any] = original_observe,
            _trace: PolicyTrace = trace,
        ) -> Any:
            _trace.observe.append(_period())
            return _orig(transition)

        def update(
            _orig: Callable[..., Any] = original_update,
            _trace: PolicyTrace = trace,
        ) -> Any:
            stats = _orig()
            if stats is not None:
                _trace.update_with_stats.append(_period())
            return stats

        policy.select_action = select_action  # type: ignore[method-assign]
        policy.observe = observe  # type: ignore[method-assign]
        policy.update = update  # type: ignore[method-assign]


def smoke_agents(
    algorithm: AlgorithmType,
    plant_resources: Sequence[Resource],
    *,
    params: Mapping[str, Any] | None = None,
    failstack: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return the two agent specs of the smoke scenario for ``make_run``.

    Args:
        algorithm: Algorithm of both agents.
        plant_resources: Four distinct resources (plant 0 for the 1-plant
            agent, plants 1..3 for the 3-plant agent).
        params: ``Agent.algorithm_params`` of both agents.
        failstack: Fail-stack model fields of both agents.

    Returns:
        Agent specs in ``Agent.id`` order.
    """
    common: dict[str, Any] = {"algorithm": algorithm, "params": dict(params or {})}
    if failstack:
        common["failstack"] = dict(failstack)
    return [
        {"plants": [(plant_resources[0], ONE_PLANT_CAPACITY)], **common},
        {
            "plants": list(zip(plant_resources[1:4], THREE_PLANT_CAPACITIES)),
            **common,
        },
    ]


def run_smoke(
    agents: Sequence[dict[str, Any]],
    *,
    strategy: MarketStrategy = MarketStrategy.PAYASBID,
    evaluation_periods: int = 0,
    seed: int = 11,
) -> SmokeRun:
    """Create and run a 60-period smoke scenario with spies on every policy.

    Args:
        agents: Agent specs for ``make_run``.
        strategy: Market strategy.
        evaluation_periods: Size of the evaluation window.
        seed: Run seed.

    Returns:
        The run result.
    """
    simulation = make_run(
        agents,
        period_number=PERIODS,
        seed=seed,
        lower=LOWER,
        upper=UPPER,
        strategy=strategy,
        evaluation_periods=evaluation_periods,
        observation_lags=LAGS,
    )
    holder: dict[str, SimulationContext] = {}
    traces: dict[int, PolicyTrace] = {}
    original_build = SimulationContext.build

    def _build(*args: Any, **kwargs: Any) -> SimulationContext:
        ctx = original_build(*args, **kwargs)
        _install_spies(ctx, traces)
        holder["ctx"] = ctx
        return ctx

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SimulationContext, "build", _build)
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    return SmokeRun(simulation=simulation, ctx=holder["ctx"], traces=traces)


def step_rows(simulation: Simulation, agent_id: int) -> list[AgentStepLog]:
    """Return the step-log rows of one agent ordered by period number.

    Args:
        simulation: The run.
        agent_id: Agent primary key.

    Returns:
        The rows with their periods loaded.
    """
    return list(
        AgentStepLog.objects.filter(agent_id=agent_id, period__market=simulation.market)
        .select_related("period")
        .order_by("period__periodNumber")
    )


def assert_run_invariants(run: SmokeRun) -> None:
    """Check the invariants every smoke run must satisfy.

    Simulation FINISHED, ``T`` periods with a finite PTF, exactly one offer per
    plant per period with a finite price inside the bid bounds, finite
    budgets, ``T - 1`` step-log rows per agent with finite floats and no
    ``extra['non_finite']``, and one intervention flag per plant.

    Args:
        run: The run result.
    """
    simulation = run.simulation
    market = simulation.market
    assert simulation.state == SimulationState.FINISHED, simulation.error
    assert simulation.currentPeriod == PERIODS

    periods = list(Period.objects.filter(market=market).order_by("periodNumber"))
    assert [p.periodNumber for p in periods] == list(range(1, PERIODS + 1))
    for period in periods:
        assert period.ptf is not None and Decimal(period.ptf).is_finite()
        assert LOWER <= period.ptf <= UPPER

    low, high = Decimal(market.lowerBidBound), Decimal(market.upperBidBound)
    offers_per_key: dict[tuple[int, int, int], int] = defaultdict(int)
    offers = Offer.objects.filter(period__market=market).select_related("period")
    for offer in offers:
        price = Decimal(offer.offerPrice)
        assert price.is_finite()
        assert low <= price <= high, (offer.period.periodNumber, price)
        offers_per_key[
            (offer.period.periodNumber, offer.agent_id, offer.resource_id)
        ] += 1
    expected_keys = {
        (t, rt.agent.pk, plant.resource_id)
        for t in range(1, PERIODS + 1)
        for rt in run.ctx.agents
        for plant in rt.plants
    }
    assert set(offers_per_key) == expected_keys
    assert set(offers_per_key.values()) == {1}

    for agent in market.agent_set.all():
        assert Decimal(agent.budget).is_finite()

    for rt in run.ctx.agents:
        rows = step_rows(simulation, rt.agent.pk)
        assert [r.period.periodNumber for r in rows] == list(range(2, PERIODS + 1))
        for row in rows:
            assert "non_finite" not in row.extra, row.extra
            for name in STEP_LOG_FLOATS:
                value = getattr(row, name)
                assert value is None or math.isfinite(value), (name, value)
            for value in row.extra.values():
                assert math.isfinite(float(value)), row.extra
            assert row.reward is not None
            assert len(row.intervened_plants) == len(rt.plants)
