"""Agent step of one period (ARCHITECTURE.md 6.1).

``step`` builds the observation of period ``t`` once and uses it for acting
and as ``next_obs`` of the stored transition (C2). The reward of ``t - 1`` is
the K1 reward over all plants (C1, B2). Prices come from the normalised
action through the market bid bounds (B6), the fail-stack rule is applied
identically for every learning policy (K3), and the stored action is the
normalised action of the prices actually submitted.

The module holds no mutable module-level state (B10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abmem.models.enums import AgentState
from abmem.models.models import Offer
from abmem.services.agent.action_space import from_price, to_price
from abmem.services.agent.failstack import (
    apply_interventions,
    is_failure,
    update_counters,
)
from abmem.services.agent.observation import build_observation
from abmem.services.agent.reward import compute_reward
from abmem.services.algorithms.base import ActionResult, Transition
from abmem.services.market.accounting import offer_profit
from abmem.services.simulation.context import PendingDecision, RunPhase

if TYPE_CHECKING:
    from abmem.services.algorithms.base import UpdateStats
    from abmem.services.simulation.context import AgentRuntimeState, SimulationContext


def step(ctx: SimulationContext, rt: AgentRuntimeState, t: int) -> list[Offer]:
    """Run the decision of one agent for period ``t`` (before clearing of ``t``).

    An agent with ``rt.evaluate_only`` (frozen warm-started policy) is in the
    evaluation phase in every period: no exploration, no stored transitions.

    Args:
        ctx: Run context. ``ctx.current_period`` is the period ``t``. It is
            inserted by ``market_service.run_period`` after clearing, the
            returned offers and the step-log row reference the instance.
        rt: Runtime state of the agent. ``rt.last_outcomes`` holds the
            cleared outcomes of ``t - 1`` when ``t > 1``.
        t: Period number, ``>= 1``.

    Returns:
        One unsaved ``Offer`` per plant in plant order, with
        ``amount = plant capacity``.

    Raises:
        ValueError: If ``t > 1`` and ``rt.last_outcomes`` is empty, or any
            component rejects its input.
    """
    phase = RunPhase.EVALUATE if rt.evaluate_only else ctx.phase_for(t)
    policy = rt.policy
    n_plants = len(rt.plants)
    obs = build_observation(t, ctx.demand, ctx.ptf_history, ctx.lags)
    obs_norm = ctx.normalizer.transform(obs.raw) if obs.complete else None
    stats: UpdateStats | None = None
    reward: float | None = None

    if t > 1:
        reward = compute_reward(rt.last_outcomes, rt.total_capacity, ctx.price_high)
        if rt.failstack_params is not None:
            min_margin = rt.failstack_params.min_margin
            update_counters(
                rt.failstack,
                [is_failure(o, offer_profit(o), min_margin) for o in rt.last_outcomes],
            )
        if (
            rt.prev is not None
            and obs_norm is not None
            and phase is RunPhase.TRAIN
            and policy.is_learning
        ):
            prev = rt.prev
            policy.observe(
                Transition(
                    obs=prev.obs_norm,
                    action=prev.u_applied,
                    reward=reward,
                    next_obs=obs_norm,
                    behaviour_log_prob=prev.action.log_prob,
                    behaviour_value=prev.action.value,
                    pre_squash=prev.action.pre_squash,
                    intervened=prev.intervened,
                )
            )
            stats = policy.update()

    explore = phase is RunPhase.TRAIN
    if obs_norm is not None:
        action = policy.select_action(obs_norm, explore=explore)
    elif not policy.is_learning:
        # Baselines ignore the observation, a zero vector satisfies the
        # shape contract of select_action before all lags exist.
        placeholder = np.zeros(policy.spec.obs_dim, dtype=np.float32)
        action = policy.select_action(placeholder, explore=explore)
    else:
        # OQ-1: pre-learning action from the agent generator, never stored.
        u = rt.rng.uniform(-1.0, 1.0, size=n_plants).astype(np.float32)
        action = ActionResult(u=u)

    prices = to_price(action.u, ctx.price_low, ctx.price_high)
    if rt.failstack_params is not None:
        prices, intervened = apply_interventions(
            rt.failstack,
            rt.failstack_params,
            prices,
            ctx.ptf_history.get(t - 1),
            ctx.price_low,
            ctx.price_high,
            rt.rng,
        )
    else:
        intervened = [False] * n_plants
    u_applied = from_price(prices, ctx.price_low, ctx.price_high)
    rt.prev = (
        PendingDecision(
            obs_norm=obs_norm,
            action=action,
            u_applied=u_applied,
            intervened=any(intervened),
        )
        if obs_norm is not None
        else None
    )

    if t > 1:
        learning_active = (
            phase is RunPhase.TRAIN
            and policy.is_learning
            and not policy.frozen
            and (stats is not None or policy.ready_to_learn)
        )
        ctx.step_logger.add(
            rt.agent,
            ctx.current_period,
            reward,
            stats,
            policy.exploration_scale(),
            intervened,
            learning_active,
            phase is RunPhase.EVALUATE,
        )

    policy.end_period()
    rt.agent.state = AgentState.WAITING
    return [
        Offer(
            period=ctx.current_period,
            agent=rt.agent,
            resource=plant.resource,
            amount=plant.capacity,
            offerPrice=price,
        )
        for plant, price in zip(rt.plants, prices)
    ]
