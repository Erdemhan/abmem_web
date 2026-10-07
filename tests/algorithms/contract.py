"""Reusable policy contract check shared by every algorithm test module.

Usage from an algorithm test::

    from tests.algorithms.contract import assert_policy_contract

    def test_contract(spec, rng):
        policy = DDPGAlgorithm(spec, DDPGConfig(...), rng)
        assert_policy_contract(policy, spec)
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from abmem.services.algorithms.base import (
    ActionResult,
    BaseAgentAlgorithm,
    PolicySpec,
    Transition,
    UpdateStats,
)


def _parameter_snapshot(policy: BaseAgentAlgorithm) -> dict[str, torch.Tensor]:
    """Return clones of all parameters of all nn.Module attributes of ``policy``."""
    snapshot: dict[str, torch.Tensor] = {}
    for attr, value in vars(policy).items():
        if isinstance(value, nn.Module):
            for name, tensor in value.state_dict().items():
                snapshot[f"{attr}.{name}"] = tensor.detach().clone()
        elif isinstance(value, torch.Tensor):
            snapshot[attr] = value.detach().clone()
    return snapshot


def _check_action(result: ActionResult, spec: PolicySpec) -> None:
    """Assert the action invariants of the policy contract."""
    assert isinstance(result, ActionResult), type(result)
    assert result.u.shape == (spec.action_dim,), result.u.shape
    assert result.u.dtype == np.float32, result.u.dtype
    assert np.all(np.isfinite(result.u)), result.u
    assert np.all(result.u >= -1.0) and np.all(result.u <= 1.0), result.u


def _check_stats(stats: UpdateStats) -> None:
    """Assert that reported losses are finite floats or None."""
    assert isinstance(stats, UpdateStats), type(stats)
    for value in (stats.critic_loss, stats.actor_loss, stats.mean_q):
        assert value is None or (
            isinstance(value, float) and math.isfinite(value)
        ), value


def assert_policy_contract(
    policy: BaseAgentAlgorithm, spec: PolicySpec, n_steps: int = 300
) -> None:
    """Check the policy contract of ARCHITECTURE.md section 5.2.

    Checks: output shape equals ``action_dim``, values in ``[-1, 1]`` and
    finite, ``explore=False`` deterministic, no update before
    ``learning_starts``, ``update`` returns UpdateStats after
    ``learning_starts`` for learning policies, a frozen policy does not change
    its parameters or buffer, and ``observe`` stores float rewards unchanged
    as float32.

    Args:
        policy: A freshly constructed, unfrozen policy.
        spec: The spec the policy was built with.
        n_steps: Number of simulated steps. Must exceed ``learning_starts`` for
            learning policies.
    """
    assert not policy.frozen
    data_rng = np.random.default_rng(20261006)
    learning = bool(policy.is_learning)
    if learning:
        assert n_steps > policy.config.learning_starts, (
            f"n_steps={n_steps} must exceed learning_starts="
            f"{policy.config.learning_starts}"
        )

    def random_obs() -> np.ndarray:
        return data_rng.uniform(-1.0, 1.0, size=spec.obs_dim).astype(np.float32)

    # Determinism of the greedy action. Non-learning policies that are random by
    # definition (RANDOM baseline) declare `stochastic_when_greedy = True` and
    # stay random in the evaluation window, so only validity is checked.
    probe = random_obs()
    first = policy.select_action(probe, explore=False)
    second = policy.select_action(probe.copy(), explore=False)
    _check_action(first, spec)
    _check_action(second, spec)
    if getattr(policy, "stochastic_when_greedy", False):
        assert (
            not policy.is_learning
        ), "only non-learning policies may be stochastic when greedy"
    else:
        np.testing.assert_array_equal(first.u, second.u)

    obs = random_obs()
    observed = 0
    stats_after_start = 0
    for _ in range(n_steps):
        result = policy.select_action(obs, explore=True)
        _check_action(result, spec)
        next_obs = random_obs()
        reward = float(data_rng.normal(0.0, 1000.0)) + 0.123456
        policy.observe(
            Transition(
                obs=obs,
                action=result.u,
                reward=reward,
                next_obs=next_obs,
                behaviour_log_prob=result.log_prob,
                behaviour_value=result.value,
                pre_squash=result.pre_squash,
            )
        )
        observed += 1
        if policy.buffer is not None and len(policy.buffer) > 0:
            stored = policy.buffer.rewards
            assert stored.dtype == np.float32, stored.dtype
            assert stored[-1] == np.float32(reward), (stored[-1], reward)
        stats = policy.update()
        if not learning or observed < policy.config.learning_starts:
            assert stats is None, f"update returned stats after {observed} observations"
        elif stats is not None:
            _check_stats(stats)
            stats_after_start += 1
        policy.end_period()
        obs = next_obs
    if learning:
        assert stats_after_start > 0, "update never returned UpdateStats"
    else:
        assert policy.exploration_scale() is None

    # Frozen policies neither store transitions nor change parameters.
    policy.set_frozen(True)
    assert policy.frozen
    before = _parameter_snapshot(policy)
    size_before = len(policy.buffer) if policy.buffer is not None else 0
    for _ in range(max(10, policy.config.batch_size)):
        result = policy.select_action(obs, explore=True)
        next_obs = random_obs()
        policy.observe(
            Transition(
                obs=obs,
                action=result.u,
                reward=1.5,
                next_obs=next_obs,
                behaviour_log_prob=result.log_prob,
                behaviour_value=result.value,
                pre_squash=result.pre_squash,
            )
        )
        assert policy.update() is None
        obs = next_obs
    size_after = len(policy.buffer) if policy.buffer is not None else 0
    assert size_after == size_before
    after = _parameter_snapshot(policy)
    assert before.keys() == after.keys()
    for key, tensor in before.items():
        assert torch.equal(tensor, after[key]), f"{key} changed while frozen"
    policy.set_frozen(False)
    assert not policy.frozen
