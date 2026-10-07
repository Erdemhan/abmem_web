"""Tests of the RDPG port (Heess et al., 2015) on BaseAgentAlgorithm (B3, B7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from abmem.services.algorithms.algorithm_config import RDPGConfig
from abmem.services.algorithms.base import PolicySpec, Transition, UpdateStats
from abmem.services.algorithms.RDPG import RDPGAlgorithm, substituted_action_q
from abmem.services.algorithms.replay_buffer import SequenceBatch
from tests.algorithms.conftest import make_spec
from tests.algorithms.contract import assert_policy_contract


def small_config(**changes: object) -> RDPGConfig:
    """Return a small, fast RDPGConfig for tests."""
    base = RDPGConfig(
        batch_size=8,
        buffer_size=256,
        learning_starts=16,
        hidden_sizes=(16,),
        seq_len=4,
        lstm_hidden=8,
    )
    return base.replace(**changes) if changes else base


def make_policy(
    spec: PolicySpec | None = None, seed: int = 0, **changes: object
) -> RDPGAlgorithm:
    """Return an RDPG policy with the small config."""
    return RDPGAlgorithm(
        spec or make_spec(action_dim=2),
        small_config(**changes),
        np.random.default_rng(seed),
    )


def act_periods(policy: RDPGAlgorithm, observations: list[np.ndarray]) -> list:
    """Run one select_action and end_period per observation, return the actions."""
    actions = []
    for obs in observations:
        actions.append(policy.select_action(obs, explore=False).u)
        policy.end_period()
    return actions


def fill_buffer(policy: RDPGAlgorithm, n: int, seed: int = 1) -> None:
    """Store ``n`` consecutive random transitions."""
    data = np.random.default_rng(seed)
    obs = data.uniform(-1, 1, policy.spec.obs_dim).astype(np.float32)
    for _ in range(n):
        next_obs = data.uniform(-1, 1, policy.spec.obs_dim).astype(np.float32)
        action = data.uniform(-1, 1, policy.spec.action_dim).astype(np.float32)
        policy.observe(Transition(obs, action, float(data.normal()), next_obs))
        obs = next_obs


# Contract -----------------------------------------------------------------------


def test_contract(spec: PolicySpec, rng: np.random.Generator) -> None:
    policy = RDPGAlgorithm(spec, small_config(), rng)
    assert_policy_contract(policy, spec)


def test_contract_buffer_exactly_seq_len() -> None:
    spec = make_spec(action_dim=3)
    policy = RDPGAlgorithm(
        spec,
        small_config(seq_len=16, learning_starts=16),
        np.random.default_rng(0),
    )
    assert_policy_contract(policy, spec, n_steps=40)


def test_no_actor_critic_import() -> None:
    import abmem.services.algorithms.RDPG as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "actor_critic" not in source
    assert "agent_algorithm" not in source


# Acting history -----------------------------------------------------------------


def test_history_holds_last_seq_len_observations() -> None:
    policy = make_policy(seq_len=4)
    observations = [np.full(3, i / 10, dtype=np.float32) for i in range(7)]
    act_periods(policy, observations[:2])
    np.testing.assert_array_equal(policy.history, np.stack(observations[:2]))
    act_periods(policy, observations[2:])
    assert policy.history.shape == (4, 3)
    np.testing.assert_array_equal(policy.history, np.stack(observations[-4:]))


def test_repeated_query_within_period_replaces_observation() -> None:
    policy = make_policy(seq_len=4)
    a, b = np.zeros(3, np.float32), np.ones(3, np.float32) * 0.5
    act_periods(policy, [a])
    policy.select_action(b, explore=True)
    policy.select_action(b, explore=False)
    np.testing.assert_array_equal(policy.history, np.stack([a, b]))
    policy.end_period()
    np.testing.assert_array_equal(policy.history, np.stack([a, b]))


def test_history_not_appended_by_observe_or_update() -> None:
    policy = make_policy(seq_len=4)
    obs = np.full(3, 0.2, np.float32)
    policy.select_action(obs, explore=True)
    fill_buffer(policy, 20)
    policy.update()
    np.testing.assert_array_equal(policy.history, obs[None, :])


def test_history_kept_while_frozen() -> None:
    policy = make_policy(seq_len=3)
    policy.set_frozen(True)
    observations = [np.full(3, i / 10, dtype=np.float32) for i in range(4)]
    act_periods(policy, observations)
    np.testing.assert_array_equal(policy.history, np.stack(observations[-3:]))


def test_action_is_actor_output_on_window_from_zero_state() -> None:
    policy = make_policy(seq_len=3)
    data = np.random.default_rng(3)
    observations = [data.uniform(-1, 1, 3).astype(np.float32) for _ in range(5)]
    actions = act_periods(policy, observations)
    window = torch.as_tensor(np.stack(observations[-3:])).unsqueeze(0)
    with torch.no_grad():
        expected, _ = policy.actor(window, None)
    np.testing.assert_allclose(actions[-1], expected[0, -1].numpy(), atol=1e-6)
    # A shorter window at the start is processed without padding.
    with torch.no_grad():
        first, _ = policy.actor(torch.as_tensor(observations[0]).view(1, 1, 3), None)
    np.testing.assert_allclose(actions[0], first[0, -1].numpy(), atol=1e-6)


def test_greedy_acting_deterministic_for_same_history() -> None:
    data = np.random.default_rng(4)
    observations = [data.uniform(-1, 1, 3).astype(np.float32) for _ in range(6)]
    first = act_periods(make_policy(seed=0), observations)
    torch.manual_seed(0)
    second = act_periods(make_policy(seed=0), observations)
    for u1, u2 in zip(first, second, strict=True):
        np.testing.assert_array_equal(u1, u2)


def test_exploration_noise_only_when_exploring() -> None:
    policy = make_policy(exploration_sigma=0.3)
    obs = np.zeros(3, np.float32)
    greedy = policy.select_action(obs, explore=False).u
    noisy = policy.select_action(obs, explore=True).u
    assert not np.allclose(greedy, noisy)
    assert np.all(np.abs(noisy) <= 1.0)
    np.testing.assert_array_equal(policy.select_action(obs, explore=False).u, greedy)


# Critic and training ------------------------------------------------------------


def test_critic_uses_per_step_actions() -> None:
    policy = make_policy(seq_len=5)
    obs = torch.randn(2, 5, 3)
    actions = torch.rand(2, 5, 2) * 2 - 1
    changed = actions.clone()
    k = 2
    changed[:, k] = -changed[:, k]
    with torch.no_grad():
        q = policy.critic(obs, actions, None)
        q_changed = policy.critic(obs, changed, None)
    torch.testing.assert_close(q[:, :k], q_changed[:, :k])
    assert not torch.allclose(q[:, k:], q_changed[:, k:])


def test_substituted_q_uses_taken_history_and_query_at_step() -> None:
    policy = make_policy(seq_len=4)
    obs = torch.randn(3, 4, 3)
    taken = torch.rand(3, 4, 2) * 2 - 1
    query = torch.rand(3, 4, 2) * 2 - 1
    with torch.no_grad():
        q = substituted_action_q(policy.critic, obs, taken, query)
        for k in range(4):
            mixed = taken.clone()
            mixed[:, k] = query[:, k]
            reference = policy.critic(obs[:, : k + 1], mixed[:, : k + 1], None)
            torch.testing.assert_close(q[:, k], reference[:, -1])
        # Equal query and taken actions reproduce the ordinary critic pass.
        torch.testing.assert_close(
            substituted_action_q(policy.critic, obs, taken, taken),
            policy.critic(obs, taken, None),
        )


def test_substituted_q_rejects_bad_shapes() -> None:
    policy = make_policy()
    with pytest.raises(ValueError):
        substituted_action_q(
            policy.critic,
            torch.zeros(1, 4, 3),
            torch.zeros(1, 4, 2),
            torch.zeros(1, 3, 2),
        )


def test_critic_targets_hand_computation() -> None:
    policy = make_policy(seq_len=3, gamma=0.9)
    fill_buffer(policy, 20)
    batch = policy.buffer.sample_sequences(4, 3, policy.device)
    targets = policy._critic_targets(batch)
    assert targets.shape == (4, 3, 1)
    obs_ext = torch.cat([batch.obs, batch.next_obs[:, -1:]], dim=1)
    with torch.no_grad():
        mu_target, _ = policy.actor_target(obs_ext, None)
        for k in range(3):
            # h_{k+1}: taken actions up to step k, mu' at step k + 1.
            acts = torch.cat(
                [batch.actions[:, : k + 1], mu_target[:, k + 1 : k + 2]], 1
            )
            q_next = policy.critic_target(obs_ext[:, : k + 2], acts, None)[:, -1]
            expected = batch.rewards[:, k] + 0.9 * q_next
            torch.testing.assert_close(targets[:, k], expected)


def test_terminated_steps_drop_bootstrap() -> None:
    policy = make_policy(seq_len=2)
    batch = SequenceBatch(
        obs=torch.randn(1, 2, 3),
        actions=torch.zeros(1, 2, 2),
        rewards=torch.tensor([[[1.0], [2.0]]]),
        next_obs=torch.randn(1, 2, 3),
        terminated=torch.ones(1, 2, 1),
        intervened=torch.zeros(1, 2, 1),
    )
    torch.testing.assert_close(policy._critic_targets(batch), batch.rewards)


def test_sequence_sampling_shapes_and_contiguity() -> None:
    policy = make_policy(spec=make_spec(action_dim=3), seq_len=5)
    fill_buffer(policy, 30)
    batch = policy.buffer.sample_sequences(8, 5, policy.device)
    assert batch.obs.shape == (8, 5, 3)
    assert batch.actions.shape == (8, 5, 3)
    assert batch.rewards.shape == (8, 5, 1)
    assert batch.obs.dtype == torch.float32
    torch.testing.assert_close(batch.obs[:, 1:], batch.next_obs[:, :-1])


def test_no_learning_before_enough_transitions_and_window() -> None:
    policy = make_policy(seq_len=20, learning_starts=8, batch_size=8)
    fill_buffer(policy, 19)
    assert len(policy.buffer) >= policy.config.learning_starts
    assert not policy.ready_to_learn
    assert policy.update() is None
    fill_buffer(policy, 1, seed=5)
    assert policy.ready_to_learn
    assert isinstance(policy.update(), UpdateStats)


def test_no_learning_before_learning_starts() -> None:
    policy = make_policy(seq_len=4, learning_starts=16)
    before = {k: v.clone() for k, v in policy.critic.state_dict().items()}
    fill_buffer(policy, 15)
    assert policy.update() is None
    for key, value in policy.critic.state_dict().items():
        assert torch.equal(value, before[key])


def test_update_runs_gradient_steps_and_moves_targets() -> None:
    policy = make_policy(gradient_steps=3, tau=0.5)
    fill_buffer(policy, 30)
    actor_before = [p.clone() for p in policy.actor.parameters()]
    target_before = [p.clone() for p in policy.critic_target.parameters()]
    calls = []
    original = policy.buffer.sample_sequences

    def counting(*args: object, **kwargs: object) -> SequenceBatch:
        calls.append(args)
        return original(*args, **kwargs)

    policy.buffer.sample_sequences = counting  # type: ignore[method-assign]
    stats = policy.update()
    assert isinstance(stats, UpdateStats)
    assert stats.actor_loss is not None and np.isfinite(stats.actor_loss)
    assert len(calls) == 3
    assert all(c[1] == policy.config.seq_len for c in calls)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(actor_before, policy.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(target_before, policy.critic_target.parameters(), strict=True)
    )


def test_critic_loss_decreases_on_constant_reward() -> None:
    policy = make_policy(seq_len=4, lr_critic=1e-2, gamma=0.5)
    data = np.random.default_rng(7)
    obs = np.zeros(3, np.float32)
    for _ in range(40):
        action = data.uniform(-1, 1, 2).astype(np.float32)
        policy.observe(Transition(obs, action, 0.5, obs))
    first = policy.update()
    for _ in range(100):
        last = policy.update()
    assert first is not None and last is not None
    assert last.critic_loss < first.critic_loss


def test_constant_observations_and_zero_reward_finite() -> None:
    policy = make_policy(seq_len=4)
    obs = np.full(3, 0.3, np.float32)
    for _ in range(30):
        u = policy.select_action(obs, explore=True).u
        policy.observe(Transition(obs, u, 0.0, obs))
        stats = policy.update()
        policy.end_period()
        if stats is not None:
            assert np.isfinite(stats.critic_loss)
    assert policy.history.shape == (4, 3)
