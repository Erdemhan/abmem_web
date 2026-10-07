"""Tests of DDPGAlgorithm (Lillicrap et al., 2016, Algorithm 1)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from abmem.models.enums import AlgorithmType
from abmem.services.algorithms.algorithm_config import DDPGConfig
from abmem.services.algorithms.base import PolicySpec, Transition, UpdateStats
from abmem.services.algorithms.DDPG import DDPGAlgorithm
from abmem.services.algorithms.registry import build_policy, get_algorithm_class
from abmem.services.algorithms.replay_buffer import Batch
from tests.algorithms.conftest import make_spec, small_ddpg_config
from tests.algorithms.contract import assert_policy_contract


def _make_batch(
    spec: PolicySpec,
    batch_size: int = 8,
    reward: float = 0.5,
    terminated: float = 0.0,
    seed: int = 1,
) -> Batch:
    """Return a fixed synthetic batch on the policy device."""
    gen = torch.Generator().manual_seed(seed)
    device = torch.device("cpu")

    def uniform(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=gen) * 2.0 - 1.0

    batch = Batch(
        obs=uniform(batch_size, spec.obs_dim),
        actions=uniform(batch_size, spec.action_dim),
        rewards=torch.full((batch_size, 1), reward),
        next_obs=uniform(batch_size, spec.obs_dim),
        terminated=torch.full((batch_size, 1), terminated),
        intervened=torch.zeros(batch_size, 1),
    )
    return Batch(**{k: v.to(device) for k, v in vars(batch).items()})


def _to_device(batch: Batch, device: torch.device) -> Batch:
    """Move every tensor of ``batch`` to ``device``."""
    return Batch(**{k: v.to(device) for k, v in vars(batch).items()})


def _forbid_noise(policy: DDPGAlgorithm, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the policy generator by one whose Gaussian draws fail.

    Generator methods are read-only, so the attribute ``policy.rng`` (used for
    exploration noise) is swapped. The replay buffer keeps its own reference
    to the original generator for minibatch sampling.
    """

    class NoNoiseGenerator:
        def normal(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("exploration noise drawn")

    monkeypatch.setattr(policy, "rng", NoNoiseGenerator())


# Contract -------------------------------------------------------------------


def test_contract_small_config(spec: PolicySpec, rng: np.random.Generator) -> None:
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    assert_policy_contract(policy, spec)


def test_contract_default_config(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, DDPGConfig(), rng)
    assert policy.config.learning_starts == 64
    assert policy.config.gradient_steps == 1
    assert_policy_contract(policy, spec, n_steps=100)


def test_class_attributes_and_registry() -> None:
    assert DDPGAlgorithm.algorithm_type is AlgorithmType.DDPG
    assert DDPGAlgorithm.is_learning is True
    assert DDPGAlgorithm.is_on_policy is False
    assert DDPGAlgorithm.config_class is DDPGConfig
    assert get_algorithm_class(AlgorithmType.DDPG) is DDPGAlgorithm
    policy = build_policy(
        AlgorithmType.DDPG,
        make_spec(),
        {"hidden_sizes": (8,)},
        np.random.default_rng(0),
    )
    assert isinstance(policy, DDPGAlgorithm)


def test_rejects_wrong_config_type(rng: np.random.Generator) -> None:
    from abmem.services.algorithms.algorithm_config import TD3Config

    with pytest.raises(TypeError):
        DDPGAlgorithm(make_spec(), TD3Config(), rng)  # type: ignore[arg-type]


# Networks and exploration ---------------------------------------------------


def test_targets_start_as_copies_without_gradients(rng: np.random.Generator) -> None:
    policy = DDPGAlgorithm(make_spec(action_dim=3), small_ddpg_config(), rng)
    for online, target in (
        (policy.actor, policy.actor_target),
        (policy.critic, policy.critic_target),
    ):
        assert online is not target
        for p_online, p_target in zip(
            online.parameters(), target.parameters(), strict=True
        ):
            assert torch.equal(p_online, p_target)
            assert p_online.data_ptr() != p_target.data_ptr()
            assert not p_target.requires_grad


def test_no_noise_when_explore_false(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(action_dim=3)
    policy = DDPGAlgorithm(spec, small_ddpg_config(exploration_sigma=0.5), rng)
    obs = np.linspace(-1.0, 1.0, spec.obs_dim, dtype=np.float32)
    _forbid_noise(policy, monkeypatch)
    result = policy.select_action(obs, explore=False)
    with torch.no_grad():
        expected = policy.actor(policy._as_tensor(obs).unsqueeze(0)).squeeze(0)
    np.testing.assert_array_equal(result.u, expected.cpu().numpy().astype(np.float32))


def test_explore_adds_clipped_gaussian_noise(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=4)
    policy = DDPGAlgorithm(spec, small_ddpg_config(exploration_sigma=0.3), rng)
    obs = np.zeros(spec.obs_dim, dtype=np.float32)
    greedy = policy.select_action(obs, explore=False).u
    samples = np.stack([policy.select_action(obs, explore=True).u for _ in range(2000)])
    assert np.all(samples >= -1.0) and np.all(samples <= 1.0)
    # The greedy action is close to 0 (final layer init U(-3e-3, 3e-3)), so
    # clipping at +-1 is negligible for sigma = 0.3.
    np.testing.assert_allclose(samples.mean(axis=0), greedy, atol=0.03)
    np.testing.assert_allclose(samples.std(axis=0), 0.3, atol=0.03)


def test_zero_sigma_explore_equals_greedy(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(exploration_sigma=0.0), rng)
    obs = np.full(spec.obs_dim, 0.3, dtype=np.float32)
    np.testing.assert_array_equal(
        policy.select_action(obs, explore=True).u,
        policy.select_action(obs, explore=False).u,
    )


def test_sigma_decays_once_per_period(rng: np.random.Generator) -> None:
    config = small_ddpg_config(
        exploration_sigma=0.4, exploration_sigma_decay=0.5, exploration_sigma_min=0.15
    )
    policy = DDPGAlgorithm(make_spec(), config, rng)
    assert policy.exploration_scale() == pytest.approx(0.4)
    policy.end_period()
    assert policy.exploration_scale() == pytest.approx(0.2)
    policy.end_period()
    assert policy.exploration_scale() == pytest.approx(0.15)


# Update rule ----------------------------------------------------------------


def test_target_uses_noise_free_target_networks(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(gamma=0.9), rng)
    # Make the target networks differ from the online networks.
    with torch.no_grad():
        for p in policy.actor_target.parameters():
            p.add_(0.1)
        for p in policy.critic_target.parameters():
            p.add_(0.05)
    batch = _to_device(_make_batch(spec, reward=0.7), policy.device)
    batch = Batch(
        **{
            **vars(batch),
            "terminated": torch.tensor([[0.0], [1.0]] * 4, device=policy.device),
        }
    )
    _forbid_noise(policy, monkeypatch)
    y = policy.compute_target(batch)
    with torch.no_grad():
        next_q = policy.critic_target(
            batch.next_obs, policy.actor_target(batch.next_obs)
        )
        expected = batch.rewards + 0.9 * (1.0 - batch.terminated) * next_q
    assert not y.requires_grad
    torch.testing.assert_close(y, expected)
    # Terminal rows reduce to the reward.
    torch.testing.assert_close(y[1::2], batch.rewards[1::2])
    # The online actor would give a different target.
    with torch.no_grad():
        online = batch.rewards + 0.9 * (1.0 - batch.terminated) * policy.critic_target(
            batch.next_obs, policy.actor(batch.next_obs)
        )
    assert not torch.allclose(online, expected)


def test_update_draws_no_exploration_noise(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    data = np.random.default_rng(3)
    for _ in range(policy.config.learning_starts):
        policy.observe(
            Transition(
                obs=data.uniform(-1, 1, spec.obs_dim),
                action=data.uniform(-1, 1, spec.action_dim),
                reward=float(data.normal()),
                next_obs=data.uniform(-1, 1, spec.obs_dim),
            )
        )
    sample_calls = 0
    original_sample = policy.buffer.sample

    def counting_sample(batch_size: int, device: torch.device) -> Batch:
        nonlocal sample_calls
        sample_calls += 1
        return original_sample(batch_size, device)

    monkeypatch.setattr(policy.buffer, "sample", counting_sample)
    _forbid_noise(policy, monkeypatch)
    stats = policy.update()
    assert isinstance(stats, UpdateStats)
    assert stats.actor_loss is not None and stats.critic_loss is not None
    assert sample_calls == policy.config.gradient_steps


def test_gradient_steps_config_controls_number_of_steps(
    rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec()
    policy = DDPGAlgorithm(spec, small_ddpg_config(gradient_steps=3), rng)
    data = np.random.default_rng(4)
    for _ in range(policy.config.learning_starts - 1):
        policy.observe(
            Transition(
                obs=data.uniform(-1, 1, spec.obs_dim),
                action=data.uniform(-1, 1, spec.action_dim),
                reward=0.0,
                next_obs=data.uniform(-1, 1, spec.obs_dim),
            )
        )
    calls = 0
    original = policy._gradient_step

    def counting(batch: Batch) -> tuple[float, float, float]:
        nonlocal calls
        calls += 1
        return original(batch)

    monkeypatch.setattr(policy, "_gradient_step", counting)
    assert policy.update() is None
    assert calls == 0
    policy.observe(
        Transition(
            obs=np.zeros(spec.obs_dim),
            action=np.zeros(spec.action_dim),
            reward=0.0,
            next_obs=np.zeros(spec.obs_dim),
        )
    )
    assert policy.update() is not None
    assert calls == 3


def test_soft_target_update_math(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    tau = 0.1
    policy = DDPGAlgorithm(spec, small_ddpg_config(tau=tau), rng)
    with torch.no_grad():
        for p in policy.actor_target.parameters():
            p.add_(0.2)
        for p in policy.critic_target.parameters():
            p.sub_(0.2)
    old_targets = {
        "actor": [p.detach().clone() for p in policy.actor_target.parameters()],
        "critic": [p.detach().clone() for p in policy.critic_target.parameters()],
    }
    batch = _to_device(_make_batch(spec), policy.device)
    policy._gradient_step(batch)
    for name, online, target in (
        ("actor", policy.actor, policy.actor_target),
        ("critic", policy.critic, policy.critic_target),
    ):
        for old, p_online, p_target in zip(
            old_targets[name], online.parameters(), target.parameters(), strict=True
        ):
            expected = tau * p_online.detach() + (1.0 - tau) * old
            torch.testing.assert_close(p_target.detach(), expected)


def test_actor_and_critic_gradients_non_zero(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=3)
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    batch = _to_device(_make_batch(spec, reward=1.0), policy.device)
    policy._gradient_step(batch)
    for module in (policy.actor, policy.critic):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads)
        total = sum(float(g.abs().sum()) for g in grads if g is not None)
        assert total > 0.0
        assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_critic_loss_decreases_on_fixed_batch(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(lr_critic=1e-3), rng)
    batch = _to_device(
        _make_batch(spec, batch_size=8, reward=0.8, terminated=1.0), policy.device
    )
    losses = [policy._gradient_step(batch)[0] for _ in range(200)]
    assert all(np.isfinite(losses))
    assert losses[-1] < 0.1 * losses[0]


@pytest.mark.parametrize("best", [-0.5, 0.6])
def test_learns_better_action_on_bandit(best: float) -> None:
    """On a one-step bandit with reward -(u - best)^2, mu(o) moves toward best."""
    torch.manual_seed(0)
    spec = make_spec(action_dim=1, obs_dim=2)
    config = small_ddpg_config(
        batch_size=32,
        buffer_size=1024,
        learning_starts=32,
        hidden_sizes=(32, 32),
        lr_actor=1e-3,
        lr_critic=3e-3,
        exploration_sigma=0.5,
    )
    policy = DDPGAlgorithm(spec, config, np.random.default_rng(1))
    obs = np.array([0.2, -0.4], dtype=np.float32)
    start = float(policy.select_action(obs, explore=False).u[0])
    for _ in range(600):
        u = policy.select_action(obs, explore=True).u
        reward = -float((u[0] - best) ** 2)
        policy.observe(
            Transition(obs=obs, action=u, reward=reward, next_obs=obs, terminated=True)
        )
        policy.update()
    final = float(policy.select_action(obs, explore=False).u[0])
    assert abs(final - best) < 0.15
    assert abs(final - best) < abs(start - best)
    # The critic ranks the better action above a worse one.
    with torch.no_grad():
        o = policy._as_tensor(obs).unsqueeze(0)
        q_best = policy.critic(o, torch.tensor([[best]], device=policy.device))
        q_far = policy.critic(o, torch.tensor([[-best * 1.5]], device=policy.device))
    assert float(q_best) > float(q_far)


def test_rewards_stored_unchanged(rng: np.random.Generator) -> None:
    spec = make_spec()
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    rewards = [-4.987654, 0.0, 0.333333, 1.0]
    for r in rewards:
        policy.observe(
            Transition(
                obs=np.zeros(spec.obs_dim),
                action=np.zeros(spec.action_dim),
                reward=r,
                next_obs=np.zeros(spec.obs_dim),
            )
        )
    np.testing.assert_array_equal(
        policy.buffer.rewards, np.asarray(rewards, dtype=np.float32)
    )


def test_constant_observation_stream_stays_finite(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=5)
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    obs = np.full(spec.obs_dim, 0.5, dtype=np.float32)
    for _ in range(64):
        u = policy.select_action(obs, explore=True).u
        policy.observe(Transition(obs=obs, action=u, reward=0.0, next_obs=obs))
        stats = policy.update()
        if stats is not None:
            assert np.isfinite(stats.critic_loss)
            assert np.isfinite(stats.actor_loss)
    assert np.all(np.isfinite(policy.select_action(obs, explore=False).u))


def test_max_grad_norm_clips_gradients(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(max_grad_norm=1e-3), rng)
    batch = _to_device(_make_batch(spec, reward=100.0, terminated=1.0), policy.device)
    policy._gradient_step(batch)
    for module in (policy.actor, policy.critic):
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad) for p in module.parameters()])
        )
        assert float(norm) <= 1e-3 * (1.0 + 1e-4)


def _run_sequence(seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    spec = make_spec(action_dim=3)
    policy = DDPGAlgorithm(spec, small_ddpg_config(), np.random.default_rng(seed))
    data = np.random.default_rng(99)
    obs = data.uniform(-1, 1, spec.obs_dim).astype(np.float32)
    actions = []
    for _ in range(60):
        u = policy.select_action(obs, explore=True).u
        actions.append(u)
        next_obs = data.uniform(-1, 1, spec.obs_dim).astype(np.float32)
        policy.observe(
            Transition(
                obs=obs, action=u, reward=float(data.normal()), next_obs=next_obs
            )
        )
        policy.update()
        policy.end_period()
        obs = next_obs
    return np.stack(actions)


def test_deterministic_under_seed() -> None:
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        first = _run_sequence(7)
        second = _run_sequence(7)
    finally:
        torch.use_deterministic_algorithms(False)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, _run_sequence(8))


def test_mse_loss_is_against_buffer_actions(rng: np.random.Generator) -> None:
    """The critic loss evaluates Q at the stored normalised actions (C5)."""
    spec = make_spec(action_dim=2)
    policy = DDPGAlgorithm(spec, small_ddpg_config(), rng)
    batch = _to_device(_make_batch(spec, reward=0.3, terminated=1.0), policy.device)
    with torch.no_grad():
        expected = nn.functional.mse_loss(
            policy.critic(batch.obs, batch.actions), batch.rewards
        )
    critic_loss, _, mean_q = policy._gradient_step(batch)
    assert critic_loss == pytest.approx(float(expected), rel=1e-5)
    assert np.isfinite(mean_q)
