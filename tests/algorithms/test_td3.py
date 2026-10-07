"""Tests of TD3Algorithm (Fujimoto et al., 2018) on the rl_core contract."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from abmem.services.algorithms import TD3 as td3_module
from abmem.services.algorithms import registry
from abmem.services.algorithms.algorithm_config import TD3Config
from abmem.services.algorithms.base import PolicySpec, Transition
from abmem.services.algorithms.TD3 import TD3Algorithm
from tests.algorithms.conftest import make_spec
from tests.algorithms.contract import assert_policy_contract


def small_td3_config(**changes: object) -> TD3Config:
    """Return a small, fast TD3Config for tests."""
    base = TD3Config(
        batch_size=8, buffer_size=256, learning_starts=16, hidden_sizes=(16,)
    )
    return base.replace(**changes) if changes else base


def fill_buffer(policy: TD3Algorithm, n: int, seed: int = 1) -> None:
    """Store ``n`` random transitions in the policy buffer."""
    data_rng = np.random.default_rng(seed)
    spec = policy.spec
    for _ in range(n):
        policy.observe(
            Transition(
                obs=data_rng.uniform(-1, 1, spec.obs_dim).astype(np.float32),
                action=data_rng.uniform(-1, 1, spec.action_dim).astype(np.float32),
                reward=float(data_rng.normal()),
                next_obs=data_rng.uniform(-1, 1, spec.obs_dim).astype(np.float32),
                terminated=bool(data_rng.random() < 0.1),
            )
        )


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return clones of the state dict of ``module``."""
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    """True if any parameter of ``module`` differs from ``before``."""
    return any(not torch.equal(before[k], v) for k, v in module.state_dict().items())


# Contract ------------------------------------------------------------------------


def test_contract(spec: PolicySpec, rng: np.random.Generator) -> None:
    policy = TD3Algorithm(spec, small_td3_config(), rng)
    assert_policy_contract(policy, spec)


def test_contract_with_gradient_clipping(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = TD3Algorithm(spec, small_td3_config(max_grad_norm=1.0), rng)
    assert_policy_contract(policy, spec, n_steps=60)


def test_registry_resolves_td3() -> None:
    cls = registry.get_algorithm_class("TD3")
    assert cls is TD3Algorithm
    assert cls.config_class is TD3Config
    assert cls.is_learning and not cls.is_on_policy
    policy = registry.build_policy(
        "TD3", make_spec(action_dim=2), None, np.random.default_rng(0)
    )
    assert policy.config == TD3Config()


def test_wrong_config_type_rejected(rng: np.random.Generator) -> None:
    from abmem.services.algorithms.algorithm_config import DDPGConfig

    with pytest.raises(TypeError):
        TD3Algorithm(make_spec(), DDPGConfig(), rng)  # type: ignore[arg-type]


# Multi-plant output (A1) -----------------------------------------------------------


@pytest.mark.parametrize("action_dim", [1, 3, 5])
def test_multi_plant_action_shape(action_dim: int, rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=action_dim)
    policy = TD3Algorithm(spec, small_td3_config(), rng)
    obs = np.linspace(-1, 1, spec.obs_dim, dtype=np.float32)
    for explore in (False, True):
        result = policy.select_action(obs, explore=explore)
        assert result.u.shape == (action_dim,)
        assert result.u.dtype == np.float32
        assert np.all(np.abs(result.u) <= 1.0)
    batch_obs = torch.zeros(4, spec.obs_dim, device=policy.device)
    assert policy.target_actions(batch_obs).shape == (4, action_dim)


def test_greedy_action_has_no_noise(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=3)
    policy = TD3Algorithm(spec, small_td3_config(), rng)
    obs = np.full(spec.obs_dim, 0.3, dtype=np.float32)
    greedy = policy.select_action(obs, explore=False).u
    with torch.no_grad():
        mu = policy.actor(torch.as_tensor(obs, device=policy.device).unsqueeze(0))
    np.testing.assert_array_equal(greedy, mu.squeeze(0).cpu().numpy())


# Target policy smoothing (C5) ------------------------------------------------------


def test_target_actions_clamped_and_noise_clipped(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=3)
    config = small_td3_config(policy_noise=1.0, noise_clip=0.3)
    policy = TD3Algorithm(spec, config, rng)
    # Drive the target actor to about +1, -1 and 0 on the three plants.
    with torch.no_grad():
        last = policy.actor_target.net[-1]
        last.weight.zero_()
        last.bias.copy_(torch.tensor([5.0, -5.0, 0.0]))
    next_obs = torch.as_tensor(
        np.random.default_rng(3).uniform(-1, 1, (2000, spec.obs_dim)),
        dtype=torch.float32,
        device=policy.device,
    )
    with torch.no_grad():
        mu = policy.actor_target(next_obs)
    target = policy.target_actions(next_obs)

    assert target.shape == mu.shape
    assert torch.all(target >= -1.0) and torch.all(target <= 1.0)
    diff = (target - mu).abs()
    assert torch.all(diff <= config.noise_clip + 1e-6)
    # The clamp to the action bounds is active at the saturated plants.
    assert torch.any(target[:, 0] == 1.0)
    assert torch.any(target[:, 1] == -1.0)
    # The noise clip is active: with sigma = 1 and c = 0.3 most draws hit +-c.
    centred = target[:, 2] - mu[:, 2]
    assert torch.isclose(centred.max(), torch.tensor(0.3, device=policy.device))
    assert torch.isclose(centred.min(), torch.tensor(-0.3, device=policy.device))
    assert float((diff[:, 2] >= 0.3 - 1e-6).float().mean()) > 0.5


def test_zero_policy_noise_gives_actor_target_output(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = TD3Algorithm(spec, small_td3_config(policy_noise=0.0), rng)
    next_obs = torch.rand(16, spec.obs_dim, device=policy.device) * 2 - 1
    with torch.no_grad():
        mu = policy.actor_target(next_obs)
    torch.testing.assert_close(policy.target_actions(next_obs), mu)


# Clipped double Q-learning ----------------------------------------------------------


@pytest.mark.parametrize("shift", [10.0, -10.0, 0.0])
def test_td_target_uses_min_of_twin_critics(
    shift: float, rng: np.random.Generator
) -> None:
    spec = make_spec(action_dim=3)
    config = small_td3_config(policy_noise=0.0, gamma=0.9)
    policy = TD3Algorithm(spec, config, rng)
    with torch.no_grad():
        policy.critic2_target.net[-1].bias.add_(shift)
    gen = torch.Generator().manual_seed(5)
    rewards = torch.randn(32, 1, generator=gen).to(policy.device)
    next_obs = (torch.rand(32, spec.obs_dim, generator=gen) * 2 - 1).to(policy.device)
    terminated = (torch.rand(32, 1, generator=gen) < 0.3).float().to(policy.device)

    y = policy.td_target(rewards, next_obs, terminated)

    with torch.no_grad():
        a = policy.actor_target(next_obs)
        q1 = policy.critic1_target(next_obs, a)
        q2 = policy.critic2_target(next_obs, a)
    expected = rewards + 0.9 * (1.0 - terminated) * torch.min(q1, q2)
    torch.testing.assert_close(y, expected)
    if shift > 0:  # Q2' larger everywhere: target uses Q1'
        torch.testing.assert_close(y, rewards + 0.9 * (1.0 - terminated) * q1)
    elif shift < 0:  # Q2' smaller everywhere: target uses Q2'
        torch.testing.assert_close(y, rewards + 0.9 * (1.0 - terminated) * q2)
    assert not y.requires_grad


def test_td_target_terminal_equals_unscaled_reward(rng: np.random.Generator) -> None:
    """The reward enters the target unchanged (no fail-stack multiplier, C7)."""
    spec = make_spec(action_dim=2)
    policy = TD3Algorithm(spec, small_td3_config(), rng)
    rewards = torch.tensor([[-5.0], [0.0], [0.75]], device=policy.device)
    next_obs = torch.zeros(3, spec.obs_dim, device=policy.device)
    terminated = torch.ones(3, 1, device=policy.device)
    torch.testing.assert_close(policy.td_target(rewards, next_obs, terminated), rewards)


# Delayed policy updates -------------------------------------------------------------


def test_actor_and_targets_updated_only_every_policy_delay(
    rng: np.random.Generator,
) -> None:
    spec = make_spec(action_dim=3)
    config = small_td3_config(
        policy_delay=2, gradient_steps=1, learning_starts=8, batch_size=8
    )
    policy = TD3Algorithm(spec, config, rng)
    fill_buffer(policy, 32)
    for call in range(1, 7):
        actor_before = snapshot(policy.actor)
        critic_before = snapshot(policy.critic1)
        targets_before = {
            name: snapshot(getattr(policy, name))
            for name in ("actor_target", "critic1_target", "critic2_target")
        }
        stats = policy.update()
        assert stats is not None
        assert changed(critic_before, policy.critic1)
        delayed_step = call % 2 == 0
        assert changed(actor_before, policy.actor) == delayed_step
        for name, before in targets_before.items():
            assert changed(before, getattr(policy, name)) == delayed_step, name
        assert (stats.actor_loss is not None) == delayed_step
        assert stats.critic_loss is not None and np.isfinite(stats.critic_loss)
    assert policy.critic_updates == 6
    assert policy.actor_updates == 3


@pytest.mark.parametrize(("gradient_steps", "policy_delay"), [(4, 2), (6, 3), (3, 1)])
def test_update_counts_per_call(
    gradient_steps: int, policy_delay: int, rng: np.random.Generator
) -> None:
    spec = make_spec(action_dim=2)
    config = small_td3_config(gradient_steps=gradient_steps, policy_delay=policy_delay)
    policy = TD3Algorithm(spec, config, rng)
    fill_buffer(policy, 20)
    stats = policy.update()
    assert stats is not None
    assert stats.extra["critic_updates"] == gradient_steps
    assert stats.extra["actor_updates"] == gradient_steps // policy_delay


def test_no_update_before_learning_starts(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = TD3Algorithm(spec, small_td3_config(learning_starts=16), rng)
    fill_buffer(policy, 15)
    assert policy.update() is None
    assert policy.critic_updates == 0
    fill_buffer(policy, 1)
    assert policy.update() is not None


# Determinism and legacy removal -----------------------------------------------------


def _run_episode(seed: int) -> np.ndarray:
    torch.manual_seed(0)
    spec = make_spec(action_dim=3)
    policy = TD3Algorithm(spec, small_td3_config(), np.random.default_rng(seed))
    data_rng = np.random.default_rng(99)
    obs = data_rng.uniform(-1, 1, spec.obs_dim).astype(np.float32)
    actions = []
    for _ in range(40):
        u = policy.select_action(obs, explore=True).u
        actions.append(u)
        next_obs = data_rng.uniform(-1, 1, spec.obs_dim).astype(np.float32)
        policy.observe(Transition(obs, u, float(data_rng.normal()), next_obs))
        policy.update()
        policy.end_period()
        obs = next_obs
    actions.append(policy.select_action(obs, explore=False).u)
    return np.stack(actions)


def test_deterministic_under_seed() -> None:
    np.testing.assert_array_equal(_run_episode(7), _run_episode(7))
    assert not np.array_equal(_run_episode(7), _run_episode(8))


def test_legacy_mechanisms_removed() -> None:
    for name in ("update_failstack", "get_failstack_adjustment", "failstack"):
        assert not hasattr(TD3Algorithm, name)
    source = inspect.getsource(td3_module)
    for forbidden in (
        "actor_critic",
        "agent_algorithm",
        "algorithm_utils",
        "failstack =",
        'torch.device("cpu")',
    ):
        assert forbidden not in source, forbidden
    assert not hasattr(td3_module, "ReplayBuffer") or (
        td3_module.ReplayBuffer.__module__ == "abmem.services.algorithms.replay_buffer"
    )
