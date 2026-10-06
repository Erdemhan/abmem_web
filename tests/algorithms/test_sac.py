"""Tests of the SAC algorithm (Haarnoja et al., 2018a, b) in normalised spaces."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abmem.services.algorithms.algorithm_config import SACConfig
from abmem.services.algorithms.base import PolicySpec, Transition
from abmem.services.algorithms.SAC import SACAlgorithm
from abmem.services.algorithms.sac_networks import (
    SquashedGaussianActor,
    squashed_gaussian_log_prob,
    tanh_log_abs_det_jacobian,
)

from .conftest import make_spec
from .contract import assert_policy_contract


def small_sac_config(**changes: object) -> SACConfig:
    """Return a small, fast SACConfig for tests."""
    base = SACConfig(
        batch_size=8, buffer_size=256, learning_starts=16, hidden_sizes=(16, 16)
    )
    return base.replace(**changes) if changes else base


def fill_buffer(policy: SACAlgorithm, n: int, seed: int = 1) -> None:
    """Store ``n`` random transitions in the policy buffer."""
    data = np.random.default_rng(seed)
    dims = (policy.spec.obs_dim, policy.spec.action_dim)
    for _ in range(n):
        policy.observe(
            Transition(
                obs=data.uniform(-1, 1, dims[0]),
                action=data.uniform(-1, 1, dims[1]),
                reward=float(data.normal()),
                next_obs=data.uniform(-1, 1, dims[0]),
            )
        )


def test_contract(spec: PolicySpec, rng: np.random.Generator) -> None:
    policy = SACAlgorithm(spec, small_sac_config(), rng)
    assert_policy_contract(policy, spec)


def test_contract_with_gradient_clipping_and_steps(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=3)
    policy = SACAlgorithm(
        spec, small_sac_config(max_grad_norm=1.0, gradient_steps=3), rng
    )
    assert_policy_contract(policy, spec, n_steps=60)


def test_log_prob_matches_numerical_density_1d() -> None:
    mean = torch.tensor([[0.4]], dtype=torch.float64)
    log_std = torch.tensor([[math.log(0.7)]], dtype=torch.float64)
    u = torch.linspace(-0.999, 0.999, 4001, dtype=torch.float64).unsqueeze(-1)
    z = torch.atanh(u)
    log_prob = squashed_gaussian_log_prob(z, mean, log_std).squeeze(-1)
    # Change of variables: p_u(u) = N(atanh(u); mean, sigma) / (1 - u^2).
    sigma = math.exp(log_std.item())
    gauss = torch.exp(-0.5 * ((z - mean) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi)
    )
    reference = (gauss / (1.0 - u**2)).squeeze(-1)
    torch.testing.assert_close(torch.exp(log_prob), reference, rtol=1e-9, atol=1e-12)
    # The density integrates to (almost) one over the open interval.
    mass = torch.trapezoid(torch.exp(log_prob), u.squeeze(-1)).item()
    assert mass == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("z_value", [-40.0, -12.0, 12.0, 40.0])
def test_log_prob_finite_near_bounds(z_value: float) -> None:
    z = torch.full((2, 3), z_value, dtype=torch.float32)
    mean = torch.full_like(z, z_value)
    log_std = torch.zeros_like(z)
    log_prob = squashed_gaussian_log_prob(z, mean, log_std)
    assert torch.all(torch.isfinite(log_prob))
    jac = tanh_log_abs_det_jacobian(z)
    expected = 2.0 * (math.log(2.0) - abs(z_value))  # asymptotic value
    torch.testing.assert_close(jac, torch.full_like(z, expected), rtol=1e-5, atol=1e-4)


def test_actor_sample_finite_with_saturated_mean() -> None:
    actor = SquashedGaussianActor(2, 2, (8,))
    with torch.no_grad():
        actor.mean_head.bias.fill_(50.0)
    u, log_prob = actor.sample(torch.zeros(4, 2))
    assert torch.all(u.abs() <= 1.0)
    assert torch.all(torch.isfinite(log_prob))


def test_deterministic_action_is_tanh_of_mean(rng: np.random.Generator) -> None:
    spec = make_spec(action_dim=2)
    policy = SACAlgorithm(spec, small_sac_config(), rng)
    obs = np.array([0.1, -0.5, 0.9], dtype=np.float32)
    first = policy.select_action(obs, explore=False).u
    second = policy.select_action(obs, explore=False).u
    np.testing.assert_array_equal(first, second)
    with torch.no_grad():
        mean, _ = policy.actor(policy._as_tensor(obs).unsqueeze(0))
    np.testing.assert_allclose(first, torch.tanh(mean).squeeze(0).cpu().numpy(), rtol=1e-6)
    sampled = [policy.select_action(obs, explore=True).u for _ in range(5)]
    assert any(not np.array_equal(s, first) for s in sampled)


def test_same_seed_gives_identical_action_sequences() -> None:
    spec = make_spec(action_dim=2)
    obs = np.array([0.3, 0.2, -0.1], dtype=np.float32)
    sequences = []
    for _ in range(2):
        torch.manual_seed(7)
        policy = SACAlgorithm(spec, small_sac_config(), np.random.default_rng(3))
        sequences.append([policy.select_action(obs, True).u for _ in range(5)])
    for a, b in zip(*sequences, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("action_dim", [1, 4])
def test_target_entropy_default_and_override(
    action_dim: int, rng: np.random.Generator
) -> None:
    spec = make_spec(action_dim=action_dim)
    assert SACAlgorithm(spec, small_sac_config(), rng).target_entropy == -action_dim
    custom = SACAlgorithm(spec, small_sac_config(target_entropy=-0.5), rng)
    assert custom.target_entropy == -0.5


def test_initial_alpha_and_exploration_scale(rng: np.random.Generator) -> None:
    policy = SACAlgorithm(make_spec(), small_sac_config(initial_alpha=0.3), rng)
    assert policy.exploration_scale() == pytest.approx(0.3, rel=1e-6)


@pytest.mark.parametrize(
    ("target_entropy", "increases"), [(50.0, True), (-50.0, False)]
)
def test_alpha_moves_toward_target_entropy(
    target_entropy: float, increases: bool, rng: np.random.Generator
) -> None:
    policy = SACAlgorithm(
        make_spec(), small_sac_config(target_entropy=target_entropy), rng
    )
    fill_buffer(policy, 16)
    before = policy.alpha
    stats = policy.update()
    assert stats is not None
    # Entropy below the target raises alpha, entropy above the target lowers it.
    assert (policy.alpha > before) is increases
    assert stats.extra["alpha"] == pytest.approx(policy.alpha)


def test_no_update_before_learning_starts(rng: np.random.Generator) -> None:
    policy = SACAlgorithm(make_spec(), small_sac_config(learning_starts=20), rng)
    fill_buffer(policy, 19)
    assert policy.update() is None
    fill_buffer(policy, 1, seed=5)
    assert policy.update() is not None


def test_targets_follow_polyak_average(rng: np.random.Generator) -> None:
    tau = 0.1
    policy = SACAlgorithm(make_spec(), small_sac_config(tau=tau), rng)
    fill_buffer(policy, 16)
    old_target = [p.detach().clone() for p in policy.q1_target.parameters()]
    policy.update()
    for old, target, online in zip(
        old_target, policy.q1_target.parameters(), policy.q1.parameters(), strict=True
    ):
        torch.testing.assert_close(target, (1 - tau) * old + tau * online.detach())


def test_critic_target_formula(rng: np.random.Generator) -> None:
    policy = SACAlgorithm(make_spec(), small_sac_config(), rng)
    fill_buffer(policy, 16)
    batch = policy.buffer.sample(8, policy.device)
    state = policy._torch_gen.get_state()
    target = policy.critic_target(batch)
    policy._torch_gen.set_state(state)
    with torch.no_grad():
        next_u, next_logp = policy.actor.sample(
            batch.next_obs, generator=policy._torch_gen
        )
        q = torch.min(
            policy.q1_target(batch.next_obs, next_u),
            policy.q2_target(batch.next_obs, next_u),
        )
        expected = batch.rewards + policy.config.gamma * (q - policy.alpha * next_logp)
    torch.testing.assert_close(target, expected)
