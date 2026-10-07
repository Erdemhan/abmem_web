"""Tests of PPOAlgorithm (Schulman et al., 2017, with GAE of Schulman et al., 2016)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from abmem.services.algorithms.algorithm_config import PPOConfig
from abmem.services.algorithms.base import PolicySpec, Transition
from abmem.services.algorithms.ppo import PPOAlgorithm, tanh_gaussian_log_prob
from tests.algorithms.conftest import make_spec
from tests.algorithms.contract import assert_policy_contract


def small_ppo_config(**changes: object) -> PPOConfig:
    """Return a small, fast PPOConfig for tests."""
    base = PPOConfig(
        batch_size=8,
        buffer_size=64,
        learning_starts=16,
        rollout_length=16,
        minibatch_size=8,
        epochs=2,
        hidden_sizes=(16,),
    )
    return base.replace(**changes) if changes else base


def make_policy(
    spec: PolicySpec | None = None, seed: int = 0, **changes: object
) -> PPOAlgorithm:
    """Return a fresh PPOAlgorithm with a small config."""
    return PPOAlgorithm(
        spec or make_spec(action_dim=2),
        small_ppo_config(**changes),
        np.random.default_rng(seed),
    )


def fill_rollout(
    policy: PPOAlgorithm,
    n: int | None = None,
    intervened: bool | list[bool] = False,
    seed: int = 1,
) -> None:
    """Observe ``n`` transitions sampled from ``policy`` (default: one rollout)."""
    data_rng = np.random.default_rng(seed)
    n = policy.config.rollout_length if n is None else n
    flags = intervened if isinstance(intervened, list) else [intervened] * n
    obs = data_rng.uniform(-1, 1, policy.spec.obs_dim).astype(np.float32)
    for i in range(n):
        result = policy.select_action(obs, explore=True)
        next_obs = data_rng.uniform(-1, 1, policy.spec.obs_dim).astype(np.float32)
        policy.observe(
            Transition(
                obs=obs,
                action=result.u,
                reward=float(data_rng.normal()),
                next_obs=next_obs,
                behaviour_log_prob=result.log_prob,
                behaviour_value=result.value,
                pre_squash=result.pre_squash,
                intervened=flags[i],
            )
        )
        obs = next_obs


def params_of(module: torch.nn.Module) -> list[torch.Tensor]:
    """Return detached clones of the parameters of ``module``."""
    return [p.detach().clone() for p in module.parameters()]


# Contract -----------------------------------------------------------------------


def test_contract(spec: PolicySpec, rng: np.random.Generator) -> None:
    assert_policy_contract(PPOAlgorithm(spec, small_ppo_config(), rng), spec)


def test_default_config_is_valid() -> None:
    policy = PPOAlgorithm(make_spec(), PPOConfig(), np.random.default_rng(0))
    assert policy.buffer.length == 64


def test_learning_starts_above_rollout_length_rejected() -> None:
    with pytest.raises(ValueError, match="rollout_length"):
        make_policy(learning_starts=32, rollout_length=16)


def test_no_legacy_mechanisms() -> None:
    policy = make_policy()
    for name in (
        "risk_tolerance",
        "update_failstack",
        "get_failstack_adjustment",
        "failstack",
        "noise_scale",
    ):
        assert not hasattr(policy, name), name


# Acting -------------------------------------------------------------------------


@pytest.mark.parametrize("action_dim", [1, 3, 5])
def test_one_action_per_plant(action_dim: int) -> None:
    policy = make_policy(make_spec(action_dim=action_dim))
    obs = np.zeros(3, dtype=np.float32)
    for explore in (True, False):
        result = policy.select_action(obs, explore=explore)
        assert result.u.shape == (action_dim,)
        assert result.pre_squash is not None
        assert result.pre_squash.shape == (action_dim,)
        assert result.log_prob is not None and result.value is not None
        np.testing.assert_allclose(result.u, np.tanh(result.pre_squash), rtol=1e-6)


def test_greedy_action_is_tanh_of_mean() -> None:
    policy = make_policy()
    obs = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    with torch.no_grad():
        mean, _ = policy.policy(torch.as_tensor(obs).unsqueeze(0))
    result = policy.select_action(obs, explore=False)
    np.testing.assert_allclose(result.u, torch.tanh(mean).squeeze(0).numpy(), rtol=1e-6)


def test_exploration_scale_is_mean_std() -> None:
    policy = make_policy(initial_log_std=-1.0)
    assert policy.exploration_scale() == pytest.approx(math.exp(-1.0), rel=1e-6)


def test_identical_seeds_give_identical_actions() -> None:
    actions = []
    for _ in range(2):
        torch.manual_seed(7)
        policy = make_policy(seed=3)
        obs = np.full(3, 0.5, dtype=np.float32)
        actions.append([policy.select_action(obs, True).u for _ in range(5)])
    np.testing.assert_array_equal(np.stack(actions[0]), np.stack(actions[1]))


def test_log_prob_matches_reference_formula() -> None:
    mean = torch.tensor([[0.2, -0.4]])
    log_std = torch.tensor([-0.5, 0.3])
    z = torch.tensor([[0.7, -1.1]])
    std = torch.exp(log_std)
    expected = (
        torch.distributions.Normal(mean, std).log_prob(z)
        - torch.log(1.0 - torch.tanh(z) ** 2)
    ).sum(-1)
    torch.testing.assert_close(tanh_gaussian_log_prob(mean, log_std, z), expected)


def test_log_prob_finite_under_tanh_saturation() -> None:
    mean = torch.zeros(1, 2)
    z = torch.tensor([[50.0, -80.0]])
    value = tanh_gaussian_log_prob(mean, torch.zeros(2), z)
    assert torch.isfinite(value).all()
    policy = make_policy(initial_log_std=4.0)
    for _ in range(50):
        result = policy.select_action(np.zeros(3, dtype=np.float32), explore=True)
        assert math.isfinite(result.log_prob)  # type: ignore[arg-type]


def test_stored_log_prob_equals_evaluate_before_update() -> None:
    policy = make_policy()
    fill_rollout(policy)
    batch = policy.buffer.to_tensors(policy.device)
    with torch.no_grad():
        recomputed = policy.evaluate(batch.obs, batch.pre_squash)
    torch.testing.assert_close(recomputed, batch.log_probs, rtol=1e-5, atol=1e-5)


# Rollout schedule ---------------------------------------------------------------


def test_update_only_on_full_rollout_and_buffer_cleared() -> None:
    policy = make_policy()
    fill_rollout(policy, n=policy.config.rollout_length - 1)
    assert not policy.ready_to_learn
    assert policy.update() is None
    fill_rollout(policy, n=1, seed=2)
    assert policy.ready_to_learn
    stats = policy.update()
    assert stats is not None
    assert stats.actor_loss is not None and math.isfinite(stats.actor_loss)
    assert stats.critic_loss is not None and math.isfinite(stats.critic_loss)
    assert len(policy.buffer) == 0
    assert not policy.ready_to_learn
    assert policy.update() is None


def test_ratio_is_one_on_first_minibatch_of_fresh_rollout() -> None:
    policy = make_policy()
    fill_rollout(policy)
    stats = policy.update()
    assert stats is not None
    assert stats.extra["first_minibatch_max_ratio_dev"] < 1e-5


def test_update_changes_parameters() -> None:
    policy = make_policy()
    before_pi, before_v = params_of(policy.policy), params_of(policy.value_net)
    fill_rollout(policy)
    assert policy.update() is not None
    assert any(
        not torch.equal(a, b) for a, b in zip(before_pi, policy.policy.parameters())
    )
    assert any(
        not torch.equal(a, b) for a, b in zip(before_v, policy.value_net.parameters())
    )


# Intervened steps (K3) ----------------------------------------------------------


def _losses_full_batch(
    policy: PPOAlgorithm, advantages: torch.Tensor, returns: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch = policy.buffer.to_tensors(policy.device)
    idx = torch.arange(len(policy.buffer))
    return policy.minibatch_losses(batch, advantages, returns, idx)


def test_intervened_steps_excluded_from_policy_loss_only() -> None:
    policy = make_policy()
    n = policy.config.rollout_length
    flags = [i % 2 == 0 for i in range(n)]
    fill_rollout(policy, intervened=flags)
    mask = torch.as_tensor(policy.buffer.intervened_mask)
    adv = torch.linspace(-1.0, 1.0, n)
    ret = torch.linspace(0.0, 2.0, n)
    base = _losses_full_batch(policy, adv, ret)
    # Changing advantages and returns only on intervened steps.
    adv_changed = torch.where(mask, adv + 100.0, adv)
    ret_changed = torch.where(mask, ret + 100.0, ret)
    changed = _losses_full_batch(policy, adv_changed, ret_changed)
    torch.testing.assert_close(changed["policy_loss"], base["policy_loss"])
    torch.testing.assert_close(changed["entropy"], base["entropy"])
    assert not torch.allclose(changed["value_loss"], base["value_loss"])
    # The value loss averages over all steps.
    values = base["values"]
    torch.testing.assert_close(base["value_loss"], ((values - ret) ** 2).mean())


def test_all_intervened_rollout_gives_zero_policy_gradient() -> None:
    policy = make_policy()
    fill_rollout(policy, intervened=True)
    n = policy.config.rollout_length
    losses = _losses_full_batch(policy, torch.linspace(-1, 1, n), torch.ones(n))
    assert losses["policy_loss"].item() == 0.0
    total = losses["policy_loss"] + losses["entropy"] + losses["value_loss"]
    policy.policy.zero_grad(set_to_none=True)
    policy.value_net.zero_grad(set_to_none=True)
    total.backward()
    for p in policy.policy.parameters():
        assert p.grad is None or torch.count_nonzero(p.grad) == 0
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in policy.value_net.parameters()
    )


def test_all_intervened_update_keeps_policy_and_trains_value() -> None:
    policy = make_policy(entropy_coef=0.01)
    fill_rollout(policy, intervened=True)
    before_pi, before_v = params_of(policy.policy), params_of(policy.value_net)
    stats = policy.update()
    assert stats is not None
    assert stats.actor_loss == 0.0
    assert stats.extra["intervened_fraction"] == 1.0
    for a, b in zip(before_pi, policy.policy.parameters()):
        assert torch.equal(a, b)
    assert any(
        not torch.equal(a, b) for a, b in zip(before_v, policy.value_net.parameters())
    )


# GAE ----------------------------------------------------------------------------


def test_gae_matches_hand_computation() -> None:
    policy = make_policy(
        make_spec(action_dim=1),
        gamma=0.9,
        gae_lambda=0.8,
        learning_starts=8,
        rollout_length=8,
        minibatch_size=4,
    )
    # V(next_obs) of the last step is the bias 0.3 of a zeroed value network.
    with torch.no_grad():
        for p in policy.value_net.parameters():
            p.zero_()
        policy.value_net[-1].bias.fill_(0.3)
    rewards = [1.0, 0.0, 2.0]
    values = [0.5, 0.2, -0.1]
    obs = np.zeros(3, dtype=np.float32)
    for r, v in zip(rewards, values):
        policy.observe(
            Transition(
                obs=obs,
                action=np.zeros(1, dtype=np.float32),
                reward=r,
                next_obs=obs,
                behaviour_log_prob=0.0,
                behaviour_value=v,
                pre_squash=np.zeros(1, dtype=np.float32),
            )
        )
    advantages, returns = policy.compute_advantages()
    # delta_2 = 2 + 0.9*0.3 + 0.1 = 2.37, delta_1 = 0 - 0.09 - 0.2 = -0.29,
    # delta_0 = 1 + 0.18 - 0.5 = 0.68, gamma*lambda = 0.72.
    expected_adv = np.array([1.699808, 1.4164, 2.37])
    np.testing.assert_allclose(advantages, expected_adv, rtol=1e-5)
    np.testing.assert_allclose(returns, expected_adv + np.array(values), rtol=1e-5)


def test_compute_advantages_on_empty_rollout_raises() -> None:
    with pytest.raises(ValueError):
        make_policy().compute_advantages()


# Frozen -------------------------------------------------------------------------


def test_frozen_policy_does_not_store_or_update() -> None:
    policy = make_policy()
    policy.set_frozen(True)
    fill_rollout(policy)
    assert len(policy.buffer) == 0
    assert policy.update() is None
