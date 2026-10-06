"""Tests of the shared ReplayBuffer and RolloutBuffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from abmem.services.algorithms.base import Transition
from abmem.services.algorithms.replay_buffer import ReplayBuffer, RolloutBuffer

CPU = torch.device("cpu")


def make_transition(
    k: float,
    obs_dim: int = 3,
    action_dim: int = 2,
    *,
    on_policy: bool = False,
    terminated: bool = False,
    intervened: bool = False,
) -> Transition:
    """Return a transition whose fields encode the step index ``k``."""
    extra = {}
    if on_policy:
        extra = {
            "behaviour_log_prob": -0.5 * k,
            "behaviour_value": 0.1 * k,
            "pre_squash": np.full(action_dim, 0.01 * k),
        }
    return Transition(
        obs=np.full(obs_dim, k, dtype=np.float32),
        action=np.full(action_dim, np.tanh(0.01 * k), dtype=np.float32),
        reward=k + 0.25,
        next_obs=np.full(obs_dim, k + 1, dtype=np.float32),
        terminated=terminated,
        intervened=intervened,
        **extra,
    )


# ReplayBuffer -----------------------------------------------------------------


def test_add_sample_shapes_and_dtypes(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(obs_dim=3, action_dim=2, capacity=50, rng=rng)
    assert len(buf) == 0
    for k in range(20):
        buf.add(make_transition(k, intervened=k % 2 == 0))
    assert len(buf) == 20
    batch = buf.sample(8, CPU)
    assert batch.obs.shape == (8, 3)
    assert batch.actions.shape == (8, 2)
    assert batch.rewards.shape == (8, 1)
    assert batch.next_obs.shape == (8, 3)
    assert batch.terminated.shape == (8, 1)
    assert batch.intervened.shape == (8, 1)
    for tensor in (batch.obs, batch.actions, batch.rewards, batch.terminated):
        assert tensor.dtype == torch.float32
    # Rows stay aligned: obs encodes k, reward = k + 0.25, next_obs = k + 1.
    k = batch.obs[:, 0]
    torch.testing.assert_close(batch.rewards[:, 0], k + 0.25)
    torch.testing.assert_close(batch.next_obs[:, 0], k + 1)
    torch.testing.assert_close(batch.intervened[:, 0], (k % 2 == 0).float())


def test_sample_without_replacement(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 30, rng)
    for k in range(30):
        buf.add(make_transition(k))
    batch = buf.sample(30, CPU)
    assert sorted(batch.obs[:, 0].tolist()) == list(range(30))


def test_rewards_stored_as_float32_without_rounding(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 10, rng)
    buf.add(make_transition(0.0))
    reward = 1234.5678
    t = make_transition(1.0)
    buf.add(Transition(t.obs, t.action, reward, t.next_obs))
    assert buf.rewards.dtype == np.float32
    assert buf.rewards[-1] == np.float32(reward)
    assert buf.rewards[0] == np.float32(0.25)


def test_sample_empty_or_too_small_raises(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 10, rng)
    with pytest.raises(ValueError):
        buf.sample(1, CPU)
    for k in range(4):
        buf.add(make_transition(k))
    with pytest.raises(ValueError, match="ready_to_learn"):
        buf.sample(5, CPU)
    with pytest.raises(ValueError):
        buf.sample(0, CPU)


def test_wrap_around_overwrites_oldest(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 5, rng)
    for k in range(12):
        buf.add(make_transition(k))
    assert len(buf) == 5
    assert buf.rewards.tolist() == [k + 0.25 for k in range(7, 12)]
    batch = buf.sample(5, CPU)
    assert sorted(batch.obs[:, 0].tolist()) == list(range(7, 12))


def test_add_rejects_bad_shapes_and_unnormalised_actions(
    rng: np.random.Generator,
) -> None:
    buf = ReplayBuffer(3, 2, 5, rng)
    with pytest.raises(ValueError):
        buf.add(make_transition(0, obs_dim=4))
    with pytest.raises(ValueError):
        buf.add(make_transition(0, action_dim=3))
    t = make_transition(0)
    with pytest.raises(ValueError, match="normalised"):
        buf.add(Transition(t.obs, np.array([1.5, 0.0]), 0.0, t.next_obs))
    with pytest.raises(TypeError):
        buf.add("not a transition")  # type: ignore[arg-type]
    assert len(buf) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"obs_dim": 0, "action_dim": 1, "capacity": 5},
        {"obs_dim": 3, "action_dim": 0, "capacity": 5},
        {"obs_dim": 3, "action_dim": 1, "capacity": 0},
    ],
)
def test_invalid_constructor_arguments(
    rng: np.random.Generator, kwargs: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(rng=rng, **kwargs)


def test_transition_rejects_nan_inputs() -> None:
    t = make_transition(0)
    with pytest.raises(ValueError):
        Transition(np.array([np.nan, 0.0, 0.0]), t.action, 0.0, t.next_obs)
    with pytest.raises(ValueError):
        Transition(t.obs, t.action, float("nan"), t.next_obs)
    with pytest.raises(ValueError):
        Transition(t.obs, t.action, float("inf"), t.next_obs)
    with pytest.raises(ValueError):
        Transition(t.obs, np.array([np.inf, 0.0]), 0.0, t.next_obs)


def test_sample_sequences_contiguous_before_wrap(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 100, rng)
    for k in range(30):
        buf.add(make_transition(k))
    batch = buf.sample_sequences(10, 6, CPU)
    assert batch.obs.shape == (10, 6, 3)
    assert batch.actions.shape == (10, 6, 2)
    assert batch.rewards.shape == (10, 6, 1)
    assert batch.terminated.shape == (10, 6, 1)
    steps = batch.obs[:, :, 0]
    assert torch.all(steps[:, 1:] - steps[:, :-1] == 1)


def test_sample_sequences_never_span_overwrite_boundary(
    rng: np.random.Generator,
) -> None:
    capacity = 10
    buf = ReplayBuffer(3, 2, capacity, rng)
    for k in range(27):  # write position ends in the middle of the array
        buf.add(make_transition(k))
    batch = buf.sample_sequences(200, 4, CPU)
    steps = batch.obs[:, :, 0]
    assert torch.all(steps[:, 1:] - steps[:, :-1] == 1)
    assert steps.min().item() >= 17 and steps.max().item() <= 26
    # Every valid window start among the stored steps 17..23 can be drawn.
    assert set(steps[:, 0].tolist()) == set(range(17, 24))


def test_sample_sequences_without_replacement_when_possible(
    rng: np.random.Generator,
) -> None:
    buf = ReplayBuffer(3, 2, 50, rng)
    for k in range(20):
        buf.add(make_transition(k))
    batch = buf.sample_sequences(16, 5, CPU)  # exactly 16 windows exist
    assert sorted(batch.obs[:, 0, 0].tolist()) == list(range(16))


def test_sample_sequences_too_short_raises(rng: np.random.Generator) -> None:
    buf = ReplayBuffer(3, 2, 50, rng)
    for k in range(3):
        buf.add(make_transition(k))
    with pytest.raises(ValueError):
        buf.sample_sequences(1, 4, CPU)
    with pytest.raises(ValueError):
        buf.sample_sequences(1, 0, CPU)


def test_sampling_reproducible_per_seed() -> None:
    def draw(seed: int) -> torch.Tensor:
        buf = ReplayBuffer(3, 2, 50, np.random.default_rng(seed))
        for k in range(40):
            buf.add(make_transition(k))
        return buf.sample(10, CPU).obs

    torch.testing.assert_close(draw(3), draw(3))


# RolloutBuffer ----------------------------------------------------------------


def test_rollout_add_full_clear() -> None:
    buf = RolloutBuffer(3, 2, length=3)
    assert len(buf) == 0 and not buf.full()
    for k in range(3):
        buf.add(make_transition(k, on_policy=True, intervened=k == 1))
    assert buf.full()
    assert buf.intervened_mask.tolist() == [False, True, False]
    with pytest.raises(ValueError, match="full"):
        buf.add(make_transition(3, on_policy=True))
    batch = buf.to_tensors(CPU)
    assert batch.obs.shape == (3, 3)
    assert batch.pre_squash.shape == (3, 2)
    assert batch.log_probs.shape == (3,)
    torch.testing.assert_close(batch.values, torch.tensor([0.0, 0.1, 0.2]))
    torch.testing.assert_close(batch.intervened_mask, torch.tensor([0.0, 1.0, 0.0]))
    buf.clear()
    assert len(buf) == 0 and not buf.full()


def test_rollout_requires_behaviour_statistics() -> None:
    buf = RolloutBuffer(3, 2, length=3)
    with pytest.raises(ValueError, match="behaviour"):
        buf.add(make_transition(0, on_policy=False))
    assert len(buf) == 0


def test_gae_three_step_hand_computed() -> None:
    gamma, lam = 0.9, 0.8
    buf = RolloutBuffer(1, 1, length=3)
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    for r, v in zip(rewards, values, strict=True):
        buf.add(
            Transition(
                obs=np.zeros(1),
                action=np.zeros(1),
                reward=r,
                next_obs=np.zeros(1),
                behaviour_log_prob=0.0,
                behaviour_value=v,
                pre_squash=np.zeros(1),
            )
        )
    last_value = 2.0
    # delta_t = r_t + gamma * V_{t+1} - V_t
    d0 = 1.0 + 0.9 * 1.0 - 0.5  # 1.4
    d1 = 2.0 + 0.9 * 1.5 - 1.0  # 2.35
    d2 = 3.0 + 0.9 * 2.0 - 1.5  # 3.3
    a2 = d2
    a1 = d1 + gamma * lam * a2
    a0 = d0 + gamma * lam * a1
    adv, ret = buf.compute_gae(last_value, gamma, lam)
    np.testing.assert_allclose(adv, [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(ret, np.array([a0, a1, a2]) + values, rtol=1e-6)
    assert adv.dtype == np.float32 and ret.dtype == np.float32


def test_gae_terminated_cuts_bootstrap() -> None:
    buf = RolloutBuffer(1, 1, length=2)
    for r, v, term in ((1.0, 0.5, True), (2.0, 1.0, False)):
        buf.add(
            Transition(
                obs=np.zeros(1),
                action=np.zeros(1),
                reward=r,
                next_obs=np.zeros(1),
                terminated=term,
                behaviour_log_prob=0.0,
                behaviour_value=v,
                pre_squash=np.zeros(1),
            )
        )
    adv, _ = buf.compute_gae(3.0, 0.9, 0.95)
    assert adv[0] == pytest.approx(1.0 - 0.5)
    assert adv[1] == pytest.approx(2.0 + 0.9 * 3.0 - 1.0)


def test_gae_length_one() -> None:
    buf = RolloutBuffer(1, 1, length=1)
    buf.add(
        Transition(
            obs=np.zeros(1),
            action=np.zeros(1),
            reward=1.0,
            next_obs=np.zeros(1),
            behaviour_log_prob=0.0,
            behaviour_value=0.5,
            pre_squash=np.zeros(1),
        )
    )
    adv, ret = buf.compute_gae(2.0, 0.99, 0.95)
    assert adv[0] == pytest.approx(1.0 + 0.99 * 2.0 - 0.5)
    assert ret[0] == pytest.approx(adv[0] + 0.5)


def test_gae_invalid_arguments() -> None:
    buf = RolloutBuffer(1, 1, length=2)
    with pytest.raises(ValueError, match="empty"):
        buf.compute_gae(0.0, 0.99, 0.95)
    buf.add(make_transition(0, obs_dim=1, action_dim=1, on_policy=True))
    with pytest.raises(ValueError):
        buf.compute_gae(float("nan"), 0.99, 0.95)
    with pytest.raises(ValueError):
        buf.compute_gae(0.0, 1.0, 0.95)
    with pytest.raises(ValueError):
        buf.compute_gae(0.0, 0.99, 1.5)
