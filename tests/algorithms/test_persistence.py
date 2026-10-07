"""Checkpoint round trip of every learning algorithm (finding A9)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import torch

from abmem.services.algorithms import persistence
from abmem.services.algorithms.algorithm_config import (
    DDPGConfig,
    PPOConfig,
    RDPGConfig,
    SACConfig,
    TD3Config,
)
from abmem.services.algorithms.base import BaseAgentAlgorithm, Transition
from abmem.services.algorithms.persistence import CheckpointError
from abmem.services.algorithms.registry import get_algorithm_class
from tests.algorithms.conftest import make_spec

SMALL_CONFIGS: dict[str, Any] = {
    "DDPG": DDPGConfig(
        batch_size=8, buffer_size=256, learning_starts=8, hidden_sizes=(16,)
    ),
    "TD3": TD3Config(
        batch_size=8, buffer_size=256, learning_starts=8, hidden_sizes=(16,)
    ),
    "SAC": SACConfig(
        batch_size=8, buffer_size=256, learning_starts=8, hidden_sizes=(16,)
    ),
    "PPO": PPOConfig(
        batch_size=8,
        buffer_size=64,
        learning_starts=8,
        rollout_length=8,
        minibatch_size=4,
        epochs=2,
        hidden_sizes=(16,),
    ),
    "RDPG": RDPGConfig(
        batch_size=4,
        buffer_size=256,
        learning_starts=8,
        hidden_sizes=(16,),
        seq_len=4,
        lstm_hidden=8,
    ),
}
LEARNING_TYPES = list(SMALL_CONFIGS)
ALLOWED_LEAF_TYPES = (torch.Tensor, int, float, bool, str, type(None))


def _build(
    algorithm: str, seed: int, action_dim: int = 2, obs_dim: int = 3
) -> BaseAgentAlgorithm:
    torch.manual_seed(seed)
    cls = get_algorithm_class(algorithm)
    return cls(
        make_spec(action_dim=action_dim, obs_dim=obs_dim),
        SMALL_CONFIGS[algorithm],
        np.random.default_rng(seed),
    )


def _step(
    policy: BaseAgentAlgorithm, obs: np.ndarray, data_rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Act with exploration, store the transition, update and end the period."""
    result = policy.select_action(obs, explore=True)
    next_obs = data_rng.normal(size=obs.shape).astype(np.float32)
    policy.observe(
        Transition(
            obs=obs,
            action=result.u,
            reward=float(data_rng.normal(0.0, 10.0)),
            next_obs=next_obs,
            behaviour_log_prob=result.log_prob,
            behaviour_value=result.value,
            pre_squash=result.pre_squash,
        )
    )
    stats = policy.update()
    policy.end_period()
    return result.u, next_obs, stats


def _train(policy: BaseAgentAlgorithm, n: int, seed: int) -> np.ndarray:
    data_rng = np.random.default_rng(seed)
    obs = data_rng.normal(size=policy.spec.obs_dim).astype(np.float32)
    for _ in range(n):
        _, obs, _ = _step(policy, obs, data_rng)
    return obs


def _parameters(policy: BaseAgentAlgorithm) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{key}": value.detach().clone()
        for name, module in policy._state_modules().items()
        for key, value in module.state_dict().items()
    }


def _reset_buffer(policy: BaseAgentAlgorithm) -> None:
    """Give ``policy`` an empty buffer, as a restored policy has."""
    policy.buffer = policy._build_buffer()
    if hasattr(policy, "_last_next_obs"):
        policy._last_next_obs = None


def _leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        assert all(isinstance(k, (str, int)) for k in value), list(value)
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, (list, tuple)):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value]


@pytest.mark.parametrize("algorithm", LEARNING_TYPES, ids=str)
def test_round_trip_acts_and_updates_identically(
    algorithm: str, tmp_path: pathlib.Path
) -> None:
    original = _build(algorithm, seed=0)
    obs = _train(original, 30, seed=1)
    path = tmp_path / "policy.pt"
    size, digest = persistence.save_checkpoint(path, original, {"run": 1})
    assert size == path.stat().st_size > 0
    assert len(digest) == 64

    restored = _build(algorithm, seed=99)
    checkpoint = persistence.load_checkpoint(path, expected_sha256=digest)
    persistence.load_state_into(restored, checkpoint, restore_runtime=True)

    for key, tensor in _parameters(original).items():
        assert torch.equal(tensor, _parameters(restored)[key]), key
    probe_rng = np.random.default_rng(5)
    for _ in range(3):
        probe = probe_rng.normal(size=3).astype(np.float32)
        a = original.select_action(probe, explore=False)
        b = restored.select_action(probe, explore=False)
        np.testing.assert_array_equal(a.u, b.u)
    assert original.exploration_scale() == restored.exploration_scale()

    # Same generator state and the same transitions give the same updates.
    _reset_buffer(original)
    _reset_buffer(restored)
    rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
    obs_a, obs_b = obs, obs.copy()
    updated = False
    for _ in range(12):
        u_a, obs_a, stats_a = _step(original, obs_a, rng_a)
        u_b, obs_b, stats_b = _step(restored, obs_b, rng_b)
        np.testing.assert_array_equal(u_a, u_b)
        assert stats_a == stats_b
        updated = updated or stats_a is not None
    assert updated
    for key, tensor in _parameters(original).items():
        assert torch.equal(tensor, _parameters(restored)[key]), key


@pytest.mark.parametrize("algorithm", LEARNING_TYPES, ids=str)
def test_checkpoint_holds_only_tensors_and_plain_types(
    algorithm: str,
) -> None:
    policy = _build(algorithm, seed=0)
    _train(policy, 12, seed=1)
    data = persistence.state_dict_of(policy, {"agent": "a0"})
    assert data["format_version"] == persistence.FORMAT_VERSION
    assert data["algorithm_key"] == algorithm
    assert data["algorithm_name"] == algorithm
    assert "algorithm_type" not in data
    assert data["spec"]["action_dim"] == 2
    for leaf in _leaves(data):
        assert isinstance(leaf, ALLOWED_LEAF_TYPES), type(leaf)


def test_torch_load_uses_weights_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = _build("DDPG", seed=0)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, policy)
    calls: list[dict[str, Any]] = []
    real_load = torch.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(persistence.torch, "load", spy)
    persistence.load_checkpoint(path)
    assert calls and all(call.get("weights_only") is True for call in calls)
    assert all(call.get("map_location") is not None for call in calls)


def test_warm_start_mode_keeps_the_generators_of_the_new_policy(
    tmp_path: pathlib.Path,
) -> None:
    original = _build("TD3", seed=0)
    _train(original, 20, seed=1)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, original)
    target = _build("TD3", seed=42)
    rng_state = target.rng.bit_generator.state
    generator_state = target._noise_generator.get_state()
    persistence.load_state_into(
        target, persistence.load_checkpoint(path), restore_runtime=False
    )
    assert target.rng.bit_generator.state == rng_state
    assert torch.equal(target._noise_generator.get_state(), generator_state)
    assert target.critic_updates == original.critic_updates
    probe = np.ones(3, dtype=np.float32)
    np.testing.assert_array_equal(
        original.select_action(probe, explore=False).u,
        target.select_action(probe, explore=False).u,
    )


def test_rdpg_warm_start_starts_with_an_empty_history(tmp_path: pathlib.Path) -> None:
    original = _build("RDPG", seed=0)
    _train(original, 10, seed=1)
    assert len(original.history) > 0
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, original)
    target = _build("RDPG", seed=3)
    persistence.load_state_into(
        target, persistence.load_checkpoint(path), restore_runtime=False
    )
    assert len(target.history) == 0


def test_frozen_flag_round_trips_only_with_runtime_restore(
    tmp_path: pathlib.Path,
) -> None:
    original = _build("DDPG", seed=0)
    original.set_frozen(True)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, original)
    checkpoint = persistence.load_checkpoint(path)
    exact = _build("DDPG", seed=1)
    persistence.load_state_into(exact, checkpoint, restore_runtime=True)
    assert exact.frozen
    warm = _build("DDPG", seed=1)
    persistence.load_state_into(warm, checkpoint, restore_runtime=False)
    assert not warm.frozen


def test_optimizer_hyperparameters_come_from_the_new_config(
    tmp_path: pathlib.Path,
) -> None:
    original = _build("DDPG", seed=0)
    _train(original, 12, seed=1)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, original)
    config = SMALL_CONFIGS["DDPG"].replace(lr_actor=1e-5)
    target = get_algorithm_class("DDPG")(
        make_spec(action_dim=2), config, np.random.default_rng(1)
    )
    persistence.load_state_into(target, persistence.load_checkpoint(path))
    assert target.actor_optimizer.param_groups[0]["lr"] == 1e-5
    assert len(target.actor_optimizer.state) == len(original.actor_optimizer.state)


@pytest.mark.parametrize(
    ("algorithm", "action_dim", "obs_dim", "message"),
    [
        ("TD3", 2, 3, "algorithm"),
        ("DDPG", 3, 3, "action_dim"),
        ("DDPG", 2, 4, "obs_dim"),
    ],
)
def test_incompatible_policy_is_rejected_and_unchanged(
    tmp_path: pathlib.Path,
    algorithm: str,
    action_dim: int,
    obs_dim: int,
    message: str,
) -> None:
    source = _build("DDPG", seed=0)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, source)
    target = _build(algorithm, seed=1, action_dim=action_dim, obs_dim=obs_dim)
    before = _parameters(target)
    with pytest.raises(CheckpointError, match=message):
        persistence.load_state_into(target, persistence.load_checkpoint(path))
    for key, tensor in before.items():
        assert torch.equal(tensor, _parameters(target)[key])


def test_different_network_shape_is_rejected_and_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    source = _build("DDPG", seed=0)
    path = tmp_path / "p.pt"
    persistence.save_checkpoint(path, source)
    config = SMALL_CONFIGS["DDPG"].replace(hidden_sizes=(8,))
    target = get_algorithm_class("DDPG")(
        make_spec(action_dim=2), config, np.random.default_rng(1)
    )
    before = _parameters(target)
    with pytest.raises(CheckpointError, match="shape"):
        persistence.load_state_into(target, persistence.load_checkpoint(path))
    for key, tensor in before.items():
        assert torch.equal(tensor, _parameters(target)[key])


def test_corrupted_missing_and_foreign_files_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    policy = _build("DDPG", seed=0)
    path = tmp_path / "p.pt"
    _, digest = persistence.save_checkpoint(path, policy)
    with path.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(CheckpointError, match="SHA-256"):
        persistence.load_checkpoint(path, expected_sha256=digest)
    with pytest.raises(CheckpointError, match="does not exist"):
        persistence.load_checkpoint(tmp_path / "missing.pt")
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="cannot be read"):
        persistence.load_checkpoint(garbage)
    old = tmp_path / "old.pt"
    torch.save({"format_version": 0}, old)
    with pytest.raises(CheckpointError, match="format version"):
        persistence.load_checkpoint(old)


def test_checkpoint_written_before_plan_2_10_is_compatible() -> None:
    """Old files store the enum value in ``algorithm_type`` and the key as name."""
    policy = _build("DDPG", seed=0)
    legacy = persistence.state_dict_of(policy)
    del legacy["algorithm_key"]
    legacy["algorithm_type"] = 0
    persistence.check_compatible(policy, legacy)
    legacy["algorithm_name"] = "TD3"
    with pytest.raises(CheckpointError, match="TD3 differs from DDPG"):
        persistence.check_compatible(policy, legacy)


def test_baselines_cannot_be_saved(tmp_path: pathlib.Path) -> None:
    cls = get_algorithm_class("MARGINAL_COST")
    baseline = cls(
        make_spec(action_dim=2),
        cls.config_class(),
        np.random.default_rng(0),
    )
    with pytest.raises(CheckpointError):
        persistence.save_checkpoint(tmp_path / "b.pt", baseline)
    assert not (tmp_path / "b.pt").exists()
