"""Tests of the frozen hyperparameter configs."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from abmem.services.algorithms import registry
from abmem.services.algorithms.algorithm_config import (
    AlgorithmConfig,
    DDPGConfig,
    FixedMarkupConfig,
    PPOConfig,
    RDPGConfig,
    SACConfig,
    TD3Config,
)

ALL_CLASSES = sorted(
    {cls.config_class for cls in registry.AGENT_ALGORITHMS.classes().values()},
    key=lambda c: c.__name__,
)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_defaults_validate_and_from_dict_none_gives_defaults(
    cls: type[AlgorithmConfig],
) -> None:
    default = cls()
    assert cls.from_dict(None) == default
    assert cls.from_dict({}) == default


def test_resolved_defaults() -> None:
    base = AlgorithmConfig()
    assert base.gradient_steps == 1
    assert base.learning_starts == 64
    assert base.batch_size == 64
    assert base.hidden_sizes == (256, 256)
    assert PPOConfig().rollout_length == 64
    assert FixedMarkupConfig().markup == pytest.approx(0.10)
    assert TD3Config().policy_delay == 2
    assert SACConfig().target_entropy is None
    assert RDPGConfig().seq_len == 24


def test_every_registered_algorithm_has_its_own_config_class() -> None:
    classes = registry.AGENT_ALGORITHMS.classes()
    assert len(ALL_CLASSES) == len(classes) == 8
    for cls in classes.values():
        assert issubclass(cls.config_class, AlgorithmConfig)


def test_frozen() -> None:
    config = DDPGConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gamma = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("key", ["NOISE_STD", "unknown_field", "rollout_length"])
def test_unknown_key_raises_and_names_key(key: str) -> None:
    with pytest.raises(ValueError, match=key) as info:
        DDPGConfig.from_dict({key: 1})
    assert "accepted keys" in str(info.value)


def test_from_dict_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        DDPGConfig.from_dict([("gamma", 0.9)])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [64.0, np.float64(64.0), np.int64(64)])
def test_integral_float_accepted_for_int_field(value: object) -> None:
    config = DDPGConfig.from_dict({"batch_size": value, "learning_starts": 128.0})
    assert config.batch_size == 64 and type(config.batch_size) is int
    assert config.learning_starts == 128 and type(config.learning_starts) is int


@pytest.mark.parametrize("value", [64.5, True, "64", math.nan, math.inf])
def test_invalid_int_field_values_raise(value: object) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        DDPGConfig.from_dict({"batch_size": value})


def test_int_accepted_for_float_field() -> None:
    config = DDPGConfig.from_dict({"tau": 1})
    assert config.tau == 1.0 and type(config.tau) is float


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("key", ["gamma", "lr_actor", "exploration_sigma"])
def test_non_finite_values_raise(key: str, value: float) -> None:
    with pytest.raises(ValueError, match=key):
        DDPGConfig.from_dict({key: value})


@pytest.mark.parametrize(
    "changes",
    [
        {"gamma": 0.0},
        {"gamma": 1.0},
        {"tau": 0.0},
        {"tau": 1.5},
        {"lr_actor": 0.0},
        {"lr_critic": -1e-3},
        {"batch_size": 0},
        {"buffer_size": 32},
        {"learning_starts": 32},
        {"gradient_steps": 0},
        {"hidden_sizes": ()},
        {"hidden_sizes": (64, 0)},
        {"hidden_sizes": "64"},
        {"exploration_sigma": -0.1},
        {"exploration_sigma_decay": 0.0},
        {"exploration_sigma_decay": 1.01},
        {"exploration_sigma_min": -0.01},
        {"max_grad_norm": 0.0},
        {"max_grad_norm": math.nan},
    ],
)
def test_invalid_common_ranges(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DDPGConfig.from_dict(changes)


@pytest.mark.parametrize(
    ("cls", "changes"),
    [
        (TD3Config, {"policy_noise": -0.1}),
        (TD3Config, {"noise_clip": -0.1}),
        (TD3Config, {"policy_delay": 0}),
        (SACConfig, {"lr_alpha": 0.0}),
        (SACConfig, {"initial_alpha": 0.0}),
        (SACConfig, {"target_entropy": math.nan}),
        (PPOConfig, {"rollout_length": 16, "minibatch_size": 32}),
        (PPOConfig, {"gae_lambda": -0.1}),
        (PPOConfig, {"gae_lambda": 1.1}),
        (PPOConfig, {"clip_eps": 0.0}),
        (PPOConfig, {"epochs": 0}),
        (PPOConfig, {"minibatch_size": 0}),
        (PPOConfig, {"entropy_coef": -0.01}),
        (PPOConfig, {"value_coef": -0.5}),
        (RDPGConfig, {"seq_len": 1}),
        (RDPGConfig, {"lstm_hidden": 0}),
        (FixedMarkupConfig, {"markup": -1.0}),
    ],
)
def test_invalid_subclass_ranges(
    cls: type[AlgorithmConfig], changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        cls.from_dict(changes)


def test_boundary_values_accepted() -> None:
    PPOConfig(gae_lambda=0.0)
    PPOConfig(gae_lambda=1.0, rollout_length=32, minibatch_size=32)
    DDPGConfig(tau=1.0, exploration_sigma=0.0, exploration_sigma_decay=1.0)
    FixedMarkupConfig(markup=-0.99)
    SACConfig(target_entropy=-3.0)
    RDPGConfig(seq_len=2)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_round_trip_to_dict_from_dict(cls: type[AlgorithmConfig]) -> None:
    config = cls().replace(gamma=0.95, hidden_sizes=(32, 16))
    as_dict = config.to_dict()
    assert as_dict["hidden_sizes"] == [32, 16]
    assert set(as_dict) == set(cls.field_names())
    assert cls.from_dict(as_dict) == config


def test_replace_validates_and_rejects_unknown() -> None:
    config = DDPGConfig()
    assert config.replace(gamma=0.9).gamma == 0.9
    assert config.gamma == 0.99
    with pytest.raises(ValueError):
        config.replace(gamma=2.0)
    with pytest.raises(ValueError, match="bogus"):
        config.replace(bogus=1)


@pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
def test_default_search_bounds_keys_are_fields_and_valid(
    cls: type[AlgorithmConfig],
) -> None:
    bounds = cls.default_search_bounds()
    fields = set(cls.field_names())
    assert set(bounds) <= fields
    for name, (low, high, scale) in bounds.items():
        assert scale in ("linear", "log", "int"), name
        assert low < high, name
        if scale == "log":
            assert low > 0, name
        # Both ends of every bound validate together with the other defaults.
        cls.from_dict({name: low})
        cls.from_dict({name: high})
    if bounds:
        cls.from_dict({name: b[0] for name, b in bounds.items()})
        cls.from_dict({name: b[1] for name, b in bounds.items()})
