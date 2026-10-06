"""Tests of the registry, the BaseAgentAlgorithm template and the contract helper.

The class-resolution tests are expected to fail until the rl_algorithms
packages land in wave 2. The wave-2 integrator removes the xfail markers.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from abmem.models.enums import AlgorithmType
from abmem.services.algorithms import registry
from abmem.services.algorithms.algorithm_config import (
    CONFIG_CLASSES,
    DDPGConfig,
    FixedMarkupConfig,
    TD3Config,
)
from abmem.services.algorithms.base import (
    ActionResult,
    BaseAgentAlgorithm,
    PolicySpec,
    Transition,
    UpdateStats,
)
from abmem.services.algorithms.replay_buffer import ReplayBuffer
from tests.algorithms.conftest import (
    ConstantBaseline,
    TinyActorCriticPolicy,
    make_spec,
    small_ddpg_config,
)
from tests.algorithms.contract import assert_policy_contract

WAVE2_XFAIL = pytest.mark.xfail(
    reason="algorithm classes land in wave 2 (rl_algorithms); integrator removes this marker",
    strict=False,
)

# Registry ---------------------------------------------------------------------


@pytest.mark.parametrize("algorithm_type", list(AlgorithmType), ids=lambda t: t.name)
def test_config_class_and_default_config(algorithm_type: AlgorithmType) -> None:
    cls = registry.config_class(algorithm_type)
    assert cls is CONFIG_CLASSES[algorithm_type]
    config = registry.default_config(algorithm_type)
    assert type(config) is cls
    assert config == cls()


def test_specific_config_classes() -> None:
    assert registry.config_class(AlgorithmType.TD3) is TD3Config
    assert registry.default_config(AlgorithmType.FIXED_MARKUP) == FixedMarkupConfig(
        markup=0.10
    )


def test_class_paths_declared_for_all_types() -> None:
    assert set(registry.ALGORITHM_CLASS_PATHS) == set(AlgorithmType)
    for path in registry.ALGORITHM_CLASS_PATHS.values():
        module, _, cls = path.partition(":")
        assert module.startswith("abmem.services.algorithms.") and cls
    assert registry.ALGORITHM_CLASS_PATHS[AlgorithmType.PPO] == (
        "abmem.services.algorithms.ppo:PPOAlgorithm"
    )
    assert registry.BASELINE_TYPES == frozenset(
        {AlgorithmType.RANDOM, AlgorithmType.MARGINAL_COST, AlgorithmType.FIXED_MARKUP}
    )


@pytest.mark.parametrize("bad", [99, "DDPG", None, -1])
def test_unknown_algorithm_type_raises_key_error(bad: object) -> None:
    for func in (
        registry.config_class,
        registry.default_config,
        registry.get_algorithm_class,
    ):
        with pytest.raises(KeyError, match="unknown algorithm type"):
            func(bad)  # type: ignore[arg-type]


@WAVE2_XFAIL
@pytest.mark.parametrize("algorithm_type", list(AlgorithmType), ids=lambda t: t.name)
def test_get_algorithm_class_resolves(algorithm_type: AlgorithmType) -> None:
    cls = registry.get_algorithm_class(algorithm_type)
    assert issubclass(cls, BaseAgentAlgorithm)
    assert cls.algorithm_type == algorithm_type
    assert cls.config_class is CONFIG_CLASSES[algorithm_type]
    assert cls.is_learning == (algorithm_type not in registry.BASELINE_TYPES)


@WAVE2_XFAIL
@pytest.mark.parametrize("algorithm_type", list(AlgorithmType), ids=lambda t: t.name)
def test_build_policy_with_none_params_satisfies_contract(
    algorithm_type: AlgorithmType,
) -> None:
    spec = make_spec(action_dim=2)
    policy = registry.build_policy(algorithm_type, spec, None, np.random.default_rng(0))
    assert policy.config == registry.default_config(algorithm_type)
    assert_policy_contract(policy, spec)


def test_build_policy_uses_from_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry, "get_algorithm_class", lambda t: TinyActorCriticPolicy
    )
    spec = make_spec(action_dim=2)
    policy = registry.build_policy(
        AlgorithmType.DDPG,
        spec,
        {"gamma": 0.9, "hidden_sizes": [8]},
        np.random.default_rng(0),
    )
    assert isinstance(policy, TinyActorCriticPolicy)
    assert policy.config == DDPGConfig(gamma=0.9, hidden_sizes=(8,))
    default = registry.build_policy(
        AlgorithmType.DDPG, spec, None, np.random.default_rng(0)
    )
    assert default.config == DDPGConfig()
    with pytest.raises(ValueError, match="NOISE_STD"):
        registry.build_policy(
            AlgorithmType.DDPG, spec, {"NOISE_STD": 0.6}, np.random.default_rng(0)
        )


def test_get_algorithm_class_rejects_wrong_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import types

    fake = types.ModuleType("fake_algorithms")
    fake.DDPGAlgorithm = TinyActorCriticPolicy  # type: ignore[attr-defined]
    fake.NotAPolicy = object  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "fake_algorithms", fake)
    monkeypatch.setattr(
        registry,
        "ALGORITHM_CLASS_PATHS",
        {
            AlgorithmType.DDPG: "fake_algorithms:DDPGAlgorithm",
            AlgorithmType.TD3: "fake_algorithms:DDPGAlgorithm",
            AlgorithmType.SAC: "fake_algorithms:NotAPolicy",
            AlgorithmType.PPO: "fake_algorithms:Missing",
        },
    )
    assert registry.get_algorithm_class(AlgorithmType.DDPG) is TinyActorCriticPolicy
    with pytest.raises(TypeError, match="algorithm_type"):
        registry.get_algorithm_class(AlgorithmType.TD3)
    with pytest.raises(TypeError):
        registry.get_algorithm_class(AlgorithmType.SAC)
    with pytest.raises(ImportError):
        registry.get_algorithm_class(AlgorithmType.PPO)


# Value types ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"obs_dim": 0},
        {"action_dim": 0},
        {"obs_dim": 2.5},
        {"price_high": 0.0},
        {"price_low": float("nan")},
        {"plant_variable_costs": (1.0, 2.0)},
        {"plant_static_costs": (float("inf"),)},
    ],
)
def test_policy_spec_validation(kwargs: dict[str, object]) -> None:
    base: dict[str, object] = {
        "obs_dim": 3,
        "action_dim": 1,
        "price_low": 0.0,
        "price_high": 200.0,
        "plant_variable_costs": (10.0,),
        "plant_static_costs": (5.0,),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        PolicySpec(**base)  # type: ignore[arg-type]


def test_policy_spec_coerces_costs_to_tuples() -> None:
    spec = PolicySpec(3, 2, 0, 200, [1, 2], [3, 4])  # type: ignore[arg-type]
    assert spec.plant_variable_costs == (1.0, 2.0)
    assert isinstance(spec.plant_static_costs, tuple)


@pytest.mark.parametrize(
    "u",
    [
        np.array([0.5, 1.5], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([0.5], dtype=np.float64),
        np.zeros((1, 2), dtype=np.float32),
        [0.0],
    ],
)
def test_action_result_validation(u: object) -> None:
    with pytest.raises(ValueError):
        ActionResult(u=u)  # type: ignore[arg-type]


def test_action_result_optional_fields() -> None:
    u = np.array([0.0, 1.0], dtype=np.float32)
    assert (
        ActionResult(u=u, log_prob=-1.0, value=2.0, pre_squash=np.zeros(2)).log_prob
        == -1.0
    )
    with pytest.raises(ValueError):
        ActionResult(u=u, log_prob=float("nan"))
    with pytest.raises(ValueError):
        ActionResult(u=u, pre_squash=np.zeros(3))


def test_transition_stores_float32_copies_and_float_reward() -> None:
    obs = np.array([1, 2, 3])
    t = Transition(obs=obs, action=[0.5], reward=3, next_obs=[0.0, 0.0, 0.0])
    assert t.obs.dtype == np.float32 and t.action.dtype == np.float32
    assert isinstance(t.reward, float)
    obs[0] = 99
    assert t.obs[0] == 1.0
    with pytest.raises(ValueError):
        Transition(obs=[0.0, 0.0], action=[0.0], reward=0.0, next_obs=[0.0])


# BaseAgentAlgorithm template ----------------------------------------------------


def test_abstract_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseAgentAlgorithm(make_spec(), DDPGConfig(), np.random.default_rng(0))  # type: ignore[abstract]


def test_constructor_type_checks() -> None:
    spec = make_spec()
    with pytest.raises(TypeError, match="DDPGConfig"):
        TinyActorCriticPolicy(spec, TD3Config(), np.random.default_rng(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TinyActorCriticPolicy(spec, DDPGConfig(), 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TinyActorCriticPolicy("spec", DDPGConfig(), np.random.default_rng(0))  # type: ignore[arg-type]


def test_select_action_runs_under_no_grad_and_validates_obs(spec: PolicySpec) -> None:
    policy = TinyActorCriticPolicy(spec, small_ddpg_config(), np.random.default_rng(0))
    result = policy.select_action(np.zeros(spec.obs_dim), explore=False)
    assert result.u.shape == (spec.action_dim,)
    assert policy.select_calls_with_grad == 0
    with pytest.raises(ValueError):
        policy.select_action(np.array([np.nan] * spec.obs_dim), explore=False)
    with pytest.raises(ValueError):
        policy.select_action(np.zeros(spec.obs_dim + 1), explore=False)


def test_select_action_checks_returned_dimension() -> None:
    class WrongDim(ConstantBaseline):
        def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
            return ActionResult(u=np.zeros(self.spec.action_dim + 1, dtype=np.float32))

    policy = WrongDim(
        make_spec(action_dim=2),
        ConstantBaseline.config_class(),
        np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="shape"):
        policy.select_action(np.zeros(3), explore=False)


def test_noise_only_when_exploring() -> None:
    policy = TinyActorCriticPolicy(
        make_spec(action_dim=5),
        small_ddpg_config(exploration_sigma=0.3),
        np.random.default_rng(0),
    )
    obs = np.full(3, 0.2, dtype=np.float32)
    greedy = policy.select_action(obs, explore=False).u
    np.testing.assert_array_equal(greedy, policy.select_action(obs, explore=False).u)
    assert not np.allclose(greedy, policy.select_action(obs, explore=True).u)


def test_update_waits_for_learning_starts() -> None:
    config = small_ddpg_config()
    policy = TinyActorCriticPolicy(make_spec(), config, np.random.default_rng(0))
    assert isinstance(policy.buffer, ReplayBuffer)
    assert policy.buffer.capacity == config.buffer_size
    assert policy.update() is None  # empty buffer
    for k in range(config.learning_starts):
        assert not policy.ready_to_learn
        assert policy.update() is None
        policy.observe(
            Transition([0.1 * k, 0.0, 0.0], [0.0], float(k), [0.0, 0.0, 0.0])
        )
    assert policy.ready_to_learn
    stats = policy.update()
    assert isinstance(stats, UpdateStats)


def test_frozen_blocks_observe_and_update() -> None:
    config = small_ddpg_config()
    policy = TinyActorCriticPolicy(make_spec(), config, np.random.default_rng(0))
    policy.set_frozen(True)
    for _ in range(config.learning_starts + 5):
        policy.observe(Transition([0.0] * 3, [0.0], 1.0, [0.0] * 3))
    assert len(policy.buffer) == 0
    assert policy.update() is None


def test_end_period_decay_and_floor() -> None:
    config = small_ddpg_config(
        exploration_sigma=0.4, exploration_sigma_decay=0.5, exploration_sigma_min=0.15
    )
    policy = TinyActorCriticPolicy(make_spec(), config, np.random.default_rng(0))
    assert policy.exploration_scale() == pytest.approx(0.4)
    policy.end_period()
    assert policy.exploration_scale() == pytest.approx(0.2)
    policy.end_period()
    assert policy.exploration_scale() == pytest.approx(0.15)
    assert config.exploration_sigma == pytest.approx(0.4)  # config stays immutable


def test_baseline_has_no_buffer_and_no_exploration_scale() -> None:
    policy = ConstantBaseline(
        make_spec(), ConstantBaseline.config_class(), np.random.default_rng(0)
    )
    assert policy.buffer is None
    assert not policy.ready_to_learn
    assert policy.exploration_scale() is None
    policy.observe(Transition([0.0] * 3, [0.0], 1.0, [0.0] * 3))
    assert policy.update() is None


def test_unimplemented_update_raises() -> None:
    class NoUpdate(TinyActorCriticPolicy):
        _update = BaseAgentAlgorithm._update

    policy = NoUpdate(make_spec(), small_ddpg_config(), np.random.default_rng(0))
    for _ in range(policy.config.learning_starts):
        policy.observe(Transition([0.0] * 3, [0.0], 1.0, [0.0] * 3))
    with pytest.raises(NotImplementedError):
        policy.update()


# Contract helper ----------------------------------------------------------------


def test_contract_helper_accepts_learning_policy(spec: PolicySpec) -> None:
    policy = TinyActorCriticPolicy(spec, small_ddpg_config(), np.random.default_rng(1))
    assert_policy_contract(policy, spec, n_steps=60)


def test_contract_helper_accepts_baseline(spec: PolicySpec) -> None:
    policy = ConstantBaseline(
        spec, ConstantBaseline.config_class(), np.random.default_rng(1)
    )
    assert_policy_contract(policy, spec, n_steps=20)


def test_contract_helper_detects_learning_while_frozen() -> None:
    class LeakyFrozen(TinyActorCriticPolicy):
        def update(self) -> UpdateStats | None:
            if self.frozen:
                with torch.no_grad():
                    for p in self.actor.parameters():
                        p.add_(1.0)
                return None
            return super().update()

    spec = make_spec()
    with pytest.raises(AssertionError, match="changed while frozen"):
        assert_policy_contract(
            LeakyFrozen(spec, small_ddpg_config(), np.random.default_rng(0)),
            spec,
            n_steps=40,
        )


def test_contract_helper_detects_early_update() -> None:
    class Eager(TinyActorCriticPolicy):
        def update(self) -> UpdateStats | None:
            return UpdateStats(critic_loss=0.0, actor_loss=0.0, mean_q=0.0)

    spec = make_spec()
    with pytest.raises(AssertionError, match="update returned stats"):
        assert_policy_contract(
            Eager(spec, small_ddpg_config(), np.random.default_rng(0)), spec
        )


# Import smoke tests -------------------------------------------------------------


def test_algorithm_package_imports_without_legacy_modules() -> None:
    """A fresh interpreter imports the new modules and agent_algorithm without
    loading any legacy algorithm module (transitional lazy imports)."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import django\n"
        "django.setup()\n"
        "import abmem.services.algorithms.registry\n"
        "import abmem.services.algorithms.agent_algorithm\n"
        "legacy = ['DDPG', 'TD3', 'SAC', 'ppo', 'RDPG', 'actor_critic', 'algorithm_utils']\n"
        "loaded = [m for m in legacy if 'abmem.services.algorithms.' + m in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_import_views_smoke() -> None:
    """Guards the transitional agent_algorithm.py change (requires Django settings)."""
    importlib.import_module("abmem.views")
    importlib.import_module("abmem.services.agent.agent_service")
    importlib.import_module("abmem.services.market.market_service")
