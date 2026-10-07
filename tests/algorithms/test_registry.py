"""Tests of the registry, the BaseAgentAlgorithm template and the contract helper.

The registry discovers the algorithm classes from the subclasses of
BaseAgentAlgorithm (PLAN 2.10). All eight existing algorithms keep their
former enum names as keys.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from django.core.exceptions import ImproperlyConfigured

from abmem.services.algorithms import registry
from abmem.services.algorithms.algorithm_config import (
    DDPGConfig,
    FixedMarkupConfig,
    MarginalCostConfig,
    PPOConfig,
    RandomConfig,
    RDPGConfig,
    SACConfig,
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
from abmem.services.plugin_registry import PluginDiscoveryError
from tests.algorithms.conftest import (
    ConstantBaseline,
    TinyActorCriticPolicy,
    make_spec,
    small_ddpg_config,
)
from tests.algorithms.contract import assert_policy_contract

# Registry ---------------------------------------------------------------------

EXPECTED: dict[str, tuple[str, str, type, bool]] = {
    "DDPG": ("abmem.services.algorithms.DDPG", "DDPGAlgorithm", DDPGConfig, True),
    "TD3": ("abmem.services.algorithms.TD3", "TD3Algorithm", TD3Config, True),
    "SAC": ("abmem.services.algorithms.SAC", "SACAlgorithm", SACConfig, True),
    "PPO": ("abmem.services.algorithms.ppo", "PPOAlgorithm", PPOConfig, True),
    "RDPG": ("abmem.services.algorithms.RDPG", "RDPGAlgorithm", RDPGConfig, True),
    "RANDOM": (
        "abmem.services.algorithms.baselines",
        "RandomPolicy",
        RandomConfig,
        False,
    ),
    "MARGINAL_COST": (
        "abmem.services.algorithms.baselines",
        "MarginalCostPolicy",
        MarginalCostConfig,
        False,
    ),
    "FIXED_MARKUP": (
        "abmem.services.algorithms.baselines",
        "FixedMarkupPolicy",
        FixedMarkupConfig,
        False,
    ),
}


def test_exactly_the_eight_existing_keys_are_registered() -> None:
    assert set(registry.AGENT_ALGORITHMS.classes()) == set(EXPECTED)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_get_algorithm_class_resolves(key: str) -> None:
    module, name, config_cls, learning = EXPECTED[key]
    cls = registry.get_algorithm_class(key)
    assert cls is getattr(importlib.import_module(module), name)
    assert cls.key == key
    assert cls.label and cls.description
    assert cls.config_class is config_cls
    assert registry.get_config_class(key) is config_cls
    assert cls.is_learning is learning
    assert registry.is_baseline(key) is (not learning)
    assert registry.is_known(key)


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_default_config(key: str) -> None:
    config = registry.default_config(key)
    assert type(config) is EXPECTED[key][2]
    assert config == EXPECTED[key][2]()


def test_specific_default_config() -> None:
    assert registry.default_config("FIXED_MARKUP") == FixedMarkupConfig(markup=0.10)


def test_available_algorithms_order_learning_first_then_by_label() -> None:
    infos = registry.available_algorithms()
    assert [info.key for info in infos] == [
        "DDPG",
        "PPO",
        "RDPG",
        "SAC",
        "TD3",
        "FIXED_MARKUP",
        "MARGINAL_COST",
        "RANDOM",
    ]
    assert all(info.is_learning for info in infos[:5])
    assert all(info.is_baseline for info in infos[5:])
    assert next(i for i in infos if i.key == "PPO").is_on_policy
    assert registry.label_for("MARGINAL_COST") == "Marginal cost"


@pytest.mark.parametrize("bad", [99, "ddpg", None, "", "UNKNOWN", 0])
def test_unknown_key(bad: object) -> None:
    assert not registry.is_known(bad)
    assert not registry.is_baseline(bad)
    for func in (
        registry.get_config_class,
        registry.default_config,
        registry.get_algorithm_class,
    ):
        with pytest.raises(KeyError, match="known keys"):
            func(bad)  # type: ignore[arg-type]
    assert registry.label_for("UNKNOWN") == "UNKNOWN"


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_build_policy_with_none_params_satisfies_contract(key: str) -> None:
    spec = make_spec(action_dim=2)
    policy = registry.build_policy(key, spec, None, np.random.default_rng(0))
    assert policy.key == key
    assert policy.config == registry.default_config(key)
    assert_policy_contract(policy, spec)


def test_build_policy_uses_from_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registry, "get_algorithm_class", lambda t: TinyActorCriticPolicy
    )
    spec = make_spec(action_dim=2)
    policy = registry.build_policy(
        "DDPG", spec, {"gamma": 0.9, "hidden_sizes": [8]}, np.random.default_rng(0)
    )
    assert isinstance(policy, TinyActorCriticPolicy)
    assert policy.config == DDPGConfig(gamma=0.9, hidden_sizes=(8,))
    default = registry.build_policy("DDPG", spec, None, np.random.default_rng(0))
    assert default.config == DDPGConfig()
    with pytest.raises(ValueError, match="NOISE_STD"):
        registry.build_policy(
            "DDPG", spec, {"NOISE_STD": 0.6}, np.random.default_rng(0)
        )


def test_duplicate_key_names_both_classes() -> None:
    with pytest.raises(ImproperlyConfigured) as info:

        class Clash(TinyActorCriticPolicy):
            key = "DDPG"

    message = str(info.value)
    assert "duplicate agent algorithm key 'DDPG'" in message
    assert "DDPGAlgorithm" in message and "Clash" in message
    assert registry.get_algorithm_class("DDPG").__name__ == "DDPGAlgorithm"


def test_abstract_subclass_is_skipped() -> None:
    class AbstractPlugin(BaseAgentAlgorithm):
        key = "TEST_ABSTRACT_PLUGIN"
        is_learning = False
        is_on_policy = False
        config_class = RandomConfig

    assert not registry.is_known("TEST_ABSTRACT_PLUGIN")


def test_subclass_inheriting_a_key_is_not_registered() -> None:
    ddpg = registry.get_algorithm_class("DDPG")

    class Variant(ddpg):  # type: ignore[misc, valid-type]
        pass

    assert registry.get_algorithm_class("DDPG") is ddpg


@pytest.mark.parametrize("bad_key", ["lower", "1ABC", "WITH SPACE", "A" * 65])
def test_invalid_key_format(bad_key: str) -> None:
    with pytest.raises(ImproperlyConfigured, match="invalid key"):
        type(
            "BadKey",
            (ConstantBaseline,),
            {"key": bad_key, "__module__": __name__},
        )
    assert not registry.is_known(bad_key)


def test_registered_class_needs_config_class_and_is_learning() -> None:
    with pytest.raises(ImproperlyConfigured, match="config_class"):
        type(
            "NoConfig",
            (ConstantBaseline,),
            {"key": "TEST_NO_CONFIG", "config_class": None, "__module__": __name__},
        )
    assert not registry.is_known("TEST_NO_CONFIG")


# Discovery of plugin modules ---------------------------------------------------

PLUGIN_SOURCE = """
import numpy as np

from abmem.services.algorithms.algorithm_config import RandomConfig
from abmem.services.algorithms.base import ActionResult, BaseAgentAlgorithm


class MidPricePolicy(BaseAgentAlgorithm):
    key = "TEST_MID_PRICE"
    label = "Mid price (test)"
    description = "Bids the middle of the bid bounds."
    is_learning = False
    is_on_policy = False
    config_class = RandomConfig

    def select_action(self, obs, explore):
        return ActionResult(u=np.zeros(self.spec.action_dim, dtype=np.float32))
"""


@pytest.fixture
def plugin_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Yield the name of a fresh importable package under ``tmp_path``.

    Modules imported from it and the keys they register are removed afterwards.
    """
    name = f"abmem_test_plugins_{uuid.uuid4().hex[:8]}"
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(registry.AGENT_ALGORITHMS.classes())
    try:
        yield name
    finally:
        for key in set(registry.AGENT_ALGORITHMS.classes()) - before:
            registry.AGENT_ALGORITHMS.unregister(key)
        for module in [m for m in sys.modules if m.startswith(name)]:
            del sys.modules[module]


def test_plugin_module_in_a_package_is_discovered(
    plugin_package: str, tmp_path: Path
) -> None:
    package = tmp_path / plugin_package
    (package / "mid_price.py").write_text(textwrap.dedent(PLUGIN_SOURCE), "utf-8")
    (package / "_private.py").write_text("raise RuntimeError('skipped')\n", "utf-8")
    assert not registry.is_known("TEST_MID_PRICE")

    registry.AGENT_ALGORITHMS.discover_package(plugin_package)

    assert registry.is_known("TEST_MID_PRICE")
    assert registry.is_baseline("TEST_MID_PRICE")
    assert "TEST_MID_PRICE" in [i.key for i in registry.available_algorithms()]
    spec = make_spec(action_dim=2)
    policy = registry.build_policy(
        "TEST_MID_PRICE", spec, None, np.random.default_rng(0)
    )
    assert_policy_contract(policy, spec, n_steps=5)


def test_failing_plugin_module_is_reported_with_traceback(
    plugin_package: str, tmp_path: Path
) -> None:
    package = tmp_path / plugin_package
    (package / "broken.py").write_text("raise RuntimeError('boom 42')\n", "utf-8")
    with pytest.raises(PluginDiscoveryError) as info:
        registry.AGENT_ALGORITHMS.discover_package(plugin_package)
    message = str(info.value)
    assert f"{plugin_package}.broken" in message
    assert "Traceback" in message and "boom 42" in message


def test_discovery_is_lazy_and_finds_every_algorithm_in_a_fresh_process() -> None:
    """``django.setup`` and importing the registry import no algorithm module and
    no torch. The first query imports the package and finds all eight keys."""
    import subprocess

    code = (
        "import sys\n"
        "import django\n"
        "django.setup()\n"
        "import abmem.models\n"
        "from abmem.services.algorithms import registry\n"
        "assert 'torch' not in sys.modules, 'torch imported at startup'\n"
        "assert 'abmem.services.algorithms.DDPG' not in sys.modules\n"
        "print(','.join(sorted(i.key for i in registry.available_algorithms())))\n"
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
    assert completed.stdout.strip().split(",") == sorted(EXPECTED)


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
    """A fresh interpreter imports the registry and the runtime services without
    loading any algorithm module eagerly, and the legacy modules are gone."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys\n"
        "import django\n"
        "django.setup()\n"
        "import abmem.services.algorithms.registry\n"
        "import importlib.util\n"
        "import abmem.services.market.market_service\n"
        "import abmem.services.simulation.simulation_service\n"
        "lazy = ['DDPG', 'TD3', 'SAC', 'ppo', 'RDPG']\n"
        "loaded = [m for m in lazy if 'abmem.services.algorithms.' + m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "removed = ['agent_algorithm', 'actor_critic', 'algorithm_utils', 'MPIP']\n"
        "found = [m for m in removed\n"
        "         if importlib.util.find_spec('abmem.services.algorithms.' + m)]\n"
        "assert not found, found\n"
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
    """The views and the runtime services import cleanly (requires Django settings)."""
    importlib.import_module("abmem.views")
    importlib.import_module("abmem.services.agent.agent_service")
    importlib.import_module("abmem.services.market.market_service")
