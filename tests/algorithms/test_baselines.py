"""Tests of the non-learning comparison policies (rl_algorithms.baselines)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from abmem.services.agent.action_space import PRICE_QUANTUM, from_price, to_price
from abmem.services.algorithms import registry
from abmem.services.algorithms.algorithm_config import (
    FixedMarkupConfig,
    MarginalCostConfig,
    RandomConfig,
)
from abmem.services.algorithms.base import PolicySpec, Transition
from abmem.services.algorithms.baselines import (
    FixedMarkupPolicy,
    MarginalCostPolicy,
    RandomPolicy,
    price_to_action,
)
from tests.algorithms.conftest import make_spec
from tests.algorithms.contract import assert_policy_contract


def cost_spec(
    variable: tuple[float, ...],
    static: tuple[float, ...],
    low: float = 0.0,
    high: float = 200.0,
) -> PolicySpec:
    """Return a PolicySpec with the given per-plant costs and bid bounds."""
    return PolicySpec(
        obs_dim=3,
        action_dim=len(variable),
        price_low=low,
        price_high=high,
        plant_variable_costs=variable,
        plant_static_costs=static,
    )


def applied_prices(u: np.ndarray, spec: PolicySpec) -> list[Decimal]:
    """Return the prices the runtime applies for action ``u``."""
    return to_price(u, Decimal(str(spec.price_low)), Decimal(str(spec.price_high)))


OBS = np.zeros(3, dtype=np.float32)


# Contract -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "config"),
    [
        (MarginalCostPolicy, MarginalCostConfig()),
        (FixedMarkupPolicy, FixedMarkupConfig()),
    ],
    ids=["MARGINAL_COST", "FIXED_MARKUP"],
)
def test_cost_policies_satisfy_contract(
    cls: type, config: object, spec: PolicySpec, rng: np.random.Generator
) -> None:
    policy = cls(spec, config, rng)
    assert_policy_contract(policy, spec)


def test_random_policy_non_learning_contract(
    spec: PolicySpec, rng: np.random.Generator
) -> None:
    # RANDOM is random also with explore=False (spec rule), so the greedy
    # determinism check of assert_policy_contract does not apply. The other
    # non-learning invariants are checked here.
    policy = RandomPolicy(spec, RandomConfig(), rng)
    assert policy.buffer is None
    assert policy.exploration_scale() is None
    assert not policy.ready_to_learn
    data_rng = np.random.default_rng(1)
    obs = data_rng.uniform(-1, 1, size=spec.obs_dim).astype(np.float32)
    for explore in (True, False) * 50:
        result = policy.select_action(obs, explore=explore)
        assert result.u.shape == (spec.action_dim,)
        assert result.u.dtype == np.float32
        assert np.all(np.abs(result.u) <= 1.0)
        assert result.log_prob is None and result.value is None
        next_obs = data_rng.uniform(-1, 1, size=spec.obs_dim).astype(np.float32)
        policy.observe(Transition(obs, result.u, 0.5, next_obs))
        assert policy.update() is None
        policy.end_period()
        obs = next_obs
    assert policy.buffer is None
    policy.set_frozen(True)
    assert policy.update() is None
    policy.set_frozen(False)


@pytest.mark.parametrize("algorithm_type", ["FIXED_MARKUP", "MARGINAL_COST", "RANDOM"])
def test_registry_resolves_baselines(algorithm_type: str) -> None:
    cls = registry.get_algorithm_class(algorithm_type)
    assert cls.key == algorithm_type
    assert cls.is_learning is False
    assert registry.is_baseline(algorithm_type)
    assert cls.config_class is registry.get_config_class(algorithm_type)
    policy = registry.build_policy(
        algorithm_type, make_spec(action_dim=2), None, np.random.default_rng(0)
    )
    assert isinstance(policy, cls)


# Marginal cost ------------------------------------------------------------------


def test_marginal_cost_round_trip_within_one_quantum() -> None:
    spec = cost_spec((37.4567, 120.0, 0.0), (5.0, 5.0, 5.0))
    policy = MarginalCostPolicy(spec, MarginalCostConfig(), np.random.default_rng(0))
    u = policy.select_action(OBS, explore=False).u
    for price, cost in zip(applied_prices(u, spec), spec.plant_variable_costs):
        assert abs(price - Decimal(str(cost))) <= PRICE_QUANTUM


def test_marginal_cost_exact_action_values() -> None:
    spec = cost_spec((50.0, 100.0, 150.0), (1.0, 1.0, 1.0))
    policy = MarginalCostPolicy(spec, MarginalCostConfig(), np.random.default_rng(0))
    u = policy.select_action(OBS, explore=True).u
    np.testing.assert_allclose(u, [-0.5, 0.0, 0.5], atol=1e-7)


def test_marginal_cost_ignores_static_cost_and_explore() -> None:
    a = MarginalCostPolicy(
        cost_spec((40.0,), (0.0,)), MarginalCostConfig(), np.random.default_rng(0)
    )
    b = MarginalCostPolicy(
        cost_spec((40.0,), (90.0,)), MarginalCostConfig(), np.random.default_rng(1)
    )
    np.testing.assert_array_equal(
        a.select_action(OBS, explore=True).u, b.select_action(OBS, explore=False).u
    )


def test_cost_above_price_high_clips_to_plus_one() -> None:
    spec = cost_spec((500.0,), (0.0,), low=0.0, high=200.0)
    policy = MarginalCostPolicy(spec, MarginalCostConfig(), np.random.default_rng(0))
    u = policy.select_action(OBS, explore=False).u
    np.testing.assert_array_equal(u, np.array([1.0], dtype=np.float32))
    assert applied_prices(u, spec) == [Decimal("200.00")]


def test_cost_below_price_low_clips_to_minus_one() -> None:
    spec = cost_spec((5.0,), (0.0,), low=20.0, high=200.0)
    policy = MarginalCostPolicy(spec, MarginalCostConfig(), np.random.default_rng(0))
    u = policy.select_action(OBS, explore=False).u
    np.testing.assert_array_equal(u, np.array([-1.0], dtype=np.float32))


# Fixed markup -------------------------------------------------------------------


def test_fixed_markup_default_price() -> None:
    spec = cost_spec((40.0, 70.0), (10.0, 30.0))
    policy = FixedMarkupPolicy(spec, FixedMarkupConfig(), np.random.default_rng(0))
    assert policy.config.markup == pytest.approx(0.10)
    u = policy.select_action(OBS, explore=False).u
    expected = [Decimal("55.00"), Decimal("110.00")]
    for price, target in zip(applied_prices(u, spec), expected):
        assert abs(price - target) <= PRICE_QUANTUM


def test_fixed_markup_custom_markup_exact_action() -> None:
    spec = cost_spec((60.0,), (20.0,), low=0.0, high=200.0)
    config = FixedMarkupConfig.from_dict({"markup": 0.25})
    policy = FixedMarkupPolicy(spec, config, np.random.default_rng(0))
    # (60 + 20) * 1.25 = 100 -> midpoint of [0, 200] -> u = 0.
    np.testing.assert_allclose(policy.select_action(OBS, explore=True).u, [0.0])


def test_fixed_markup_zero_markup_equals_total_cost() -> None:
    spec = cost_spec((30.0,), (20.0,))
    policy = FixedMarkupPolicy(
        spec, FixedMarkupConfig(markup=0.0), np.random.default_rng(0)
    )
    u = policy.select_action(OBS, explore=False).u
    assert abs(applied_prices(u, spec)[0] - Decimal("50")) <= PRICE_QUANTUM


def test_fixed_markup_clips_at_both_bounds() -> None:
    spec = cost_spec((300.0, 1.0), (50.0, 1.0), low=10.0, high=200.0)
    policy = FixedMarkupPolicy(spec, FixedMarkupConfig(), np.random.default_rng(0))
    u = policy.select_action(OBS, explore=False).u
    np.testing.assert_array_equal(u, np.array([1.0, -1.0], dtype=np.float32))


@pytest.mark.parametrize("markup", [-1.0, -2.5])
def test_fixed_markup_config_rejects_markup_le_minus_one(markup: float) -> None:
    with pytest.raises(ValueError, match="markup"):
        FixedMarkupConfig(markup=markup)


def test_fixed_markup_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError):
        FixedMarkupConfig.from_dict({"mark_up": 0.2})


def test_cost_policy_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError):
        FixedMarkupPolicy(
            make_spec(), MarginalCostConfig(), np.random.default_rng(0)  # type: ignore[arg-type]
        )


# Random -------------------------------------------------------------------------


def test_random_reproducible_with_same_seed(spec: PolicySpec) -> None:
    a = RandomPolicy(spec, RandomConfig(), np.random.default_rng(42))
    b = RandomPolicy(spec, RandomConfig(), np.random.default_rng(42))
    for i in range(50):
        explore = bool(i % 2)
        np.testing.assert_array_equal(
            a.select_action(OBS, explore=explore).u,
            b.select_action(OBS, explore=explore).u,
        )


def test_random_differs_across_seeds_and_calls() -> None:
    spec = make_spec(action_dim=3)
    a = RandomPolicy(spec, RandomConfig(), np.random.default_rng(1))
    b = RandomPolicy(spec, RandomConfig(), np.random.default_rng(2))
    first = a.select_action(OBS, explore=False).u
    second = a.select_action(OBS, explore=False).u
    assert not np.array_equal(first, second)
    assert not np.array_equal(first, b.select_action(OBS, explore=False).u)


def test_random_covers_action_range() -> None:
    policy = RandomPolicy(
        make_spec(action_dim=1), RandomConfig(), np.random.default_rng(3)
    )
    draws = np.array(
        [policy.select_action(OBS, explore=True).u[0] for _ in range(2000)]
    )
    assert draws.min() < -0.95 and draws.max() > 0.95
    assert abs(float(draws.mean())) < 0.05


# Price map ----------------------------------------------------------------------


def test_price_to_action_inverts_action_space_map() -> None:
    prices = [Decimal("0.00"), Decimal("12.34"), Decimal("100.00"), Decimal("200.00")]
    expected = from_price(prices, Decimal("0"), Decimal("200"))
    got = price_to_action([float(p) for p in prices], 0.0, 200.0)
    np.testing.assert_allclose(got, expected, atol=1e-7)
    assert got.dtype == np.float32


@pytest.mark.parametrize(
    ("prices", "low", "high"),
    [
        ([1.0], 5.0, 5.0),
        ([1.0], 6.0, 5.0),
        ([float("nan")], 0.0, 1.0),
        ([np.inf], 0.0, 1.0),
    ],
)
def test_price_to_action_rejects_invalid_input(
    prices: list[float], low: float, high: float
) -> None:
    with pytest.raises(ValueError):
        price_to_action(prices, low, high)
