"""Tests for abmem.services.agent.failstack (PLAN 1.5, K3)."""

from __future__ import annotations

import ast
import random
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abmem.models.enums import AlgorithmType
from abmem.services.agent import failstack
from abmem.services.agent.failstack import (
    FailStackParams,
    FailStackState,
    apply_interventions,
    intervention_price,
    is_failure,
    margin,
    update_counters,
)
from abmem.services.market.accounting import OfferOutcome, offer_profit

LOW = Decimal("0")
HIGH = Decimal("200")


def _o(q, acc_q, acc_p, c_s="4.50", c_v="12.50", price="80.00") -> OfferOutcome:
    return OfferOutcome(
        quantity=Decimal(q),
        offer_price=Decimal(price),
        accepted_quantity=Decimal(acc_q),
        acceptance_price=None if acc_p is None else Decimal(acc_p),
        static_cost_rate=Decimal(c_s),
        variable_cost_rate=Decimal(c_v),
    )


def _params(
    limit: int = 6, delta: str = "15", min_margin: str = "0"
) -> FailStackParams:
    return FailStackParams(
        limit=limit, delta=Decimal(delta), min_margin=Decimal(min_margin)
    )


# --------------------------------------------------------------------------- params


def test_params_valid_defaults() -> None:
    p = _params()
    assert p.limit == 6 and p.delta == Decimal("15") and p.min_margin == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": 0},
        {"limit": -3},
        {"limit": True},
        {"limit": 2.0},
        {"delta": Decimal("-0.01")},
        {"delta": Decimal("NaN")},
        {"delta": Decimal("Infinity")},
        {"min_margin": Decimal("NaN")},
        {"min_margin": Decimal("-Infinity")},
        {"delta": 15.0},
    ],
)
def test_params_invalid(kwargs) -> None:
    base = {"limit": 6, "delta": Decimal("15"), "min_margin": Decimal("0")}
    base.update(kwargs)
    with pytest.raises(ValueError):
        FailStackParams(**base)


def test_from_agent_disabled_returns_none() -> None:
    agent = SimpleNamespace(failstack_enabled=False, algorithm_type=AlgorithmType.DDPG)
    assert FailStackParams.from_agent(agent) is None


def test_from_agent_without_fields_returns_none() -> None:
    assert FailStackParams.from_agent(SimpleNamespace()) is None


@pytest.mark.parametrize(
    "algo",
    [AlgorithmType.RANDOM, AlgorithmType.MARGINAL_COST, AlgorithmType.FIXED_MARKUP, 5],
)
def test_from_agent_baseline_returns_none(algo) -> None:
    agent = SimpleNamespace(failstack_enabled=True, algorithm_type=algo)
    assert FailStackParams.from_agent(agent) is None


def test_from_agent_enabled_uses_defaults() -> None:
    agent = SimpleNamespace(failstack_enabled=True, algorithm_type=AlgorithmType.PPO)
    assert FailStackParams.from_agent(agent) == _params()


def test_from_agent_enabled_without_algorithm_type() -> None:
    agent = SimpleNamespace(
        failstack_enabled=True,
        failstack_limit=3,
        failstack_delta=Decimal("7.50"),
        failstack_min_margin=0.05,
    )
    assert FailStackParams.from_agent(agent) == _params(3, "7.50", "0.05")


def test_from_agent_invalid_values_raise() -> None:
    agent = SimpleNamespace(
        failstack_enabled=True, algorithm_type=AlgorithmType.TD3, failstack_limit=0
    )
    with pytest.raises(ValueError):
        FailStackParams.from_agent(agent)


# --------------------------------------------------------------------------- state


def test_state_zeros() -> None:
    assert FailStackState.zeros(3).counters == [0, 0, 0]


@pytest.mark.parametrize("n", [0, -1, True])
def test_state_zeros_invalid(n) -> None:
    with pytest.raises(ValueError):
        FailStackState.zeros(n)


# --------------------------------------------------------------------------- failure


def test_margin_value() -> None:
    o = _o("200", "150", "90.00")
    g = offer_profit(o)  # 13500 - (900 + 1875) = 10725
    assert margin(o, g) == Decimal("10725") / Decimal("13500")


def test_rejected_is_failure_and_has_no_margin() -> None:
    o = _o("200", "0", None)
    assert margin(o, offer_profit(o)) is None
    assert is_failure(o, offer_profit(o), Decimal("-100")) is True


def test_accepted_profitable_is_success() -> None:
    o = _o("200", "150", "90.00")
    assert is_failure(o, offer_profit(o), Decimal("0")) is False


def test_accepted_loss_is_failure_with_default_threshold() -> None:
    o = _o("200", "150", "10.00")
    assert offer_profit(o) < 0
    assert is_failure(o, offer_profit(o), Decimal("0")) is True


def test_margin_threshold() -> None:
    o = _o("200", "150", "90.00")
    g = offer_profit(o)
    m = margin(o, g)
    assert is_failure(o, g, m) is False  # margin == m_min is not below
    assert is_failure(o, g, m + Decimal("0.0001")) is True


@pytest.mark.parametrize(
    ("c_s", "c_v", "expected"),
    [("0", "0", False), ("1", "0", True), ("0", "5", True)],
)
def test_zero_revenue_oq13(c_s, c_v, expected) -> None:
    o = _o("100", "50", "0.00", c_s=c_s, c_v=c_v)
    g = offer_profit(o)
    assert margin(o, g) is None
    # Failure iff profit < 0, independent of the threshold.
    assert is_failure(o, g, Decimal("0.5")) is expected


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity")])
def test_is_failure_non_finite(bad) -> None:
    o = _o("200", "150", "90.00")
    with pytest.raises(ValueError):
        is_failure(o, bad, Decimal("0"))
    with pytest.raises(ValueError):
        is_failure(o, offer_profit(o), bad)


# --------------------------------------------------------------------------- counters


def test_update_counters_reset_and_increment() -> None:
    s = FailStackState(counters=[2, 5, 0])
    update_counters(s, [True, False, True])
    assert s.counters == [3, 0, 1]


def test_update_counters_length_mismatch() -> None:
    with pytest.raises(ValueError):
        update_counters(FailStackState.zeros(2), [True])


def _run_sequence(seq: str, limit: int) -> list[bool]:
    """Simulate decisions; return the intervened flag of each decision."""
    state = FailStackState.zeros(1)
    params = _params(limit=limit)
    rng = np.random.default_rng(0)
    flags: list[bool] = []
    for ch in seq:
        update_counters(state, [ch == "F"])
        _, f = apply_interventions(
            state, params, [Decimal("150")], Decimal("100"), LOW, HIGH, rng
        )
        flags.append(f[0])
    return flags


def test_sequence_single_intervention_at_ninth_decision() -> None:
    flags = _run_sequence("FFSFFFFFF", limit=6)
    assert flags == [False] * 8 + [True]


def test_interventions_only_after_l_consecutive_failures() -> None:
    flags = _run_sequence("F" * 13, limit=6)
    assert [i for i, f in enumerate(flags) if f] == [5, 11]


def test_limit_one_intervenes_after_every_failure() -> None:
    flags = _run_sequence("FSFF", limit=1)
    assert flags == [True, False, True, True]


def test_counter_reset_after_intervention() -> None:
    state = FailStackState(counters=[6])
    apply_interventions(
        state,
        _params(),
        [Decimal("150")],
        Decimal("100"),
        LOW,
        HIGH,
        np.random.default_rng(1),
    )
    assert state.counters == [0]


def test_no_intervention_without_ptf() -> None:
    state = FailStackState(counters=[9, 0])
    prices = [Decimal("150"), Decimal("20")]
    applied, flags = apply_interventions(
        state, _params(), prices, None, LOW, HIGH, np.random.default_rng(0)
    )
    assert applied == prices and flags == [False, False]
    assert state.counters == [9, 0]


def test_apply_interventions_length_mismatch() -> None:
    with pytest.raises(ValueError):
        apply_interventions(
            FailStackState.zeros(2),
            _params(),
            [Decimal("1")],
            Decimal("100"),
            LOW,
            HIGH,
            np.random.default_rng(0),
        )


def test_multi_plant_independence() -> None:
    state = FailStackState(counters=[6, 5, 0, 7])
    prices = [Decimal("150"), Decimal("151"), Decimal("152"), Decimal("153")]
    applied, flags = apply_interventions(
        state, _params(), prices, Decimal("100"), LOW, HIGH, np.random.default_rng(3)
    )
    assert flags == [True, False, False, True]
    assert applied[1] == prices[1] and applied[2] == prices[2]
    for i in (0, 3):
        assert Decimal("85") <= applied[i] < Decimal("100")
    assert state.counters == [0, 5, 0, 0]


# --------------------------------------------------------------------------- price


def test_intervention_price_bounds_10000_draws() -> None:
    rng = np.random.default_rng(12345)
    ptf = Decimal("100.00")
    draws = [
        intervention_price(ptf, Decimal("15"), LOW, HIGH, rng) for _ in range(10_000)
    ]
    assert all(Decimal("85") <= p < ptf for p in draws)
    assert all(p == p.quantize(Decimal("0.01")) for p in draws)
    assert min(draws) < Decimal("86") and max(draws) > Decimal("99")


def test_intervention_price_ptf_near_lower_bound() -> None:
    rng = np.random.default_rng(7)
    low = Decimal("10")
    ptf = Decimal("12.00")
    draws = [
        intervention_price(ptf, Decimal("15"), low, HIGH, rng) for _ in range(2000)
    ]
    assert all(low <= p < ptf for p in draws)


def test_intervention_price_empty_interval_delta_zero() -> None:
    rng = np.random.default_rng(0)
    assert intervention_price(Decimal("100"), Decimal("0"), LOW, HIGH, rng) == Decimal(
        "100"
    )


@pytest.mark.parametrize("ptf", [Decimal("10"), Decimal("5"), Decimal("-3")])
def test_intervention_price_ptf_at_or_below_low(ptf) -> None:
    rng = np.random.default_rng(0)
    assert intervention_price(ptf, Decimal("15"), Decimal("10"), HIGH, rng) == Decimal(
        "10"
    )


def test_intervention_price_ptf_above_high_clipped() -> None:
    rng = np.random.default_rng(0)
    draws = [
        intervention_price(Decimal("250"), Decimal("15"), LOW, HIGH, rng)
        for _ in range(100)
    ]
    assert all(p == HIGH for p in draws)


@pytest.mark.parametrize(
    ("ptf", "delta", "low", "high"),
    [
        (Decimal("NaN"), Decimal("15"), LOW, HIGH),
        (Decimal("100"), Decimal("Infinity"), LOW, HIGH),
        (Decimal("100"), Decimal("-1"), LOW, HIGH),
        (Decimal("100"), Decimal("15"), Decimal("NaN"), HIGH),
        (Decimal("100"), Decimal("15"), HIGH, LOW),
    ],
)
def test_intervention_price_invalid(ptf, delta, low, high) -> None:
    with pytest.raises(ValueError):
        intervention_price(ptf, delta, low, high, np.random.default_rng(0))


def test_determinism_under_seed() -> None:
    def run(seed: int) -> list[Decimal]:
        rng = np.random.default_rng(seed)
        return [
            intervention_price(Decimal("100"), Decimal("15"), LOW, HIGH, rng)
            for _ in range(50)
        ]

    assert run(42) == run(42)
    assert run(42) != run(43)


# --------------------------------------------------------------------------- unit invariance


def _scale_outcome(o: OfferOutcome, k: Decimal) -> OfferOutcome:
    return OfferOutcome(
        quantity=o.quantity,
        offer_price=o.offer_price * k,
        accepted_quantity=o.accepted_quantity,
        acceptance_price=None if o.acceptance_price is None else o.acceptance_price * k,
        static_cost_rate=o.static_cost_rate * k,
        variable_cost_rate=o.variable_cost_rate * k,
    )


def test_unit_invariance_failure_counters_and_flags() -> None:
    rnd = random.Random(2026)
    k = Decimal(100)
    n_plants = 4
    params = _params(limit=3, min_margin="0.05")
    params_k = FailStackParams(
        limit=3, delta=params.delta * k, min_margin=params.min_margin
    )
    s, s_k = FailStackState.zeros(n_plants), FailStackState.zeros(n_plants)
    rng, rng_k = np.random.default_rng(5), np.random.default_rng(5)
    n_failures = 0
    for _ in range(25):  # 25 periods x 4 plants = 100 random outcomes
        outcomes = []
        for _ in range(n_plants):
            q = Decimal(rnd.randint(1, 300))
            acc_q = Decimal(rnd.choice([0, rnd.randint(1, int(q))]))
            acc_p = None if acc_q == 0 else Decimal(rnd.randint(0, 20000)) / 100
            outcomes.append(
                _o(
                    q,
                    acc_q,
                    acc_p,
                    c_s=str(Decimal(rnd.randint(0, 2000)) / 100),
                    c_v=str(Decimal(rnd.randint(0, 8000)) / 100),
                )
            )
        f = [is_failure(o, offer_profit(o), params.min_margin) for o in outcomes]
        scaled = [_scale_outcome(o, k) for o in outcomes]
        f_k = [is_failure(o, offer_profit(o), params_k.min_margin) for o in scaled]
        assert f == f_k
        n_failures += sum(f)
        update_counters(s, f)
        update_counters(s_k, f_k)
        assert s.counters == s_k.counters
        ptf = Decimal(rnd.randint(0, 20000)) / 100
        prices = [Decimal("150")] * n_plants
        _, flags = apply_interventions(s, params, prices, ptf, LOW, HIGH, rng)
        _, flags_k = apply_interventions(
            s_k, params_k, [p * k for p in prices], ptf * k, LOW * k, HIGH * k, rng_k
        )
        assert flags == flags_k
        assert s.counters == s_k.counters
    assert 0 < n_failures < 100


# --------------------------------------------------------------------------- purity


def test_module_does_not_import_django_or_models_at_module_level() -> None:
    tree = ast.parse(Path(failstack.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert not name.startswith("django")
            assert not name.startswith("abmem.models")
