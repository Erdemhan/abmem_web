"""Tests for abmem.services.agent.reward (K1 reward)."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from abmem.services.agent.reward import compute_reward
from abmem.services.market.accounting import OfferOutcome, offer_profit


def _o(q, acc_q, acc_p, c_s="4.50", c_v="12.50", price="80.00") -> OfferOutcome:
    return OfferOutcome(
        quantity=Decimal(q),
        offer_price=Decimal(price),
        accepted_quantity=Decimal(acc_q),
        acceptance_price=None if acc_p is None else Decimal(acc_p),
        static_cost_rate=Decimal(c_s),
        variable_cost_rate=Decimal(c_v),
    )


def test_single_plant_equals_g_over_q_pmax() -> None:
    o = _o("200", "150", "90.00")
    r = compute_reward([o], Decimal(200), Decimal(3000))
    assert r == pytest.approx(10725 / (200 * 3000))


def test_second_plant_accepted_gives_nonzero_reward() -> None:
    """B2 regression: the reward must include every plant, not only the first."""
    first = _o("100", "0", None, c_s="0", c_v="0")
    second = _o("100", "100", "50.00", c_s="0", c_v="10.00")
    r = compute_reward([first, second], Decimal(200), Decimal(100))
    assert r != 0.0
    assert r == pytest.approx((100 * 50 - 10 * 100) / (200 * 100))


def test_all_rejected_gives_negative_unclipped_reward() -> None:
    outcomes = [_o("200", "0", None), _o("300", "0", "0")]
    r = compute_reward(outcomes, Decimal(500), Decimal(10))
    assert r == pytest.approx(-(4.5 * 500) / (500 * 10))
    assert r < -0.4  # not clipped to [-1, 0) or similar ranges


def test_unit_invariance() -> None:
    base = [_o("200", "150", "90.00"), _o("100", "0", None, c_s="2.00", c_v="7.00")]
    scaled = [
        _o("200", "150", "9000", c_s="450", c_v="1250", price="8000"),
        _o("100", "0", None, c_s="200", c_v="700", price="8000"),
    ]
    r_base = compute_reward(base, Decimal(300), Decimal(3000))
    r_scaled = compute_reward(scaled, Decimal(300), Decimal(300000))
    assert r_base == r_scaled


def test_returns_python_float_preserved_through_float32() -> None:
    """C1 regression: a small reward must not be truncated to 0."""
    o = _o("1000", "10", "60.00", c_s="0", c_v="50.00")
    r = compute_reward([o], Decimal(1000), Decimal(3000))
    assert type(r) is float
    expected = float(offer_profit(o) / Decimal(1000 * 3000))
    assert r == expected
    assert 0.0 < abs(r) < 1.0
    assert float(np.float32(r)) == pytest.approx(r, rel=1e-6)
    assert float(np.float32(r)) != 0.0


def test_empty_outcomes_raise() -> None:
    with pytest.raises(ValueError):
        compute_reward([], Decimal(1), Decimal(1))


@pytest.mark.parametrize(
    "cap", [Decimal(0), Decimal(-5), Decimal("NaN"), Decimal("Infinity")]
)
def test_invalid_total_capacity_raises(cap: Decimal) -> None:
    with pytest.raises(ValueError):
        compute_reward([_o("1", "0", None)], cap, Decimal(100))


@pytest.mark.parametrize("p_max", [Decimal(0), Decimal(-1), Decimal("NaN")])
def test_invalid_price_cap_raises(p_max: Decimal) -> None:
    with pytest.raises(ValueError):
        compute_reward([_o("1", "0", None)], Decimal(1), p_max)


def test_integer_arguments_are_accepted() -> None:
    # Plant.capacity and Market.upperBidBound are IntegerFields.
    r = compute_reward([_o("200", "150", "90.00")], 200, 3000)  # type: ignore[arg-type]
    assert r == pytest.approx(10725 / 600000)
