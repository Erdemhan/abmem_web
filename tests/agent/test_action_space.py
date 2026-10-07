"""Tests for abmem.services.agent.action_space."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from abmem.services.agent.action_space import PRICE_QUANTUM, from_price, to_price

LOW = Decimal(0)
HIGH = Decimal(3000)


def test_bounds_map_to_minus_one_and_plus_one() -> None:
    assert to_price(np.array([-1.0, 1.0, 0.0]), LOW, HIGH) == [
        Decimal("0.00"),
        Decimal("3000.00"),
        Decimal("1500.00"),
    ]
    u = from_price([LOW, HIGH, Decimal(1500)], LOW, HIGH)
    assert u.dtype == np.float32
    np.testing.assert_array_equal(u, [-1.0, 1.0, 0.0])


def test_negative_lower_bound() -> None:
    assert to_price(np.array([-1.0, 0.0, 1.0]), Decimal(-100), Decimal(100)) == [
        Decimal("-100.00"),
        Decimal("0.00"),
        Decimal("100.00"),
    ]


def test_quantisation_half_even() -> None:
    # u maps to 0.005 exactly: low=0, high=0.01 -> p = (u + 1) / 2 * 0.01
    prices = to_price(np.array([0.0]), Decimal(0), Decimal("0.01"))
    assert prices == [Decimal("0.00")]  # 0.005 rounds half to even -> 0.00
    prices = to_price(np.array([0.5]), Decimal(0), Decimal("0.02"))
    assert prices == [Decimal("0.02")]  # 0.015 -> 0.02
    for p in to_price(np.linspace(-1, 1, 101), LOW, Decimal("123.45")):
        assert p == p.quantize(PRICE_QUANTUM)


def test_round_trip_within_quantum() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, size=1000)
    back = from_price(to_price(u, LOW, HIGH), LOW, HIGH)
    tol = 2 * float(PRICE_QUANTUM) / float(HIGH - LOW) + 1e-6
    assert np.max(np.abs(back - u)) <= tol


def test_stored_action_reproduces_applied_price() -> None:
    """B6: from_price(to_price(u)) maps back to exactly the same price."""
    rng = np.random.default_rng(1)
    u = rng.uniform(-1, 1, size=200)
    prices = to_price(u, LOW, Decimal(200))
    stored = from_price(prices, LOW, Decimal(200))
    assert to_price(stored, LOW, Decimal(200)) == prices


def test_out_of_range_is_clipped() -> None:
    prices = to_price(np.array([-5.0, 7.0]), LOW, HIGH)
    assert prices == [LOW, HIGH]
    np.testing.assert_array_equal(
        from_price([Decimal(-10), Decimal(4000)], LOW, HIGH), [-1.0, 1.0]
    )


def test_output_within_bounds() -> None:
    low, high = Decimal("10.005"), Decimal("20.005")  # bounds not on the quantum
    for p in to_price(np.linspace(-1, 1, 51), low, high):
        assert low <= p <= high


def test_multidimensional_input_is_flattened() -> None:
    assert len(to_price(np.zeros((2, 3)), LOW, HIGH)) == 6


def test_empty_inputs() -> None:
    assert to_price(np.array([]), LOW, HIGH) == []
    assert from_price([], LOW, HIGH).shape == (0,)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_u_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        to_price(np.array([0.0, bad]), LOW, HIGH)


def test_low_equal_high() -> None:
    with pytest.raises(ValueError):
        from_price([Decimal(5)], Decimal(5), Decimal(5))
    assert to_price(np.array([0.3]), Decimal(5), Decimal(5)) == [Decimal("5.00")]


def test_inverted_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        to_price(np.array([0.0]), HIGH, LOW)
    with pytest.raises(ValueError):
        from_price([Decimal(1)], HIGH, LOW)


def test_non_finite_price_rejected() -> None:
    with pytest.raises(ValueError):
        from_price([Decimal("NaN")], LOW, HIGH)
