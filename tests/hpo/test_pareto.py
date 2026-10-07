"""Tests for abmem.services.hpo.pareto (maximisation)."""

from __future__ import annotations

import math

import pytest

from abmem.services.hpo.pareto import dominates, non_dominated_sort, pareto_front


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ((2, 2), (1, 1), True),
        ((2, 1), (1, 1), True),
        ((1, 1), (1, 1), False),  # tie: no dominance
        ((2, 0), (0, 2), False),  # trade-off
        ((1, 1), (2, 1), False),
        ((0, -math.inf), (-1, -math.inf), True),
        ((5,), (4,), True),
    ],
)
def test_dominates(a: tuple, b: tuple, expected: bool) -> None:
    assert dominates(a, b) is expected


def test_dominates_is_asymmetric() -> None:
    assert dominates((3, 2), (1, 2))
    assert not dominates((1, 2), (3, 2))


def test_dominates_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        dominates((1, 2), (1,))
    with pytest.raises(ValueError):
        dominates((), ())
    with pytest.raises(ValueError):
        dominates((math.nan, 1), (0, 0))


def test_ranks_on_hand_example() -> None:
    points = [(4, 1), (1, 4), (3, 3), (2, 2), (1, 1), (2, 2), (0, 0)]
    assert non_dominated_sort(points) == [1, 1, 1, 2, 3, 2, 4]
    assert pareto_front(points) == [0, 1, 2]


def test_empty_and_single_point() -> None:
    assert non_dominated_sort([]) == []
    assert pareto_front([]) == []
    assert non_dominated_sort([(1.0, 2.0)]) == [1]
    assert pareto_front([(1.0, 2.0)]) == [0]


def test_duplicate_points_all_on_front() -> None:
    assert pareto_front([(1, 1), (1, 1), (0, 0)]) == [0, 1]


def test_sort_rejects_mixed_lengths() -> None:
    with pytest.raises(ValueError):
        non_dominated_sort([(1, 2), (1, 2, 3)])
