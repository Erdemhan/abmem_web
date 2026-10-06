"""Pareto dominance helpers for maximisation problems (Deb, 2001).

The HPO objective is a weighted single objective (OQ-11). These helpers only
provide the Pareto front of the evaluated (agent_score, market_score) pairs
for display.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _as_tuple(point: Sequence[float]) -> tuple[float, ...]:
    """Converts a point to a tuple of floats and rejects NaN.

    Args:
        point: Objective vector.

    Returns:
        Tuple of floats.

    Raises:
        ValueError: If a component is NaN.
    """
    values = tuple(float(v) for v in point)
    if any(math.isnan(v) for v in values):
        raise ValueError(f"objective vectors must not contain NaN, got {point!r}")
    return values


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """Tests Pareto dominance for maximisation.

    ``a`` dominates ``b`` if ``a_m >= b_m`` for every objective ``m`` and
    ``a_m > b_m`` for at least one ``m`` (Deb, 2001).

    Args:
        a: First objective vector.
        b: Second objective vector of the same length.

    Returns:
        True if ``a`` dominates ``b``. Equal vectors do not dominate each
        other.

    Raises:
        ValueError: If the lengths differ, the vectors are empty or contain
            NaN.
    """
    ta, tb = _as_tuple(a), _as_tuple(b)
    if len(ta) != len(tb):
        raise ValueError(f"objective vectors differ in length: {len(ta)} != {len(tb)}")
    if not ta:
        raise ValueError("objective vectors must not be empty")
    return all(x >= y for x, y in zip(ta, tb)) and any(x > y for x, y in zip(ta, tb))


def non_dominated_sort(points: Sequence[Sequence[float]]) -> list[int]:
    """Assigns a Pareto rank to every point (fast non-dominated sort, Deb et al., 2002).

    Args:
        points: Objective vectors of equal length.

    Returns:
        Rank per point in input order. Rank 1 is the first (non-dominated)
        front. An empty input returns an empty list.

    Raises:
        ValueError: If the vectors differ in length, are empty or contain NaN.
    """
    pts = [_as_tuple(p) for p in points]
    n = len(pts)
    if n == 0:
        return []
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(pts[i], pts[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(pts[j], pts[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
    ranks = [0] * n
    front = [i for i in range(n) if domination_count[i] == 0]
    rank = 1
    while front:
        next_front: list[int] = []
        for i in front:
            ranks[i] = rank
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        front = next_front
        rank += 1
    return ranks


def pareto_front(points: Sequence[Sequence[float]]) -> list[int]:
    """Returns the indices of the non-dominated points.

    Args:
        points: Objective vectors of equal length.

    Returns:
        Indices (ascending) of all points that no other point dominates.
        Duplicated non-dominated points are all included.

    Raises:
        ValueError: If the vectors differ in length, are empty or contain NaN.
    """
    return [i for i, r in enumerate(non_dominated_sort(points)) if r == 1]
