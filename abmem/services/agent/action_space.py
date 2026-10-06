"""Affine mapping between normalised actions in [-1, 1] and offer prices.

``p = low + (clip(u, -1, 1) + 1) / 2 * (high - low)``, quantised to
``PRICE_QUANTUM`` (``Offer.offerPrice`` has two decimals) and clipped to
``[low, high]``. The inverse is ``u = 2 * (p - low) / (high - low) - 1``.
The runtime stores ``from_price(to_price(u))`` so the stored action equals the
applied price (B6).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Final

import numpy as np

PRICE_QUANTUM: Final[Decimal] = Decimal("0.01")


def _validate_bounds(
    low: Decimal, high: Decimal, *, strict: bool
) -> tuple[Decimal, Decimal]:
    """Validate price bounds.

    Args:
        low: Lower bound.
        high: Upper bound.
        strict: If True, require ``high > low``, else ``high >= low``.

    Returns:
        ``(low, high)`` as ``Decimal``.

    Raises:
        ValueError: If a bound is non-finite or the order is violated.
    """
    low = Decimal(low)
    high = Decimal(high)
    if not (low.is_finite() and high.is_finite()):
        raise ValueError(f"price bounds must be finite, got [{low}, {high}]")
    if high < low or (strict and high == low):
        relation = ">" if strict else ">="
        raise ValueError(f"high must be {relation} low, got [{low}, {high}]")
    return low, high


def to_price(u: np.ndarray, low: Decimal, high: Decimal) -> list[Decimal]:
    """Map normalised actions to quantised offer prices.

    Args:
        u: Actions of any shape, flattened in C order. Values outside [-1, 1]
            are clipped.
        low: Lower bid bound.
        high: Upper bid bound (``>= low``).

    Returns:
        One price per action, each a multiple of ``PRICE_QUANTUM`` within
        ``[low, high]`` (up to the quantisation of the bounds themselves).

    Raises:
        ValueError: If ``u`` contains NaN or Inf, or the bounds are invalid.
    """
    low, high = _validate_bounds(low, high, strict=False)
    actions = np.asarray(u, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(actions)):
        raise ValueError("u contains NaN or Inf")
    clipped = np.clip(actions, -1.0, 1.0)
    span = high - low
    prices: list[Decimal] = []
    for value in clipped:
        price = low + (Decimal(float(value)) + 1) / 2 * span
        price = price.quantize(PRICE_QUANTUM, rounding=ROUND_HALF_EVEN)
        prices.append(min(max(price, low), high))
    return prices


def from_price(prices: Sequence[Decimal], low: Decimal, high: Decimal) -> np.ndarray:
    """Map offer prices back to normalised actions.

    Args:
        prices: Offer prices.
        low: Lower bid bound.
        high: Upper bid bound (``> low``).

    Returns:
        float32 array of shape ``(len(prices),)`` clipped to [-1, 1].

    Raises:
        ValueError: If ``high <= low`` or a price is non-finite.
    """
    low, high = _validate_bounds(low, high, strict=True)
    span = high - low
    actions = np.empty(len(prices), dtype=np.float64)
    for i, p in enumerate(prices):
        price = Decimal(p)
        if not price.is_finite():
            raise ValueError(f"price at index {i} is not finite: {price}")
        actions[i] = float(2 * (price - low) / span - 1)
    return np.clip(actions, -1.0, 1.0).astype(np.float32)
