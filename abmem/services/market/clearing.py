"""Pure uniform-price merit-order market clearing (PLAN 1.7, B4, B8, B11, OQ-18).

The module has no ORM access. Bids are grouped by equal price and processed in
ascending price order. A group whose total quantity fits the remaining demand
is fully accepted. The first group whose total quantity exceeds the remaining
demand is split by equal shares capped at bid quantity (water-filling) and the
clearing stops there. The split is computed with exact rational arithmetic and
then quantised to ``quantum`` with the largest-remainder method, so the
accepted quantities sum exactly to the cleared volume.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

from ...models.enums import MarketStrategy

__all__ = ["Bid", "ClearingResult", "acceptance_prices", "clear_market"]

_ZERO = Decimal(0)


@dataclass(frozen=True)
class Bid:
    """A single supply bid entering the clearing.

    Attributes:
        key: Identifier of the bid (the index of the offer in the offer list).
        price: Offered price per MWh.
        quantity: Offered quantity in MWh.
    """

    key: Hashable
    price: Decimal
    quantity: Decimal


@dataclass(frozen=True)
class ClearingResult:
    """Outcome of one clearing.

    Attributes:
        accepted: Accepted quantity per bid key. Every bid has an entry, rejected
            bids map to ``Decimal(0)``.
        ptf: Price of the highest accepted price group, ``None`` when nothing
            is accepted.
        met_demand: Sum of the accepted quantities.
        offered_volume: Sum of all offered quantities (the current semantics of
            ``Period.marketVolume``).
        marginal_group: Keys of the bids with positive quantity whose price
            equals ``ptf``, in deterministic key order. Empty when ``ptf`` is
            ``None``.
    """

    accepted: Mapping[Hashable, Decimal]
    ptf: Decimal | None
    met_demand: Decimal
    offered_volume: Decimal
    marginal_group: tuple[Hashable, ...] = field(default_factory=tuple)


def _sort_token(key: Hashable) -> tuple[int, Any]:
    """Return a sort token that orders keys deterministically.

    Numeric keys are ordered numerically, every other key by ``repr``. This
    keeps the ordering independent of the input order of the bids.

    Args:
        key: A bid key.

    Returns:
        A tuple usable as a sort key across mixed key types.
    """
    if isinstance(key, (int, float, Decimal, Fraction)) and not isinstance(key, bool):
        return (0, key)
    return (1, repr(key))


def _check_decimal(value: Any, name: str) -> Decimal:
    """Validate that ``value`` is a finite, non-negative ``Decimal``.

    Args:
        value: Value to check.
        name: Name used in the error message.

    Returns:
        The value as a ``Decimal``.

    Raises:
        ValueError: If the value is not a ``Decimal``/``int``, is NaN or
            infinite, or is negative.
    """
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise ValueError(
            f"{name} must be a Decimal, got {type(value).__name__}: {value!r}"
        )
    value = Decimal(value)
    if not value.is_finite():
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return value


def _water_fill(
    quantities: Mapping[Hashable, Fraction], volume: Fraction
) -> dict[Hashable, Fraction]:
    """Split ``volume`` equally among bids, capping each bid at its quantity.

    Bids are visited in ascending quantity. While the smallest remaining bid
    fits the current equal share it is fully accepted and the rest of the
    volume is re-split among the remaining bids. Once the smallest remaining
    bid exceeds the equal share, every remaining bid receives that share.

    Args:
        quantities: Positive quantity per bid key.
        volume: Volume to split, ``0 <= volume < sum(quantities)``.

    Returns:
        Exact allocation per bid key. The allocations sum to ``volume``.
    """
    ordered = sorted(quantities.items(), key=lambda kv: (kv[1], _sort_token(kv[0])))
    allocation: dict[Hashable, Fraction] = {}
    remaining = volume
    for position, (key, quantity) in enumerate(ordered):
        share = remaining / (len(ordered) - position)
        if quantity <= share:
            allocation[key] = quantity
            remaining -= quantity
            continue
        for other_key, _ in ordered[position:]:
            allocation[other_key] = share
        break
    return allocation


def _largest_remainder(
    allocation: Mapping[Hashable, Fraction], total_units: int, unit: Fraction
) -> dict[Hashable, int]:
    """Round exact allocations to whole quanta with the largest-remainder method.

    Args:
        allocation: Exact allocation per key.
        total_units: Number of quanta the rounded allocations must sum to.
        unit: Size of one quantum.

    Returns:
        Number of quanta per key, summing to ``total_units``. Leftover quanta go
        to the keys with the largest fractional parts, ties broken by ascending
        key.
    """
    units: dict[Hashable, int] = {}
    remainders: list[tuple[Fraction, Hashable]] = []
    for key, value in allocation.items():
        scaled = value / unit
        whole = scaled.numerator // scaled.denominator
        units[key] = whole
        remainders.append((scaled - whole, key))
    leftover = total_units - sum(units.values())
    remainders.sort(key=lambda item: (-item[0], _sort_token(item[1])))
    for _, key in remainders[:leftover]:
        units[key] += 1
    return units


def clear_market(
    bids: Sequence[Bid], demand: Decimal, quantum: Decimal = Decimal("0.001")
) -> ClearingResult:
    """Clear the market with a uniform-price merit order.

    Bids are grouped by equal price in ascending order. A group whose total
    quantity fits the remaining demand ``D_rem`` is fully accepted. The first
    group whose total exceeds ``D_rem`` is split by equal share capped at bid
    quantity (water-filling, OQ-18) and the loop stops. The split is quantised
    to ``quantum`` with the largest-remainder method so that its sum equals
    ``D_rem`` exactly.

    Args:
        bids: Bids to clear. Keys must be unique.
        demand: Demand to cover in MWh.
        quantum: Resolution of accepted quantities. ``demand`` and every bid
            quantity must be integer multiples of it.

    Returns:
        The clearing result. ``sum(accepted.values()) == min(demand,
        offered_volume)`` holds exactly.

    Raises:
        ValueError: If a price, quantity or the demand is negative, NaN,
            infinite or not a ``Decimal``, if ``quantum`` is not positive, if
            a quantity or the demand is not a multiple of ``quantum``, or if
            bid keys repeat.

    Example:
        >>> r = clear_market([Bid(0, Decimal(60), Decimal(100)),
        ...                   Bid(1, Decimal(60), Decimal(10))], Decimal(60))
        >>> r.accepted[0], r.accepted[1]
        (Decimal('50.000'), Decimal('10'))
    """
    quantum = _check_decimal(quantum, "quantum")
    if quantum == 0:
        raise ValueError("quantum must be positive")
    demand = _check_decimal(demand, "demand")
    unit = Fraction(quantum)
    if Fraction(demand) % unit != 0:
        raise ValueError(f"demand {demand} is not a multiple of quantum {quantum}")

    seen: set[Hashable] = set()
    checked: list[Bid] = []
    for bid in bids:
        if bid.key in seen:
            raise ValueError(f"duplicate bid key {bid.key!r}")
        seen.add(bid.key)
        price = _check_decimal(bid.price, f"price of bid {bid.key!r}")
        quantity = _check_decimal(bid.quantity, f"quantity of bid {bid.key!r}")
        if Fraction(quantity) % unit != 0:
            raise ValueError(
                f"quantity {quantity} of bid {bid.key!r} is not a multiple of quantum {quantum}"
            )
        checked.append(Bid(bid.key, price, quantity))

    accepted: dict[Hashable, Decimal] = {bid.key: _ZERO for bid in checked}
    offered_volume = sum((bid.quantity for bid in checked), _ZERO)

    groups: dict[Decimal, list[Bid]] = defaultdict(list)
    for bid in checked:
        if bid.quantity > 0:
            groups[bid.price].append(bid)

    remaining = Fraction(demand)
    ptf: Decimal | None = None
    for price in sorted(groups):
        if remaining <= 0:
            break
        group = groups[price]
        total = sum((Fraction(bid.quantity) for bid in group), Fraction(0))
        ptf = price
        if total <= remaining:
            for bid in group:
                accepted[bid.key] = bid.quantity
            remaining -= total
            continue
        exact = _water_fill(
            {bid.key: Fraction(bid.quantity) for bid in group}, remaining
        )
        total_units = remaining / unit
        units = _largest_remainder(
            exact, total_units.numerator // total_units.denominator, unit
        )
        for key, count in units.items():
            accepted[key] = Decimal(count) * quantum
        remaining = Fraction(0)
        break

    marginal: tuple[Hashable, ...] = ()
    if ptf is not None:
        marginal = tuple(sorted((bid.key for bid in groups[ptf]), key=_sort_token))
    met_demand = sum(accepted.values(), _ZERO)
    return ClearingResult(
        accepted=accepted,
        ptf=ptf,
        met_demand=met_demand,
        offered_volume=offered_volume,
        marginal_group=marginal,
    )


def acceptance_prices(
    result: ClearingResult, bids: Sequence[Bid], strategy: MarketStrategy
) -> dict[Hashable, Decimal | None]:
    """Return the price paid per accepted MWh for every bid.

    Args:
        result: Result of :func:`clear_market` for ``bids``.
        bids: The bids that were cleared.
        strategy: ``PAYASBID`` pays the bid price, ``PAYASPTF`` pays
            ``result.ptf``.

    Returns:
        Price per bid key. ``None`` for rejected bids (accepted quantity 0).

    Raises:
        ValueError: If ``strategy`` is not a known market strategy.
        KeyError: If a bid key is missing from ``result.accepted``.
    """
    if strategy == MarketStrategy.PAYASBID:
        pay_bid = True
    elif strategy == MarketStrategy.PAYASPTF:
        pay_bid = False
    else:
        raise ValueError(f"unknown market strategy {strategy!r}")
    prices: dict[Hashable, Decimal | None] = {}
    for bid in bids:
        if result.accepted[bid.key] > 0:
            prices[bid.key] = bid.price if pay_bid else result.ptf
        else:
            prices[bid.key] = None
    return prices
