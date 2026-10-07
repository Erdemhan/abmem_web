"""Uniform-price merit-order clearing, the default clearing algorithm (PLAN 1.7, 2.10).

Bids are grouped by equal price and processed in ascending price order. A
group whose total quantity fits the remaining demand is fully accepted. The
first group whose total quantity exceeds the remaining demand is split by
equal shares capped at bid quantity (water-filling, OQ-18) and the clearing
stops there. The split is computed with exact rational arithmetic and then
quantised to ``quantum`` with the largest-remainder method, so the accepted
quantities sum exactly to the cleared volume (B4, B8, B11).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from abmem.services.market.clearing import (
    Bid,
    ClearingResult,
    check_decimal,
    sort_token,
)

from .base import BaseClearingAlgorithm
from .config import ClearingConfig

__all__ = ["MeritOrderClearing", "MeritOrderConfig"]

_ZERO = Decimal(0)


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
    ordered = sorted(quantities.items(), key=lambda kv: (kv[1], sort_token(kv[0])))
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
    remainders.sort(key=lambda item: (-item[0], sort_token(item[1])))
    for _, key in remainders[:leftover]:
        units[key] += 1
    return units


@dataclass(frozen=True)
class MeritOrderConfig(ClearingConfig):
    """The merit order has no parameters."""


class MeritOrderClearing(BaseClearingAlgorithm):
    """Uniform-price merit order with water-filling at the marginal price group."""

    key = "MERIT_ORDER"
    label = "Uniform-price merit order"
    description = (
        "Accepts bids in ascending price order until the demand is met. The "
        "marginal price group is split by equal shares and its price is the PTF."
    )
    config_class = MeritOrderConfig

    def clear(
        self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal = Decimal("0.001")
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
            offered_volume)`` holds exactly. ``ptf`` is the price of the highest
            accepted price group and ``marginal_group`` holds the keys of the
            bids with positive quantity at that price.

        Raises:
            ValueError: If a price, quantity or the demand is negative, NaN,
                infinite or not a ``Decimal``, if ``quantum`` is not positive, if
                a quantity or the demand is not a multiple of ``quantum``, or if
                bid keys repeat.
        """
        quantum = check_decimal(quantum, "quantum")
        if quantum == 0:
            raise ValueError("quantum must be positive")
        demand = check_decimal(demand, "demand")
        unit = Fraction(quantum)
        if Fraction(demand) % unit != 0:
            raise ValueError(f"demand {demand} is not a multiple of quantum {quantum}")

        seen: set[Hashable] = set()
        checked: list[Bid] = []
        for bid in bids:
            if bid.key in seen:
                raise ValueError(f"duplicate bid key {bid.key!r}")
            seen.add(bid.key)
            price = check_decimal(bid.price, f"price of bid {bid.key!r}")
            quantity = check_decimal(bid.quantity, f"quantity of bid {bid.key!r}")
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
            marginal = tuple(sorted((bid.key for bid in groups[ptf]), key=sort_token))
        met_demand = sum(accepted.values(), _ZERO)
        return ClearingResult(
            accepted=accepted,
            ptf=ptf,
            met_demand=met_demand,
            offered_volume=offered_volume,
            marginal_group=marginal,
        )
