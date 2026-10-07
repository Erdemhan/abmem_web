"""Clearing data types, the clearing result contract and settlement prices.

The module has no ORM access. It defines the input (:class:`Bid`) and output
(:class:`ClearingResult`) of every clearing algorithm, the contract a result
must satisfy before the engine uses it (:func:`validate_clearing_result`,
PLAN 2.10) and the settlement rule (:func:`acceptance_prices`), which stays
separate from the clearing algorithm.

Clearing algorithms are subclasses of
:class:`~abmem.services.market.clearing_algorithms.base.BaseClearingAlgorithm`.
The default uniform-price merit order (PLAN 1.7, B4, B8, B11, OQ-18) lives in
:mod:`abmem.services.market.clearing_algorithms.merit_order`.
:func:`clear_market` is a thin wrapper around it.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

from ...models.enums import MarketStrategy

__all__ = [
    "Bid",
    "ClearingContractError",
    "ClearingResult",
    "acceptance_prices",
    "check_decimal",
    "clear_market",
    "sort_token",
    "validate_clearing_result",
]

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
        ptf: Market clearing price, ``None`` when nothing is accepted.
        met_demand: Sum of the accepted quantities.
        offered_volume: Sum of all offered quantities (the current semantics of
            ``Period.marketVolume``).
        marginal_group: Keys of the bids that set the price, in deterministic
            key order. Empty when ``ptf`` is ``None``.
    """

    accepted: Mapping[Hashable, Decimal]
    ptf: Decimal | None
    met_demand: Decimal
    offered_volume: Decimal
    marginal_group: tuple[Hashable, ...] = field(default_factory=tuple)


class ClearingContractError(ValueError):
    """A clearing algorithm returned a result that breaks the contract."""


def sort_token(key: Hashable) -> tuple[int, Any]:
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


def check_decimal(value: Any, name: str) -> Decimal:
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


def clear_market(
    bids: Sequence[Bid], demand: Decimal, quantum: Decimal = Decimal("0.001")
) -> ClearingResult:
    """Clear the market with the default uniform-price merit order.

    Thin wrapper around
    :class:`~abmem.services.market.clearing_algorithms.merit_order.MeritOrderClearing`
    (key ``MERIT_ORDER``).

    Args:
        bids: Bids to clear. Keys must be unique.
        demand: Demand to cover in MWh.
        quantum: Resolution of accepted quantities.

    Returns:
        The clearing result.

    Raises:
        ValueError: If an input is invalid (see ``MeritOrderClearing.clear``).

    Example:
        >>> r = clear_market([Bid(0, Decimal(60), Decimal(100)),
        ...                   Bid(1, Decimal(60), Decimal(10))], Decimal(60))
        >>> r.accepted[0], r.accepted[1]
        (Decimal('50.000'), Decimal('10'))
    """
    from .clearing_algorithms.merit_order import MeritOrderClearing

    return MeritOrderClearing().clear(bids, demand, quantum)


def _broken(algorithm: str, rule: str) -> ClearingContractError:
    """Return the contract error of ``algorithm`` for the broken ``rule``."""
    return ClearingContractError(
        f"clearing algorithm {algorithm} broke the result contract: {rule}"
    )


def validate_clearing_result(
    result: Any,
    bids: Sequence[Bid],
    demand: Decimal,
    quantum: Decimal,
    low: Decimal,
    high: Decimal,
    *,
    algorithm: str = "?",
) -> ClearingResult:
    """Check a clearing result against the contract of every clearing algorithm.

    The rules (PLAN 2.10, ``specs/pluggable_algorithms.json``):

    1. ``accepted`` has exactly the bid keys.
    2. Every accepted quantity is a finite ``Decimal`` with
       ``0 <= accepted <= bid quantity`` and a multiple of ``quantum``.
    3. ``met_demand == sum(accepted)`` and ``met_demand <= min(demand,
       offered_volume)``.
    4. ``offered_volume`` equals the sum of the bid quantities.
    5. ``ptf`` is None exactly when ``met_demand == 0``. Otherwise it is a
       finite ``Decimal`` with ``low <= ptf <= high``.
    6. ``marginal_group`` is a tuple of keys of ``accepted`` without repeats.

    Args:
        result: Return value of the algorithm.
        bids: The bids that were cleared.
        demand: The demand that was cleared.
        quantum: Resolution of accepted quantities.
        low: Lower bid bound of the market.
        high: Upper bid bound of the market.
        algorithm: Key of the algorithm, used in the error message.

    Returns:
        ``result`` unchanged.

    Raises:
        ClearingContractError: Naming the algorithm and the first broken rule.
    """
    if not isinstance(result, ClearingResult):
        raise _broken(
            algorithm, f"returned {type(result).__name__}, expected ClearingResult"
        )
    accepted = result.accepted
    if not isinstance(accepted, Mapping):
        raise _broken(algorithm, "accepted is not a mapping")
    bid_keys = [bid.key for bid in bids]
    if set(accepted) != set(bid_keys) or len(accepted) != len(set(bid_keys)):
        missing = [k for k in bid_keys if k not in accepted]
        extra = [k for k in accepted if k not in set(bid_keys)]
        raise _broken(
            algorithm,
            f"accepted must have exactly the bid keys (missing {missing}, "
            f"extra {extra})",
        )
    unit = Fraction(quantum)
    total = _ZERO
    for bid in bids:
        value = accepted[bid.key]
        if (
            isinstance(value, bool)
            or not isinstance(value, Decimal)
            or not value.is_finite()
        ):
            raise _broken(
                algorithm,
                f"accepted quantity of bid {bid.key!r} must be a finite Decimal, "
                f"got {value!r}",
            )
        if value < 0 or value > bid.quantity:
            raise _broken(
                algorithm,
                f"accepted quantity {value} of bid {bid.key!r} is outside "
                f"[0, {bid.quantity}]",
            )
        if Fraction(value) % unit != 0:
            raise _broken(
                algorithm,
                f"accepted quantity {value} of bid {bid.key!r} is not a multiple "
                f"of quantum {quantum}",
            )
        total += value
    offered = sum((bid.quantity for bid in bids), _ZERO)
    if result.offered_volume != offered:
        raise _broken(
            algorithm,
            f"offered_volume {result.offered_volume} differs from the sum of bid "
            f"quantities {offered}",
        )
    if result.met_demand != total:
        raise _broken(
            algorithm,
            f"met_demand {result.met_demand} differs from the sum of accepted "
            f"quantities {total}",
        )
    if total > min(demand, offered):
        raise _broken(
            algorithm,
            f"met_demand {total} exceeds min(demand, offered_volume) = "
            f"{min(demand, offered)}",
        )
    ptf = result.ptf
    if total == 0:
        if ptf is not None:
            raise _broken(
                algorithm, f"ptf must be None when met_demand is 0, got {ptf}"
            )
    else:
        if (
            isinstance(ptf, bool)
            or not isinstance(ptf, (Decimal, int))
            or not Decimal(ptf).is_finite()
        ):
            raise _broken(
                algorithm,
                f"ptf must be a finite Decimal when met_demand > 0, got {ptf!r}",
            )
        if not low <= ptf <= high:
            raise _broken(
                algorithm, f"ptf {ptf} is outside the bid bounds [{low}, {high}]"
            )
    group = result.marginal_group
    if not isinstance(group, tuple) or len(set(group)) != len(group):
        raise _broken(algorithm, "marginal_group must be a tuple without repeated keys")
    stray = [k for k in group if k not in accepted]
    if stray:
        raise _broken(algorithm, f"marginal_group has keys that are not bids: {stray}")
    return result


def acceptance_prices(
    result: ClearingResult, bids: Sequence[Bid], strategy: MarketStrategy
) -> dict[Hashable, Decimal | None]:
    """Return the price paid per accepted MWh for every bid (settlement rule).

    Args:
        result: Clearing result for ``bids``.
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
