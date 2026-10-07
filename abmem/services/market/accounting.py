"""Shared profit accounting for cleared offers.

One profit function is used for agent budgets and for the reward signal, so
both always agree (spec ``observation_reward``, K1).

Profit of one offer with offered quantity ``q``, accepted quantity ``q_acc``,
acceptance price ``p_acc``, static cost rate ``c_s`` and variable cost rate
``c_v`` (all rates in TL/MWh):

    g = q_acc * p_acc - (c_s * q + c_v * q_acc)

For a rejected offer (``q_acc == 0``) this reduces to ``g = -c_s * q``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abmem.models import Offer


def _require_finite_decimal(name: str, value: Decimal) -> None:
    """Raise if ``value`` is not a finite ``Decimal``.

    Args:
        name: Field name used in the error message.
        value: Value to check.

    Raises:
        TypeError: If ``value`` is not a ``Decimal``.
        ValueError: If ``value`` is NaN or infinite.
    """
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be a Decimal, got {type(value).__name__}")
    if not value.is_finite():
        raise ValueError(f"{name} must be finite, got {value}")


@dataclass(frozen=True)
class OfferOutcome:
    """Outcome of one cleared offer.

    Attributes:
        quantity: Offered quantity q (MWh).
        offer_price: Offered price (TL/MWh).
        accepted_quantity: Accepted quantity q_acc (MWh), 0 for a rejected offer.
        acceptance_price: Price paid for the accepted quantity (TL/MWh). May be
            None only when ``accepted_quantity`` is 0.
        static_cost_rate: ``Resource.staticCost()`` (omCost + investmentCost +
            co2Cost), TL/MWh, charged on the offered quantity.
        variable_cost_rate: ``Resource.variableCost()`` (fuelCost), TL/MWh,
            charged on the accepted quantity.
    """

    quantity: Decimal
    offer_price: Decimal
    accepted_quantity: Decimal
    acceptance_price: Decimal | None
    static_cost_rate: Decimal
    variable_cost_rate: Decimal

    def __post_init__(self) -> None:
        """Validate the outcome.

        Raises:
            TypeError: If a field is not a ``Decimal`` (``acceptance_price``
                may also be None).
            ValueError: If a field is non-finite, a quantity is negative, or
                ``acceptance_price`` is None while ``accepted_quantity > 0``.
        """
        _require_finite_decimal("quantity", self.quantity)
        _require_finite_decimal("offer_price", self.offer_price)
        _require_finite_decimal("accepted_quantity", self.accepted_quantity)
        _require_finite_decimal("static_cost_rate", self.static_cost_rate)
        _require_finite_decimal("variable_cost_rate", self.variable_cost_rate)
        if self.acceptance_price is not None:
            _require_finite_decimal("acceptance_price", self.acceptance_price)
        if self.quantity < 0:
            raise ValueError(f"quantity must be >= 0, got {self.quantity}")
        if self.accepted_quantity < 0:
            raise ValueError(
                f"accepted_quantity must be >= 0, got {self.accepted_quantity}"
            )
        if self.accepted_quantity > 0 and self.acceptance_price is None:
            raise ValueError(
                "acceptance_price is None although accepted_quantity > 0 "
                "(inconsistent outcome)"
            )

    @property
    def accepted(self) -> bool:
        """Whether any quantity of the offer was accepted.

        Returns:
            True if ``accepted_quantity > 0``.
        """
        return self.accepted_quantity > 0


def offer_profit(o: OfferOutcome) -> Decimal:
    """Profit g of one cleared offer in exact Decimal arithmetic.

    ``g = q_acc * p_acc - (c_s * q + c_v * q_acc)``. For ``q_acc == 0`` this is
    ``-c_s * q``.

    Args:
        o: Validated offer outcome.

    Returns:
        Profit in TL as a ``Decimal``.
    """
    static_cost = o.static_cost_rate * o.quantity
    if not o.accepted:
        return -static_cost
    # acceptance_price is not None here, guaranteed by OfferOutcome validation.
    assert o.acceptance_price is not None
    revenue = o.accepted_quantity * o.acceptance_price
    variable_cost = o.variable_cost_rate * o.accepted_quantity
    return revenue - (static_cost + variable_cost)


def _to_decimal(name: str, value: object) -> Decimal:
    """Convert an integer or Decimal model field to ``Decimal``.

    Args:
        name: Field name used in the error message.
        value: Field value.

    Returns:
        The value as a ``Decimal``.

    Raises:
        ValueError: If ``value`` is None.
        TypeError: If ``value`` is neither ``int`` nor ``Decimal``.
    """
    if value is None:
        raise ValueError(f"{name} is None, the offer has not been cleared")
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise TypeError(f"{name} must be int or Decimal, got {type(value).__name__}")
    return Decimal(value)


def outcome_from_offer(offer: Offer) -> OfferOutcome:
    """Build an ``OfferOutcome`` from a cleared ``Offer`` instance.

    The offer's resource must already be loaded (or assigned). No database
    query is issued beyond what accessing ``offer.resource`` implies.

    Args:
        offer: Cleared offer with ``acceptanceAmount`` set.

    Returns:
        The corresponding ``OfferOutcome``.

    Raises:
        ValueError: If ``amount`` or ``acceptanceAmount`` is None, or the
            outcome is inconsistent.
    """
    resource = offer.resource
    accepted_quantity = _to_decimal("acceptanceAmount", offer.acceptanceAmount)
    acceptance_price = (
        None
        if offer.acceptancePrice is None
        else _to_decimal("acceptancePrice", offer.acceptancePrice)
    )
    return OfferOutcome(
        quantity=_to_decimal("amount", offer.amount),
        offer_price=_to_decimal("offerPrice", offer.offerPrice),
        accepted_quantity=accepted_quantity,
        acceptance_price=acceptance_price,
        static_cost_rate=_to_decimal("staticCost", resource.staticCost()),
        variable_cost_rate=_to_decimal("variableCost", resource.variableCost()),
    )
