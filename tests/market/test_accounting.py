"""Tests for abmem.services.market.accounting (shared profit function)."""

from __future__ import annotations

from decimal import Decimal

import pytest

from abmem.services.market.accounting import (
    OfferOutcome,
    offer_profit,
    outcome_from_offer,
)


def _outcome(
    quantity: str = "200",
    offer_price: str = "80.00",
    accepted_quantity: str = "150",
    acceptance_price: str | None = "90.00",
    static_cost_rate: str = "4.50",
    variable_cost_rate: str = "12.50",
) -> OfferOutcome:
    return OfferOutcome(
        quantity=Decimal(quantity),
        offer_price=Decimal(offer_price),
        accepted_quantity=Decimal(accepted_quantity),
        acceptance_price=(
            None if acceptance_price is None else Decimal(acceptance_price)
        ),
        static_cost_rate=Decimal(static_cost_rate),
        variable_cost_rate=Decimal(variable_cost_rate),
    )


def _unsaved_offer(amount, acceptance_amount, acceptance_price, acceptance=None):
    """Unsaved Offer with an unsaved Resource (characterization costs), no DB."""
    from tests.factories import make_resource

    from abmem.models import Offer

    resource = make_resource(
        fuel_cost=Decimal("12.50"),
        om_cost=Decimal("1.25"),
        investment_cost=Decimal("2.50"),
        co2_cost=Decimal("0.75"),
        save=False,
    )
    return Offer(
        resource=resource,
        amount=amount,
        offerPrice=Decimal("80.00"),
        acceptance=acceptance,
        acceptancePrice=acceptance_price,
        acceptanceAmount=acceptance_amount,
    )


def test_accepted_offer_profit_matches_characterization_case() -> None:
    # 150 * 90 - (4.5 * 200 + 12.5 * 150) = 10725
    assert offer_profit(_outcome()) == Decimal("10725.00")


def test_rejected_offer_profit_is_minus_static_cost() -> None:
    o = _outcome(accepted_quantity="0", acceptance_price="0")
    assert not o.accepted
    assert offer_profit(o) == Decimal("-900.00")


def test_rejected_offer_with_none_acceptance_price() -> None:
    o = _outcome(accepted_quantity="0", acceptance_price=None)
    assert offer_profit(o) == Decimal("-900.00")


def test_payasptf_acceptance_price_differs_from_offer_price() -> None:
    # Under PAYASPTF the accepted quantity is paid the PTF, not the offer price.
    o = _outcome(offer_price="50.00", acceptance_price="120.00")
    # 150 * 120 - (4.5 * 200 + 12.5 * 150) = 18000 - 2775
    assert offer_profit(o) == Decimal("15225.00")


def test_decimal_exactness() -> None:
    o = _outcome(
        quantity="3",
        accepted_quantity="3",
        acceptance_price="0.10",
        static_cost_rate="0.10",
        variable_cost_rate="0.10",
    )
    # 0.3 - (0.3 + 0.3) = -0.3 exactly; floats would give -0.30000000000000004
    result = offer_profit(o)
    assert isinstance(result, Decimal)
    assert result == Decimal("-0.30")


def test_zero_quantity_offer_has_zero_profit() -> None:
    o = _outcome(quantity="0", accepted_quantity="0", acceptance_price=None)
    assert offer_profit(o) == 0


def test_none_acceptance_price_with_accepted_quantity_raises() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        _outcome(acceptance_price=None)


@pytest.mark.parametrize("field", ["quantity", "accepted_quantity"])
def test_negative_quantity_raises(field: str) -> None:
    with pytest.raises(ValueError):
        _outcome(**{field: "-1"})


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_field_raises(value: str) -> None:
    with pytest.raises(ValueError):
        _outcome(static_cost_rate=value)


def test_float_field_raises_type_error() -> None:
    with pytest.raises(TypeError):
        OfferOutcome(
            quantity=200.0,  # type: ignore[arg-type]
            offer_price=Decimal("1"),
            accepted_quantity=Decimal("0"),
            acceptance_price=None,
            static_cost_rate=Decimal("1"),
            variable_cost_rate=Decimal("1"),
        )


@pytest.mark.parametrize(
    ("amount", "acc_amount", "acc_price"),
    [
        (200, 150, Decimal("90.00")),
        (200, 0, Decimal("0")),
        (500, 500, Decimal("33.33")),
    ],
)
def test_parity_with_current_budget_calculation(amount, acc_amount, acc_price) -> None:
    from abmem.services.market import market_service

    offer = _unsaved_offer(amount, acc_amount, acc_price, acceptance=acc_amount > 0)
    expected = market_service.budgetCalculation(offer)
    actual = offer_profit(outcome_from_offer(offer))
    assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_outcome_from_offer_fields() -> None:
    offer = _unsaved_offer(200, 150, Decimal("90.00"), acceptance=True)
    o = outcome_from_offer(offer)
    assert o == OfferOutcome(
        quantity=Decimal(200),
        offer_price=Decimal("80.00"),
        accepted_quantity=Decimal(150),
        acceptance_price=Decimal("90.00"),
        static_cost_rate=Decimal("4.50"),
        variable_cost_rate=Decimal("12.50"),
    )
    assert o.accepted


def test_outcome_from_uncleared_offer_raises() -> None:
    offer = _unsaved_offer(200, None, None)
    with pytest.raises(ValueError, match="not been cleared"):
        outcome_from_offer(offer)
