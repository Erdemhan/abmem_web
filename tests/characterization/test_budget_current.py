"""Characterization tests for the current ``budgetCalculation`` implementation.

Current rule (floats):
    accepted:  growth = acc_amount * acc_price - (staticCost * amount + fuelCost * acc_amount)
    rejected:  growth = -staticCost * amount   (acceptanceAmount == 0)
with ``staticCost = omCost + investmentCost + co2Cost`` and
``variableCost = fuelCost``.
"""

from decimal import Decimal

import pytest

from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offer,
    make_period,
    make_resource,
)

pytestmark = [pytest.mark.characterization, pytest.mark.django_db]


@pytest.fixture
def offer_context():
    """Saved period, agent and a resource with explicit costs."""
    market = make_market()
    resource = make_resource(
        fuel_cost=Decimal("12.50"),
        om_cost=Decimal("1.25"),
        investment_cost=Decimal("2.50"),
        co2_cost=Decimal("0.75"),
    )
    agent = make_agent_with_plants(market, plants=[(resource, 500)])
    period = make_period(market, demand=1000)
    return period, agent, resource


def _budget(offer):
    from abmem.services.market import market_service

    return market_service.budgetCalculation(offer)


def test_static_and_variable_cost_definitions(offer_context):
    _, _, resource = offer_context
    assert resource.staticCost() == Decimal("4.50")
    assert resource.variableCost() == Decimal("12.50")


def test_accepted_offer_growth(offer_context):
    period, agent, resource = offer_context
    offer = make_offer(
        period,
        agent,
        resource,
        amount=200,
        offer_price=Decimal("80.00"),
        acceptance=True,
        acceptance_price=Decimal("90.00"),
        acceptance_amount=150,
    )

    growth = _budget(offer)

    # 150 * 90 - (4.5 * 200 + 12.5 * 150) = 13500 - (900 + 1875) = 10725
    assert isinstance(growth, float)
    assert growth == pytest.approx(10725.0)


def test_rejected_offer_growth(offer_context):
    period, agent, resource = offer_context
    offer = make_offer(
        period,
        agent,
        resource,
        amount=200,
        offer_price=Decimal("80.00"),
        acceptance=False,
        acceptance_price=Decimal("0"),
        acceptance_amount=0,
    )

    growth = _budget(offer)

    # -staticCost * amount = -4.5 * 200
    assert growth == pytest.approx(-900.0)


def test_unset_acceptance_amount_raises(offer_context):
    """Offers left untouched by clearing (acceptanceAmount None) are not handled."""
    period, agent, resource = offer_context
    offer = make_offer(
        period, agent, resource, amount=200, offer_price=Decimal("80.00")
    )

    with pytest.raises(TypeError):
        _budget(offer)
