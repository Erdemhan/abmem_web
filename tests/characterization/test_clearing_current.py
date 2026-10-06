"""Characterization tests for ``marketClearing``.

These tests pinned the pre-1.7 behaviour. The expectations marked
``characterization_changes_in_1_7`` and the ``met_demand`` expectation of the
distinct-price example now pin the corrected behaviour of PLAN 1.7:
``marketClearing`` delegates to ``clearing.clear_market``, splits an equal-price
marginal group by equal share capped at bid quantity (B4, OQ-18) and returns
``met_demand`` as the sum of the accepted quantities (B11). Accepted
quantities are ``Decimal`` (B8).
"""

from decimal import Decimal

import pytest

from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offers,
    make_period,
    make_resource,
)

pytestmark = [pytest.mark.characterization, pytest.mark.django_db]


@pytest.fixture
def clearing_context():
    """Saved market, period, agent and resource for building offers."""
    market = make_market()
    resource = make_resource()
    agent = make_agent_with_plants(market, plants=[(resource, 1000)])
    period = make_period(market, demand=0)
    return market, period, agent, resource


def _clear(market, offers, demand):
    from abmem.services.market import market_service

    return market_service.marketClearing(market, offers, demand)


def test_distinct_prices_partial_marginal_offer(clearing_context):
    market, period, agent, resource = clearing_context
    a, b, c, d = make_offers(
        period, agent, resource, [(400, 50), (300, 70), (500, 90), (200, 120)]
    )

    offers, met_demand, ptf, volume = _clear(market, [a, b, c, d], 1000)

    assert [o.acceptanceAmount for o in offers[:3]] == [400, 300, 300]
    assert [o.acceptance for o in offers[:3]] == [True, True, True]
    assert [o.acceptancePrice for o in offers[:3]] == [
        Decimal(50),
        Decimal(70),
        Decimal(90),
    ]
    # D keeps its constructed acceptance state.
    assert d.acceptance is None
    assert d.acceptanceAmount is None
    assert d.acceptancePrice is None
    assert ptf == Decimal(90)
    assert volume == 1400
    # B11: met demand is the sum of the accepted quantities (400 + 300 + 300).
    # The pre-1.7 code returned 700.
    assert met_demand == 1000


@pytest.mark.characterization_changes_in_1_7
def test_same_price_group_b4_example(clearing_context):
    market, period, agent, resource = clearing_context
    a, b = make_offers(period, agent, resource, [(100, 60), (10, 60)])

    offers, met_demand, ptf, volume = _clear(market, [a, b], 60)

    # Equal share capped at bid quantity (OQ-18): the share 30 exceeds B's 10,
    # so B is capped at 10 and A receives the remaining 50. The pre-1.7 code
    # accepted only B=10, A=20.
    assert b.acceptanceAmount == Decimal("10")
    assert a.acceptanceAmount == Decimal("50")
    assert isinstance(a.acceptanceAmount, Decimal)
    assert a.acceptance is True and b.acceptance is True
    assert a.acceptanceAmount + b.acceptanceAmount == 60
    assert ptf == Decimal(60)
    assert volume == 110
    # The returned list is the input list, unchanged in order and length.
    assert offers == [a, b]
    # B11: met demand equals the accepted total. The pre-1.7 code returned 0.
    assert met_demand == 60


def test_supply_shortfall_accepts_everything(clearing_context):
    market, period, agent, resource = clearing_context
    offers_in = make_offers(period, agent, resource, [(100, 40), (200, 80), (50, 60)])

    offers, met_demand, ptf, volume = _clear(market, offers_in, 1000)

    assert all(o.acceptance is True for o in offers)
    assert [o.acceptanceAmount for o in offers] == [100, 200, 50]
    assert [o.acceptancePrice for o in offers] == [o.offerPrice for o in offers]
    assert ptf == Decimal(80)
    assert met_demand == 350
    assert volume == 350


def test_zero_demand_accepts_nothing(clearing_context):
    market, period, agent, resource = clearing_context
    offers_in = make_offers(period, agent, resource, [(100, 40), (200, 80)])

    offers, met_demand, ptf, volume = _clear(market, offers_in, 0)

    assert all(o.acceptance is None for o in offers)
    assert all(o.acceptanceAmount is None for o in offers)
    assert ptf == 0
    assert met_demand == 0
    assert volume == 300


def test_clearing_saves_market_state(clearing_context):
    from abmem.models.enums import MarketState

    market, period, agent, resource = clearing_context
    offers_in = make_offers(period, agent, resource, [(10, 40)])

    _clear(market, offers_in, 5)

    market.refresh_from_db()
    assert market.state == MarketState.CALCULATING
