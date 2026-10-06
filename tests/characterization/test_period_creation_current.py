"""Characterization tests for the current ``createPeriod`` implementation.

``createPeriod`` reads ``marketData['demand'][market.simulation.currentPeriod]``
from the module-level ``marketData`` global, so row 0 is never used when the
simulation starts at period 1.
"""

import pytest

from abmem.models.models import Period
from tests.factories import make_market, make_simulation

pytestmark = [pytest.mark.characterization, pytest.mark.django_db]


@pytest.mark.characterization_changes_in_1_7
def test_create_period_uses_current_period_as_demand_index(monkeypatch):
    from abmem.services.market import market_service

    monkeypatch.setattr(market_service, "marketData", {"demand": [111, 222, 333]})
    market = make_market(simulation=make_simulation(current_period=1))

    period = market_service.createPeriod(market)

    assert period.pk is not None
    assert period.market_id == market.pk
    assert period.periodNumber == 1
    assert period.demand == 222
    assert period.metDemand == -1
    assert period.marketVolume == -1
    assert period.ptf == -1

    stored = Period.objects.get(pk=period.pk)
    assert stored.demand == 222
    assert stored.ptf == -1
