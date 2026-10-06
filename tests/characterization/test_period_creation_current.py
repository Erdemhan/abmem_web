"""Characterization tests for ``createPeriod``.

Since PLAN 1.7 ``createPeriod`` reads the demand of period ``t`` from the
validated module-level ``demandSeries`` with ``DemandSeries.at(t)``, which is
row ``t - 1``. Before 1.7 it read row ``t`` and row 0 was never used (A8).
"""

import numpy as np
import pytest

from abmem.models.models import Period
from abmem.services.market.demand import DemandSeries
from tests.factories import make_market, make_simulation

pytestmark = [pytest.mark.characterization, pytest.mark.django_db]


@pytest.mark.characterization_changes_in_1_7
def test_create_period_uses_period_number_minus_one_as_demand_index(monkeypatch):
    from abmem.services.market import market_service

    monkeypatch.setattr(
        market_service,
        "demandSeries",
        DemandSeries(values=np.array([111.0, 222.0, 333.0]), source="test"),
    )
    market = make_market(simulation=make_simulation(current_period=1))

    period = market_service.createPeriod(market)

    assert period.pk is not None
    assert period.market_id == market.pk
    assert period.periodNumber == 1
    assert period.demand == 111
    assert period.metDemand == -1
    assert period.marketVolume == -1
    assert period.ptf == -1

    stored = Period.objects.get(pk=period.pk)
    assert stored.demand == 111
    assert stored.ptf == -1
