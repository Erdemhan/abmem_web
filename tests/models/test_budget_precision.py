"""Regression test for B13: Agent.budget must hold large accumulated budgets."""

from decimal import Decimal

import pytest

from abmem.models import Agent
from tests.factories import make_agent_with_plants, make_market, make_simulation


@pytest.mark.django_db
@pytest.mark.parametrize(
    "budget",
    [
        Decimal("12345678901.23"),  # above the old max_digits=12 limit (~9.99e9)
        Decimal("-98765432109876.54"),  # large accumulated loss
        Decimal("999999999999999999.99"),  # max_digits=20, decimal_places=2
    ],
)
def test_large_budget_round_trips(budget: Decimal) -> None:
    market = make_market(make_simulation())
    agent = make_agent_with_plants(market, budget=budget)
    assert Agent.objects.get(pk=agent.pk).budget == budget
