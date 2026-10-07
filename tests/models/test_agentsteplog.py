"""AgentStepLog, Simulation run settings and decimal demand (data_model spec, 0010)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from django.db import IntegrityError, transaction

from abmem.models import AgentStepLog, Simulation
from abmem.models.enums import SimulationState
from abmem.models.models import Period
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_period,
    make_simulation,
)

pytestmark = pytest.mark.django_db


def test_steplog_json_defaults_and_nullable_floats() -> None:
    market = make_market()
    agent = make_agent_with_plants(market)
    period = make_period(market, period_number=1)
    log = AgentStepLog.objects.create(agent=agent, period=period)
    log.refresh_from_db()
    assert log.intervened_plants == []
    assert log.extra == {}
    assert log.reward is None and log.critic_loss is None and log.mean_q is None
    assert log.learning_active is False and log.evaluation_phase is False
    assert list(agent.step_logs.all()) == [log]
    assert list(period.step_logs.all()) == [log]


def test_steplog_unique_agent_period() -> None:
    market = make_market()
    agent = make_agent_with_plants(market)
    period = make_period(market, period_number=1)
    AgentStepLog.objects.create(agent=agent, period=period)
    with pytest.raises(IntegrityError), transaction.atomic():
        AgentStepLog.objects.create(agent=agent, period=period)


def test_steplog_cascades_with_period() -> None:
    market = make_market()
    agent = make_agent_with_plants(market)
    period = make_period(market, period_number=1)
    AgentStepLog.objects.create(agent=agent, period=period)
    period.delete()
    assert AgentStepLog.objects.count() == 0


def test_simulation_run_setting_defaults() -> None:
    simulation = make_simulation()
    simulation.refresh_from_db()
    assert simulation.seed == 17081999
    assert simulation.observation_lags == [1, 24]
    assert simulation.evaluation_periods == 0
    assert simulation.error == ""
    other = make_simulation()
    other.observation_lags.append(168)
    assert simulation.observation_lags == [1, 24]


def test_failed_state_round_trip() -> None:
    simulation = make_simulation()
    simulation.state = SimulationState.FAILED
    simulation.error = "boom"
    simulation.save()
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert simulation.error == "boom"


@pytest.mark.parametrize(
    "fields",
    [
        {"seed": -1},
        {"observation_lags": [1, 168]},
        {"observation_lags": []},
        {"evaluation_periods": 10},
    ],
)
def test_simulation_constraints_reject(fields: dict[str, Any]) -> None:
    simulation = make_simulation(period_number=10, save=False)
    for key, value in fields.items():
        setattr(simulation, key, value)
    with pytest.raises(IntegrityError), transaction.atomic():
        simulation.save()


def test_simulation_constraints_accept() -> None:
    simulation = make_simulation(period_number=10, save=False)
    simulation.seed = 2**31 - 1
    simulation.observation_lags = [1, 24, 168]
    simulation.evaluation_periods = 9
    simulation.save()
    assert Simulation.objects.filter(id=simulation.id).exists()


def test_period_demand_keeps_decimals() -> None:
    market = make_market()
    period = make_period(market, period_number=1, demand=Decimal("1234.567"))
    period.refresh_from_db()
    assert period.demand == Decimal("1234.567")
    assert isinstance(Period.objects.get(id=period.id).demand, Decimal)
