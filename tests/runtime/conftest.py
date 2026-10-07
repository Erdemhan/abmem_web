"""Shared builders for the runtime tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Any

import numpy as np
import pytest
import torch

from abmem.models import Agent, Market, Simulation
from abmem.models.enums import (
    AgentType,
    MarketStrategy,
    PeriodType,
    SimulationMode,
    SimulationState,
)
from abmem.models.models import Plant, Portfolio, Resource
from abmem.services.market.demand import DemandSeries
from tests.factories import make_resource


@pytest.fixture(autouse=True)
def _restore_torch_determinism() -> Iterator[None]:
    """Undo ``torch.use_deterministic_algorithms(True)`` set by seeding."""
    before = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(before)


def constant_demand(n: int = 400, value: float = 1000.0) -> DemandSeries:
    """Return a demand series of ``n`` periods with a slowly varying profile.

    Args:
        n: Number of periods.
        value: Mean demand in MWh.

    Returns:
        The series.
    """
    t = np.arange(n, dtype=np.float64)
    values = np.round(value + 100.0 * np.sin(2 * np.pi * t / 24.0), 2)
    return DemandSeries(values=values, source="test")


def make_run(
    agents: Sequence[dict[str, Any]],
    period_number: int = 5,
    *,
    seed: int = 7,
    lower: int = 0,
    upper: int = 200,
    strategy: MarketStrategy = MarketStrategy.PAYASBID,
    evaluation_periods: int = 0,
    observation_lags: Sequence[int] = (1, 24),
    mode: SimulationMode = SimulationMode.ONLYRESULT,
) -> Simulation:
    """Create a run (non-proxy simulation) with a market, agents and plants.

    Args:
        agents: One dict per agent with keys ``plants`` (list of
            ``(Resource, capacity)``) and optional ``algorithm``,
            ``params``, ``budget``, ``failstack`` (dict of fail-stack fields).
        period_number: Number of periods.
        seed: Run seed.
        lower: Lower bid bound.
        upper: Upper bid bound.
        strategy: Market strategy.
        evaluation_periods: Size of the evaluation window.
        observation_lags: Observation lag set.
        mode: Simulation mode.

    Returns:
        The saved simulation in state CREATED.
    """
    simulation = Simulation.objects.create(
        name="rt",
        mode=mode,
        state=SimulationState.CREATED,
        periodType=PeriodType.HOUR,
        periodNumber=period_number,
        currentPeriod=-1,
        day=0,
        seed=seed,
        observation_lags=list(observation_lags),
        evaluation_periods=evaluation_periods,
    )
    market = Market.objects.create(
        simulation=simulation,
        strategy=strategy,
        lowerBidBound=lower,
        upperBidBound=upper,
    )
    for index, spec in enumerate(agents):
        failstack = spec.get("failstack", {})
        agent = Agent.objects.create(
            name=f"a{index}",
            market=market,
            type=AgentType.HYBRID,
            budget=spec.get("budget", Decimal("0")),
            algorithm_type=spec.get("algorithm", "RANDOM"),
            algorithm_params=spec.get("params", {}),
            **failstack,
        )
        portfolio = Portfolio.objects.create(agent=agent)
        for resource, capacity in spec["plants"]:
            Plant.objects.create(
                portfolio=portfolio, resource=resource, capacity=capacity
            )
    return simulation


def resources(n: int) -> list[Resource]:
    """Create ``n`` resources with distinct, positive costs.

    Args:
        n: Number of resources.

    Returns:
        The saved resources.
    """
    return [
        make_resource(
            fuel_cost=Decimal(10 + 5 * i),
            om_cost=Decimal("1.00"),
            investment_cost=Decimal("0.50"),
            co2_cost=Decimal("0.25"),
        )
        for i in range(n)
    ]


SMALL_DDPG: dict[str, Any] = {
    "batch_size": 8,
    "learning_starts": 8,
    "buffer_size": 512,
    "hidden_sizes": [16],
}


@pytest.fixture
def fast_demand(monkeypatch: pytest.MonkeyPatch) -> DemandSeries:
    """Replace the Excel demand file with an in-memory series of 400 periods."""
    series = constant_demand()

    def _from_excel(*_args: Any, **_kwargs: Any) -> DemandSeries:
        return series

    monkeypatch.setattr(DemandSeries, "from_excel", staticmethod(_from_excel))
    return series
