"""Plain builder functions for ``abmem`` model instances used in tests.

The builders avoid third-party factory libraries. Every builder that persists
an object requires the ``django_db`` marker on the calling test.
"""

from __future__ import annotations

import itertools
from decimal import Decimal
from typing import Iterable, Optional, Sequence, Tuple

from abmem.models import Agent, Market, Simulation
from abmem.models.enums import (
    AgentType,
    EnergyType,
    MarketStrategy,
    PeriodType,
    SimulationMode,
)
from abmem.models.models import Offer, Period, Plant, Portfolio, Resource

_name_counter = itertools.count(1)


def _unique_name(prefix: str) -> str:
    """Return a short unique name (Resource.name is unique, max length 20).

    Args:
        prefix: Leading text of the generated name.

    Returns:
        A name such as ``"res-7"``.
    """
    return f"{prefix}-{next(_name_counter)}"[:20]


def make_resource(
    name: Optional[str] = None,
    energy_type: EnergyType = EnergyType.FOSSIL,
    fuel_cost: Decimal = Decimal("0"),
    om_cost: Decimal = Decimal("0"),
    investment_cost: Decimal = Decimal("0"),
    co2_cost: Decimal = Decimal("0"),
    emission: Decimal = Decimal("0"),
    save: bool = True,
) -> Resource:
    """Build a Resource.

    Args:
        name: Unique resource name. Generated when omitted.
        energy_type: Energy type enum value.
        fuel_cost: Variable cost per MWh.
        om_cost: Operation and maintenance cost per MWh of capacity.
        investment_cost: Investment cost per MWh of capacity.
        co2_cost: CO2 cost per MWh of capacity.
        emission: Emission factor.
        save: Persist the instance when True.

    Returns:
        The Resource instance.
    """
    resource = Resource(
        name=name or _unique_name("res"),
        energyType=energy_type,
        fuelCost=fuel_cost,
        omCost=om_cost,
        investmentCost=investment_cost,
        co2Cost=co2_cost,
        emission=emission,
    )
    if save:
        resource.save()
    return resource


def make_simulation(
    name: str = "test-sim",
    mode: SimulationMode = SimulationMode.ONLYRESULT,
    period_type: PeriodType = PeriodType.HOUR,
    period_number: int = 10,
    current_period: int = 0,
    day: int = 0,
    save: bool = True,
) -> Simulation:
    """Build a Simulation.

    Args:
        name: Simulation name.
        mode: Simulation mode enum value.
        period_type: Period type enum value.
        period_number: Total number of periods.
        current_period: Index of the current period.
        day: Current day counter.
        save: Persist the instance when True.

    Returns:
        The Simulation instance.
    """
    simulation = Simulation(
        name=name,
        mode=mode,
        periodType=period_type,
        periodNumber=period_number,
        currentPeriod=current_period,
        day=day,
    )
    if save:
        simulation.save()
    return simulation


def make_market(
    simulation: Optional[Simulation] = None,
    strategy: MarketStrategy = MarketStrategy.PAYASBID,
    lower_bid_bound: int = 0,
    upper_bid_bound: int = 3000,
    save: bool = True,
) -> Market:
    """Build a Market (one-to-one with a Simulation).

    Args:
        simulation: Owning simulation. A saved one is created when omitted.
        strategy: Market pricing strategy.
        lower_bid_bound: Lowest admissible bid price.
        upper_bid_bound: Highest admissible bid price.
        save: Persist the instance when True.

    Returns:
        The Market instance.
    """
    market = Market(
        simulation=simulation if simulation is not None else make_simulation(),
        strategy=strategy,
        lowerBidBound=lower_bid_bound,
        upperBidBound=upper_bid_bound,
    )
    if save:
        market.save()
    return market


def make_agent_with_plants(
    market: Market,
    plants: Iterable[Tuple[Resource, int]] = (),
    name: Optional[str] = None,
    agent_type: AgentType = AgentType.FOSSIL,
    budget: Decimal = Decimal("0"),
) -> Agent:
    """Build and persist an Agent with a Portfolio and Plants.

    Args:
        market: Market the agent belongs to.
        plants: Pairs of (resource, capacity) for each plant.
        name: Agent name. Generated when omitted.
        agent_type: Agent type enum value.
        budget: Initial budget.

    Returns:
        The saved Agent instance.
    """
    agent = Agent(
        name=name or _unique_name("agent"),
        market=market,
        type=agent_type,
        budget=budget,
    )
    agent.save()
    portfolio = Portfolio(agent=agent)
    portfolio.save()
    for resource, capacity in plants:
        Plant(portfolio=portfolio, resource=resource, capacity=capacity).save()
    return agent


def make_period(
    market: Market,
    period_number: int = 0,
    demand: int = 0,
    met_demand: int = -1,
    market_volume: int = -1,
    ptf: Optional[Decimal] = Decimal("-1"),
    save: bool = True,
) -> Period:
    """Build a Period.

    Args:
        market: Owning market.
        period_number: Period index.
        demand: Demand in MWh.
        met_demand: Met demand in MWh.
        market_volume: Total offered volume in MWh.
        ptf: Market clearing price.
        save: Persist the instance when True.

    Returns:
        The Period instance.
    """
    period = Period(
        market=market,
        periodNumber=period_number,
        demand=demand,
        metDemand=met_demand,
        marketVolume=market_volume,
        ptf=ptf,
    )
    if save:
        period.save()
    return period


def make_offer(
    period: Period,
    agent: Agent,
    resource: Resource,
    amount: int,
    offer_price: Decimal,
    acceptance: Optional[bool] = None,
    acceptance_price: Optional[Decimal] = None,
    acceptance_amount: Optional[int] = None,
    save: bool = False,
) -> Offer:
    """Build an Offer. Not persisted by default.

    Args:
        period: Period the offer belongs to.
        agent: Offering agent.
        resource: Offered resource.
        amount: Offered amount in MWh.
        offer_price: Offer price.
        acceptance: Acceptance flag.
        acceptance_price: Accepted price.
        acceptance_amount: Accepted amount in MWh.
        save: Persist the instance when True.

    Returns:
        The Offer instance.
    """
    offer = Offer(
        period=period,
        agent=agent,
        resource=resource,
        amount=amount,
        offerPrice=offer_price,
        acceptance=acceptance,
        acceptancePrice=acceptance_price,
        acceptanceAmount=acceptance_amount,
    )
    if save:
        offer.save()
    return offer


def make_offers(
    period: Period,
    agent: Agent,
    resource: Resource,
    specs: Sequence[Tuple[int, Decimal]],
) -> list[Offer]:
    """Build unsaved Offers from (amount, price) pairs, preserving order.

    Args:
        period: Period the offers belong to.
        agent: Offering agent.
        resource: Offered resource.
        specs: Sequence of (amount, offer_price) pairs.

    Returns:
        The list of Offer instances in the order of ``specs``.
    """
    return [
        make_offer(period, agent, resource, amount, Decimal(price))
        for amount, price in specs
    ]
