"""
Seed a small SYNTHETIC template simulation for the SQLite demo.

This is NOT real data — it's a minimal, self-consistent market so the scenario
runner (and the existing optimizer) can execute end-to-end without the real
Postgres database. Run from the project root with the sqlite toggle:

    USE_SQLITE=1 python seed_synthetic.py

It prints the template Simulation id; pass that to the runner via TEMPLATE_PROXY_ID.
"""
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
os.environ.setdefault("USE_SQLITE", "1")
import django
django.setup()

from decimal import Decimal
from abmem.models import Simulation, Market, Agent, Portfolio, Plant, Resource
from abmem.models.enums import (
    EnergyType, PeriodType, SimulationMode, SimulationState,
    MarketStrategy, MarketState, AgentType, AgentState,
)

# ---- Resources (costs drive staticCost = om+investment+co2, variableCost = fuel) ----
RESOURCES = [
    # name,           energyType,            fuel, om,  invest, co2, emission
    ("hydro",         EnergyType.RENEWABLE,  0,    4,   3,      0,   0),
    ("naturalgas",    EnergyType.FOSSIL,     60,   8,   5,      10,  4),
    ("lignite",       EnergyType.FOSSIL,     25,   6,   5,      15,  9),
]

# ---- Agents: (name, type, resource_name, capacity) — one plant each ----
AGENTS = [
    ("HydroCo", AgentType.RENEWABLE, "hydro",      12000),
    ("GasCo",   AgentType.FOSSIL,    "naturalgas", 12000),
    ("CoalCo",  AgentType.FOSSIL,    "lignite",    12000),
]

PERIOD_NUMBER = 48  # 2 days of hourly periods — enough to trigger learning (>3), small enough to be quick


def get_or_make_resources():
    out = {}
    for name, etype, fuel, om, inv, co2, em in RESOURCES:
        res, _ = Resource.objects.get_or_create(
            name=name,
            defaults=dict(
                energyType=etype,
                fuelCost=Decimal(fuel), omCost=Decimal(om),
                investmentCost=Decimal(inv), co2Cost=Decimal(co2),
                emission=Decimal(em),
            ),
        )
        out[name] = res
    return out


def main():
    resources = get_or_make_resources()

    # Template simulation. mode=ONLYRESULT so the serial run loop does not block
    # waiting on market.state (PERIODBYPERIOD busy-waits forever in this flow).
    sim = Simulation.objects.create(
        name="demo_template",
        mode=SimulationMode.ONLYRESULT,
        state=SimulationState.CREATED,
        periodType=PeriodType.HOUR,
        periodNumber=PERIOD_NUMBER,
        currentPeriod=1,
        day=0,
        proxy=True,  # makes create_simulation_from_proxy clone it
    )

    market = Market.objects.create(
        strategy=MarketStrategy.PAYASPTF,
        state=MarketState.CREATED,
        lowerBidBound=0,
        upperBidBound=200,
        simulation=sim,
        proxy=True,
    )

    for name, atype, res_name, cap in AGENTS:
        agent = Agent.objects.create(
            name=name, market=market, state=AgentState.CREATED,
            budget=Decimal(0), type=atype, proxy=True,
        )
        portfolio = Portfolio.objects.create(agent=agent, proxy=True)
        Plant.objects.create(portfolio=portfolio, resource=resources[res_name],
                             capacity=cap, proxy=True)

    print(f"SEED_OK template_simulation_id={sim.id} "
          f"agents={len(AGENTS)} resources={len(RESOURCES)} periods={PERIOD_NUMBER}")


if __name__ == "__main__":
    main()
