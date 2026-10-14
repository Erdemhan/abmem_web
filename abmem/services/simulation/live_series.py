"""Incremental budget and offer series of a run for the watch page.

:func:`live_series` returns, for the periods after a given period number, the
clearing price of each period and, per agent, the budget after the period,
the mean offer price and the number of offers and accepted offers. The watch
page asks for the periods after the last one it already has and appends them
to its charts, so a poll during a long run only transfers the new periods.

Budget definition. The run does not store a budget per period. As in
:mod:`abmem.services.analysis.run_analysis`, the budget after period ``t`` is
reconstructed backwards from the stored ``Agent.budget`` with
:func:`~abmem.services.analysis.run_analysis.previous_budget`, using the
summed offer profit ``G_t`` of the agent in each period
(:func:`abmem.services.market.accounting.offer_profit`). The stored budget is
the budget after the latest period whose offers are visible, because
``market_service.run_period`` writes the period, its offers and the agent
budgets in one transaction. The offers and the budgets are read in a single
SQL statement, so both come from the same database snapshot even while the
run commits new periods. For a request with ``after = 0`` on a finished run
the series is the analysis series without its initial value. For a request
with ``after > 0`` it starts the reconstruction from the budget after the
latest period of that request, which gives the same values unless a
half-even rounding tie occurred (then the values differ by at most 0.01 per
tie, see the module docstring of ``run_analysis``).

Profit aggregation. The offers are grouped in the database by period, agent
and resource. For one group with offered quantities ``q_i``, accepted
quantities ``a_i`` and acceptance prices ``p_i`` the summed profit is
``sum(a_i * p_i) - c_s * sum(q_i) - c_v * sum(a_i)``, which equals the sum of
:func:`~abmem.services.market.accounting.offer_profit` over the group in
exact decimal arithmetic. The cost rates ``c_s`` and ``c_v`` come from
``Resource.staticCost()`` and ``Resource.variableCost()``.

Queries. Every call issues the same number of queries regardless of the
number of periods and agents: one for the agents, one grouped query for the
offers and one for the resources of the returned groups (skipped when there
are none). Nothing is written.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from django.db.models import Count, DecimalField, F, Q, Sum

from abmem.services.analysis.run_analysis import previous_budget

if TYPE_CHECKING:
    from abmem.models import Simulation

REVENUE_FIELD = DecimalField(max_digits=30, decimal_places=6)
"""Output field of the summed revenue ``acceptanceAmount * acceptancePrice``."""


@dataclass(frozen=True)
class AgentLiveSeries:
    """Series of one agent over the returned periods.

    Attributes:
        id: Agent id.
        name: Agent name.
        budget: Budget after each returned period (USD).
        budget_before: Budget before the first returned period (USD), None
            when no period is returned.
        mean_bid: Unweighted mean offer price of the agent in each period
            (USD/MWh), None for a period without offers of the agent.
        offers: Number of offers of the agent in each period.
        accepted_offers: Number of accepted offers of the agent in each
            period.
    """

    id: int
    name: str
    budget: list[float]
    budget_before: float | None
    mean_bid: list[float | None]
    offers: list[int]
    accepted_offers: list[int]


@dataclass(frozen=True)
class LiveSeries:
    """Series of a run for the periods after ``after``.

    Attributes:
        after: The requested lower bound (exclusive).
        periods: Period numbers in ascending order.
        ptf: Market clearing price of each period (USD/MWh), None if unset.
        agents: One entry per agent of the run, by agent id.
    """

    after: int
    periods: list[int]
    ptf: list[float | None]
    agents: list[AgentLiveSeries]

    def to_dict(self) -> dict[str, Any]:
        """Return the series as a JSON-serialisable dict.

        Returns:
            ``{"after", "periods", "ptf", "agents": [{"id", "name",
            "budget", "budget_before", "mean_bid", "offers",
            "accepted_offers"}]}``.
        """
        return asdict(self)


def _float_or_none(value: Decimal | None) -> float | None:
    """Convert a nullable decimal to float.

    Args:
        value: The value.

    Returns:
        ``float(value)``, or None.
    """
    return None if value is None else float(value)


def live_series(simulation: Simulation, after: int = 0) -> LiveSeries:
    """Return the budget and offer series of the periods after ``after``.

    Read-only, constant number of queries (see the module docstring). Works
    for runs in any state. A run without stored periods, or without periods
    after ``after``, gives empty series.

    Args:
        simulation: The run.
        after: Only periods with ``periodNumber > after`` are returned.
            Values below 0 are treated as 0.

    Returns:
        The series.
    """
    from abmem.models import Agent
    from abmem.models.models import Offer, Resource

    after = max(0, int(after))
    # Read before the offers, see the docstring of the budget anchor below.
    agents = list(
        Agent.objects.filter(market__simulation=simulation)
        .order_by("id")
        .values_list("id", "name", "budget")
    )
    groups = list(
        Offer.objects.filter(
            period__market__simulation=simulation, period__periodNumber__gt=after
        )
        .values(
            "period__periodNumber",
            "period__ptf",
            "agent_id",
            "agent__budget",
            "resource_id",
        )
        .annotate(
            offer_count=Count("id"),
            accepted_count=Count("id", filter=Q(acceptance=True)),
            price_sum=Sum("offerPrice"),
            quantity_sum=Sum("amount"),
            accepted_quantity_sum=Sum("acceptanceAmount"),
            revenue_sum=Sum(
                F("acceptanceAmount") * F("acceptancePrice"),
                output_field=REVENUE_FIELD,
            ),
        )
        .order_by("period__periodNumber", "agent_id", "resource_id")
    )
    resources = (
        Resource.objects.in_bulk({group["resource_id"] for group in groups})
        if groups
        else {}
    )

    periods: list[int] = []
    ptf: list[float | None] = []
    profit: dict[int, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    price_sum: dict[int, dict[int, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    offers: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    accepted: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    # Budget anchor: the stored budget read in the same statement as the
    # offers, so it is the budget after the latest period of ``groups``. An
    # agent without offers after ``after`` keeps the budget of the agents
    # query, which was read earlier and has not changed since because the
    # budget changes only through offer profits.
    anchor: dict[int, Decimal] = {agent_id: budget for agent_id, _, budget in agents}
    for group in groups:
        number = int(group["period__periodNumber"])
        if not periods or periods[-1] != number:
            periods.append(number)
            ptf.append(_float_or_none(group["period__ptf"]))
        agent_id = int(group["agent_id"])
        anchor[agent_id] = group["agent__budget"]
        resource = resources[group["resource_id"]]
        quantity = Decimal(group["quantity_sum"] or 0)
        accepted_quantity = Decimal(group["accepted_quantity_sum"] or 0)
        revenue = Decimal(group["revenue_sum"] or 0)
        profit[agent_id][number] += (
            revenue
            - Decimal(resource.staticCost()) * quantity
            - Decimal(resource.variableCost()) * accepted_quantity
        )
        price_sum[agent_id][number] += Decimal(group["price_sum"])
        offers[agent_id][number] += int(group["offer_count"])
        accepted[agent_id][number] += int(group["accepted_count"])

    series: list[AgentLiveSeries] = []
    for agent_id, name, _ in agents:
        budget = Decimal(anchor[agent_id])
        backwards = [budget]
        for number in reversed(periods):
            budget = previous_budget(budget, profit[agent_id].get(number, Decimal(0)))
            backwards.append(budget)
        backwards.reverse()
        counts = [offers[agent_id].get(number, 0) for number in periods]
        series.append(
            AgentLiveSeries(
                id=int(agent_id),
                name=str(name),
                budget=[float(value) for value in backwards[1:]],
                budget_before=float(backwards[0]) if periods else None,
                mean_bid=[
                    (float(price_sum[agent_id][number] / count) if count else None)
                    for number, count in zip(periods, counts)
                ],
                offers=counts,
                accepted_offers=[
                    accepted[agent_id].get(number, 0) for number in periods
                ],
            )
        )
    return LiveSeries(after=after, periods=periods, ptf=ptf, agents=series)
