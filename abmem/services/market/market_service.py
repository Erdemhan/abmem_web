"""Market period of a run (ARCHITECTURE.md 6.2).

``run_period`` creates the period, collects one offer per plant from every
agent in ``Agent.id`` order, clears the market with the pure merit-order
clearing, books profits in ``Decimal`` and writes everything with a constant
number of bulk statements inside one transaction (A10).

The module holds no mutable module-level state (B10).
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Decimal
from typing import TYPE_CHECKING, Final

from django.db import transaction

from abmem.models import Agent
from abmem.models.enums import MarketState, MarketStrategy
from abmem.models.models import Offer, Period
from abmem.services.agent import agent_service
from abmem.services.market.accounting import offer_profit, outcome_from_offer
from abmem.services.market.clearing import Bid, acceptance_prices, clear_market

if TYPE_CHECKING:
    from abmem.services.simulation.context import SimulationContext

QUANTITY_QUANTUM: Final[Decimal] = Decimal("0.001")
BUDGET_QUANTUM: Final[Decimal] = Decimal("0.01")
PTF_SENTINEL: Final[int] = -1


def _set_market_state(ctx: SimulationContext, state: MarketState) -> None:
    """Persist a market state change with a single-column update.

    Args:
        ctx: Run context.
        state: New market state.
    """
    ctx.market.state = state
    ctx.market.save(update_fields=["state", "updated_at"])


def run_period(ctx: SimulationContext, t: int) -> Period:
    """Run period ``t``: agent steps, clearing, accounting and persistence.

    Args:
        ctx: Run context.
        t: Period number, ``1 <= t <= periodNumber``.

    Returns:
        The saved period with ``ptf``, ``metDemand`` and ``marketVolume`` set.

    Raises:
        RuntimeError: If the clearing yields no PTF (no positive bid).
        ValueError: If a component rejects its input. The transaction of the
            period is rolled back.
    """
    demand = ctx.demand.at(t).quantize(QUANTITY_QUANTUM, rounding=ROUND_HALF_EVEN)
    try:
        with transaction.atomic():
            period = Period.objects.create(
                market=ctx.market,
                periodNumber=t,
                demand=demand,
                metDemand=PTF_SENTINEL,
                marketVolume=PTF_SENTINEL,
                ptf=PTF_SENTINEL,
            )
            ctx.current_period = period
            _set_market_state(ctx, MarketState.WAITINGAGENTS)

            offers: list[Offer] = []
            owner_index: list[int] = []
            for index, rt in enumerate(ctx.agents):
                agent_offers = agent_service.step(ctx, rt, t)
                offers.extend(agent_offers)
                owner_index.extend([index] * len(agent_offers))

            bids = [
                Bid(
                    key=i,
                    price=Decimal(offer.offerPrice),
                    quantity=Decimal(offer.amount),
                )
                for i, offer in enumerate(offers)
            ]
            result = clear_market(bids, demand, quantum=QUANTITY_QUANTUM)
            if result.ptf is None:
                raise RuntimeError(f"period {t}: clearing produced no PTF")
            prices = acceptance_prices(
                result, bids, MarketStrategy(int(ctx.market.strategy))
            )

            profits = [Decimal(0)] * len(ctx.agents)
            outcomes: list[list] = [[] for _ in ctx.agents]
            for i, offer in enumerate(offers):
                accepted = result.accepted[i]
                offer.acceptance = accepted > 0
                offer.acceptanceAmount = accepted
                offer.acceptancePrice = prices[i]
                outcome = outcome_from_offer(offer)
                owner = owner_index[i]
                outcomes[owner].append(outcome)
                profits[owner] += offer_profit(outcome)

            agents: list[Agent] = []
            for rt, profit in zip(ctx.agents, profits):
                rt.agent.budget = (Decimal(rt.agent.budget) + profit).quantize(
                    BUDGET_QUANTUM, rounding=ROUND_HALF_EVEN
                )
                agents.append(rt.agent)

            Offer.objects.bulk_create(offers)
            period.ptf = result.ptf
            period.metDemand = result.met_demand
            period.marketVolume = int(result.offered_volume)
            period.save(
                update_fields=["ptf", "metDemand", "marketVolume", "updated_at"]
            )
            Agent.objects.bulk_update(agents, ["budget", "state"])
            ctx.step_logger.flush()
            _set_market_state(ctx, MarketState.PERIODEND)
    except BaseException:
        ctx.step_logger.clear()
        raise

    ctx.ptf_history[t] = result.ptf
    for rt, agent_outcomes in zip(ctx.agents, outcomes):
        rt.last_outcomes = tuple(agent_outcomes)
    return period
