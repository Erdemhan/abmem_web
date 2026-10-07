"""Market period of a run (ARCHITECTURE.md 6.2).

``run_period`` collects one offer per plant from every agent in ``Agent.id``
order, clears the market with the clearing algorithm of the market (PLAN 2.10),
checks the result against the clearing contract, books profits in
``Decimal`` and writes everything with a constant number of statements inside
one transaction (A10, PLAN 1.12): one period insert, one offer
``bulk_create``, one agent ``bulk_update``, one step-log ``bulk_create`` and
one market state update. The market state ``WAITINGAGENTS`` is not written,
because inside the period transaction it is never visible to another
connection.

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
from abmem.services.market.clearing import (
    Bid,
    acceptance_prices,
    validate_clearing_result,
)

if TYPE_CHECKING:
    from abmem.services.simulation.context import SimulationContext

QUANTITY_QUANTUM: Final[Decimal] = Decimal("0.001")
BUDGET_QUANTUM: Final[Decimal] = Decimal("0.01")


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
        ClearingContractError: If the clearing algorithm returns a result that
            breaks the contract of ``validate_clearing_result``.
        ValueError: If a component rejects its input. The transaction of the
            period is rolled back.
    """
    demand = ctx.demand.at(t).quantize(QUANTITY_QUANTUM, rounding=ROUND_HALF_EVEN)
    # The period row is inserted once, after clearing, with its final values.
    # Offers and step-log rows reference the unsaved instance and receive its
    # primary key when they are bulk-created after the insert.
    period = Period(market=ctx.market, periodNumber=t, demand=demand)
    ctx.current_period = period
    states_before = [rt.agent.state for rt in ctx.agents]
    budgets_before = [rt.agent.budget for rt in ctx.agents]
    try:
        with transaction.atomic():
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
            result = validate_clearing_result(
                ctx.clearing.clear(bids, demand, QUANTITY_QUANTUM),
                bids,
                demand,
                QUANTITY_QUANTUM,
                ctx.price_low,
                ctx.price_high,
                algorithm=ctx.clearing.key,
            )
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

            period.ptf = result.ptf
            period.metDemand = result.met_demand
            period.marketVolume = int(result.offered_volume)
            period.save(force_insert=True)
            Offer.objects.bulk_create(offers)
            agent_fields = ["budget"]
            if any(
                rt.agent.state != before
                for rt, before in zip(ctx.agents, states_before)
            ):
                agent_fields.append("state")
            Agent.objects.bulk_update(agents, agent_fields)
            ctx.step_logger.flush()
            _set_market_state(ctx, MarketState.PERIODEND)
    except BaseException:
        ctx.step_logger.clear()
        # Keep the in-memory agents equal to the rolled-back rows, so the
        # state comparison above stays valid for a later period.
        for rt, state, budget in zip(ctx.agents, states_before, budgets_before):
            rt.agent.state = state
            rt.agent.budget = budget
        raise

    ctx.ptf_history[t] = result.ptf
    for rt, agent_outcomes in zip(ctx.agents, outcomes):
        rt.last_outcomes = tuple(agent_outcomes)
    return period
