"""Scores of a finished evaluation simulation, computed from the database (PLAN 1.11).

The definitions are the legacy HPO metrics ``agent_score`` and
``market_score`` (``docs/legacy_rewards.md``, section "HPO evaluation
metric") with the B9 corrections:

* the profit of an agent is the sum of the offer profits ``g`` over the
  scored periods (equal to final budget minus the true initial budget when
  all periods are scored), instead of the budget difference between the first
  and the last JSON record,
* the capacity of an agent is the total capacity of all its plants, instead of
  the amount of its first offer,
* the price cap ``P_max`` is ``Market.upperBidBound``, instead of the constant
  200.

Notation for agent ``a`` over the scored periods ``P`` (``T = |P|``):
``G_a = sum g``, ``C_a`` total plant capacity, ``Q_a`` sum of the offered
quantity of accepted offers, ``RMSE_a`` root mean square difference between
offer price and PTF over all offers of ``a``.

    norm_budget_a = clip(G_a / (C_a * T * P_max), -1, 1)
    norm_profit_a = clip((G_a / Q_a) / P_max, -1, 1)     (0 if Q_a = 0)
    similarity_a  = 1 - RMSE_a / P_max
    score_a       = clip(0.4 * norm_budget_a + 0.4 * norm_profit_a
                         + 0.2 * similarity_a, 0, 1)
    agent_score   = mean_a score_a

Over all offers of all agents in ``P`` with mean PTF ``m`` (offer-weighted, as
in the legacy metric) and mean offer price ``o``:

    market_score = 0.6 * (1 - m / P_max) + 0.4 * (m / o)   (second term 0 if o <= 0)

The study objective combines both with per-study weights (OQ-11), see
:func:`abmem.services.hpo.abc_optimizer.weighted_score`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abmem.models import Simulation

AGENT_WEIGHTS: tuple[float, float, float] = (0.4, 0.4, 0.2)
"""Legacy weights of (norm_budget, norm_profit, similarity)."""
MARKET_WEIGHTS: tuple[float, float] = (0.6, 0.4)
"""Legacy weights of (1 - market clearing index, market price index)."""


@dataclass(frozen=True)
class TrialScores:
    """Scores of one evaluation simulation.

    Attributes:
        agent_score: Mean per-agent score in ``[0, 1]``.
        market_score: Market score.
    """

    agent_score: float
    market_score: float


def _clip(value: float, low: float, high: float) -> float:
    """Clip ``value`` to ``[low, high]``."""
    return max(low, min(high, value))


def scored_periods(simulation: Simulation) -> range:
    """Return the period numbers that are scored by default.

    Args:
        simulation: The run.

    Returns:
        The evaluation window ``N - E + 1 .. N`` when
        ``evaluation_periods = E > 0`` (OQ-5), else ``1 .. N``.
    """
    total = int(simulation.periodNumber)
    window = int(simulation.evaluation_periods)
    if window > 0:
        return range(total - window + 1, total + 1)
    return range(1, total + 1)


def score_simulation(
    simulation: Simulation, periods: range | None = None
) -> TrialScores:
    """Compute agent score and market score of a finished run from the database.

    Args:
        simulation: Finished run with its market, agents, plants, periods and
            cleared offers.
        periods: Period numbers to score, :func:`scored_periods` when None.

    Returns:
        The scores.

    Raises:
        ValueError: If ``upperBidBound <= 0``, no cleared offer exists in the
            scored periods, a scored period has no PTF or an agent with
            offers has no positive capacity.
    """
    from abmem.models.models import Offer, Period, Plant
    from abmem.services.market.accounting import offer_profit, outcome_from_offer

    if periods is None:
        periods = scored_periods(simulation)
    market = simulation.market
    p_max = float(market.upperBidBound)
    if not p_max > 0.0:
        raise ValueError(f"upperBidBound must be > 0, got {market.upperBidBound}")

    period_rows = Period.objects.filter(
        market=market,
        periodNumber__gte=periods.start,
        periodNumber__lt=periods.stop,
    )
    n_periods = period_rows.count()
    offers = list(
        Offer.objects.filter(period__in=period_rows)
        .select_related("resource", "period")
        .order_by("period__periodNumber", "agent_id", "id")
    )
    offers = [o for o in offers if o.acceptanceAmount is not None]
    if n_periods == 0 or not offers:
        raise ValueError(
            f"simulation {simulation.pk} has no cleared offers in periods "
            f"{periods.start}..{periods.stop - 1}"
        )

    capacity: dict[int, Decimal] = defaultdict(Decimal)
    for agent_id, cap in Plant.objects.filter(
        portfolio__agent__market=market
    ).values_list("portfolio__agent_id", "capacity"):
        capacity[agent_id] += Decimal(cap)

    profit: dict[int, Decimal] = defaultdict(Decimal)
    accepted_amount: dict[int, Decimal] = defaultdict(Decimal)
    squared_error: dict[int, float] = defaultdict(float)
    count: dict[int, int] = defaultdict(int)
    ptf_sum = 0.0
    price_sum = 0.0
    for offer in offers:
        if offer.period.ptf is None:
            raise ValueError(f"period {offer.period.periodNumber} has no PTF")
        agent_id = offer.agent_id
        profit[agent_id] += offer_profit(outcome_from_offer(offer))
        if offer.acceptance:
            accepted_amount[agent_id] += Decimal(offer.amount)
        ptf = float(offer.period.ptf)
        price = float(offer.offerPrice)
        squared_error[agent_id] += (price - ptf) ** 2
        count[agent_id] += 1
        ptf_sum += ptf
        price_sum += price

    w_budget, w_profit, w_similarity = AGENT_WEIGHTS
    agent_scores: list[float] = []
    for agent_id in sorted(count):
        cap = float(capacity.get(agent_id, Decimal(0)))
        if not cap > 0.0:
            raise ValueError(f"agent {agent_id} has no positive plant capacity")
        g = float(profit[agent_id])
        norm_budget = _clip(g / (cap * n_periods * p_max), -1.0, 1.0)
        q_acc = float(accepted_amount[agent_id])
        norm_profit = _clip((g / q_acc) / p_max, -1.0, 1.0) if q_acc > 0.0 else 0.0
        rmse = math.sqrt(squared_error[agent_id] / count[agent_id])
        similarity = 1.0 - rmse / p_max
        score = (
            w_budget * norm_budget + w_profit * norm_profit + w_similarity * similarity
        )
        agent_scores.append(_clip(score, 0.0, 1.0))

    n_offers = len(offers)
    mean_ptf = ptf_sum / n_offers
    mean_price = price_sum / n_offers
    w_mci, w_mpi = MARKET_WEIGHTS
    mci = mean_ptf / p_max
    mpi = mean_ptf / mean_price if mean_price > 0.0 else 0.0
    market_score = w_mci * (1.0 - mci) + w_mpi * mpi
    return TrialScores(
        agent_score=float(sum(agent_scores) / len(agent_scores)),
        market_score=float(market_score),
    )
