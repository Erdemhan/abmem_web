"""Legacy composite agent score and market score (archived pre-B9 definition).

This module is an exact port of ``compute_advanced_scores`` in
``abc_log/try/sim/compute_advanced_scores.py`` without its console output.
It is kept only to compare new runs with results produced by the legacy
analysis script (``docs/legacy_rewards.md``, section "HPO evaluation
metric"). The corrected definition (B9) is
:func:`abmem.services.hpo.objective.score_simulation`.

The legacy rows are the offers of the legacy JSON export, ordered by period,
agent and offer id. ``budget`` is the budget of the agent after the period of
the row. Per agent ``a`` with first offer amount ``q_a`` (taken as capacity),
first-row budget ``B_a^first``, last-row budget ``B_a^last``, ``T`` distinct
periods and ``P_max = 200``:

    G_a           = B_a^last - B_a^first
    norm_budget_a = clip(G_a / (q_a * T * P_max), -1, 1)    (0 if q_a * T * P_max <= 0)
    norm_profit_a = clip((G_a / Q_a) / P_max, -1, 1)          (Q_a = sum of accepted amounts,
                                                               G_a / Q_a := 0 if Q_a <= 0)
    similarity_a  = 1 - RMSE(offer_price - market_price) / P_max
    score_a       = clip(0.4 * norm_budget_a + 0.4 * norm_profit_a + 0.2 * similarity_a, 0, 1)

Over all rows with mean market price ``m`` and mean offer price ``o``:

    market_score = 0.6 * (1 - m / P_max) + 0.4 * (m / o)      (second ratio 0 if o <= 0)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

LEGACY_PRICE_CAP: Final[float] = 200.0
"""Hard-coded price cap ``P_max`` of the legacy script."""


def _clip(value: float, low: float, high: float) -> float:
    """Clip ``value`` to ``[low, high]`` (``max(low, min(value, high))``)."""
    return max(low, min(value, high))


def _mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    return math.fsum(values) / len(values)


def legacy_composite_scores(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], float]:
    """Compute the legacy composite agent scores and the legacy market score.

    Archived pre-B9 definition kept for comparison (``docs/legacy_rewards.md``).
    It uses the constant price cap 200, the amount of the first offer of an
    agent as its capacity and the first-row budget (the budget after the
    first period) as its initial budget.

    Args:
        rows: Offer rows in legacy order (period, agent id, offer id). Each
            row has the keys ``period`` (int), ``agent`` (str), ``amount``
            (number), ``offer_price`` (float), ``market_price`` (float, PTF
            of the period), ``accepted`` (bool) and ``budget`` (float, budget
            of the agent after the period).

    Returns:
        ``(agent_scores, market_score)``. ``agent_scores`` maps agent names in
        order of first appearance to scores in ``[0, 1]``.

    Raises:
        ValueError: If ``rows`` is empty.
    """
    if not rows:
        raise ValueError("legacy scores need at least one offer row")

    price_cap = LEGACY_PRICE_CAP
    num_periods = len({row["period"] for row in rows})

    by_agent: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_agent.setdefault(row["agent"], []).append(row)

    agent_scores: dict[str, float] = {}
    for agent, agent_rows in by_agent.items():
        capacity = float(agent_rows[0]["amount"])
        initial_budget = float(agent_rows[0]["budget"])
        final_budget = float(agent_rows[-1]["budget"])
        raw_budget_change = final_budget - initial_budget
        max_possible_revenue = capacity * num_periods * price_cap
        norm_budget = (
            raw_budget_change / max_possible_revenue if max_possible_revenue > 0 else 0
        )
        norm_budget = _clip(norm_budget, -1.0, 1.0)

        total_profit = raw_budget_change
        total_amount = math.fsum(
            float(row["amount"]) for row in agent_rows if row["accepted"]
        )
        avg_profit_per_unit = total_profit / total_amount if total_amount > 0 else 0
        norm_profit = avg_profit_per_unit / price_cap if price_cap > 0 else 0
        norm_profit = _clip(norm_profit, -1.0, 1.0)

        rmse = math.sqrt(
            _mean(
                [
                    (float(row["offer_price"]) - float(row["market_price"])) ** 2
                    for row in agent_rows
                ]
            )
        )
        similarity = 1 - (rmse / price_cap)

        score = 0.4 * norm_budget + 0.4 * norm_profit + 0.2 * similarity
        agent_scores[agent] = float(_clip(score, 0, 1))

    mcp_avg = _mean([float(row["market_price"]) for row in rows])
    avg_offer_price = _mean([float(row["offer_price"]) for row in rows])
    mci = mcp_avg / price_cap
    mpi = mcp_avg / avg_offer_price if avg_offer_price > 0 else 0
    market_score = 0.6 * (1 - mci) + 0.4 * mpi
    return agent_scores, float(market_score)
