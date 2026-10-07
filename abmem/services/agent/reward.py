"""K1 reward of an agent for one period.

``r = sum_i g_i / (sum_i q_i * P_max)`` over all offers of the agent in the
period (B2), where ``g_i`` is the profit from
:func:`abmem.services.market.accounting.offer_profit`, ``sum_i q_i`` is the
total plant capacity of the agent and ``P_max`` is ``Market.upperBidBound``.
The ratio is unitless, returned as a Python float, never clipped and never
converted to int (C1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from decimal import Decimal

from abmem.services.market.accounting import OfferOutcome, offer_profit


def compute_reward(
    outcomes: Sequence[OfferOutcome], total_capacity: Decimal, price_cap: Decimal
) -> float:
    """Compute the K1 reward over all offers of one agent in one period.

    Args:
        outcomes: Outcomes of every offer the agent submitted in the period.
        total_capacity: Sum of the agent's plant capacities (MWh).
        price_cap: Upper bid bound ``P_max`` of the market (USD/MWh).

    Returns:
        The reward as a finite Python float.

    Raises:
        ValueError: If ``outcomes`` is empty, or ``total_capacity`` or
            ``price_cap`` is not strictly positive and finite.
    """
    if len(outcomes) == 0:
        raise ValueError("outcomes must not be empty, an agent has at least one plant")
    total_capacity = Decimal(total_capacity)
    price_cap = Decimal(price_cap)
    if not total_capacity.is_finite() or total_capacity <= 0:
        raise ValueError(f"total_capacity must be finite and > 0, got {total_capacity}")
    if not price_cap.is_finite() or price_cap <= 0:
        raise ValueError(f"price_cap must be finite and > 0, got {price_cap}")

    total_profit = sum((offer_profit(o) for o in outcomes), Decimal(0))
    reward = float(total_profit / (total_capacity * price_cap))
    assert math.isfinite(reward), f"non-finite reward {reward}"
    return reward
