"""Single, dimensionally consistent fail-stack rule (PLAN 1.5, K3).

Per plant ``i`` and period ``t``:

* Failure: ``f_i(t) = 1`` if the offer was rejected (``q_acc == 0``) or
  ``margin_i = g_i / (p_acc * q_acc) < m_min``. The margin is USD / USD and
  therefore unitless. If the accepted revenue ``p_acc * q_acc`` is 0, the
  offer is a failure only if ``g_i < 0`` (OQ-13).
* Counter: ``n_i <- n_i + 1`` on failure, ``n_i <- 0`` on success, so the
  counter holds the number of consecutive failures.
* Intervention: if ``n_i >= L`` at the next decision, the price of plant ``i``
  is drawn from ``U[max(low, PTF(t-1) - delta), PTF(t-1))``, clipped to
  ``[low, high]``, and ``n_i <- 0``. ``delta`` is in USD/MWh, the unit of the
  price, so the comparison is dimensionally consistent.

The module is pure. It holds no global state, reads no settings and does not
import django at module level. Randomness comes only from the injected
``numpy.random.Generator``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from abmem.services.agent.action_space import PRICE_QUANTUM
from abmem.services.market.accounting import OfferOutcome

if TYPE_CHECKING:
    from abmem.models import Agent

DEFAULT_LIMIT: Final[int] = 6
DEFAULT_DELTA: Final[Decimal] = Decimal("15")
DEFAULT_MIN_MARGIN: Final[Decimal] = Decimal("0")


def _require_finite(name: str, value: Any) -> Decimal:
    """Return ``value`` as a finite ``Decimal`` or raise.

    Args:
        name: Name used in the error message.
        value: A ``Decimal`` or ``int`` (``bool`` is rejected).

    Returns:
        The value as a ``Decimal``.

    Raises:
        TypeError: If ``value`` is neither ``Decimal`` nor ``int``.
        ValueError: If ``value`` is NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
        raise TypeError(f"{name} must be a Decimal, got {type(value).__name__}")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"{name} must be finite, got {result}")
    return result


def _to_decimal(name: str, value: Any) -> Decimal:
    """Convert a model attribute (Decimal, int, float or str) to ``Decimal``.

    Floats are converted through ``str`` so that ``15.0`` becomes
    ``Decimal('15.0')`` instead of its binary expansion.

    Args:
        name: Attribute name used in the error message.
        value: Attribute value.

    Returns:
        The value as a ``Decimal`` (not yet checked for finiteness).

    Raises:
        TypeError: If ``value`` cannot be interpreted as a number.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, got bool")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        return Decimal(str(value))
    raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


@dataclass(frozen=True)
class FailStackParams:
    """Per-agent fail-stack parameters.

    Attributes:
        limit: L, number of consecutive failures that triggers an
            intervention. Must be ``>= 1``.
        delta: delta in USD/MWh, width of the intervention interval below
            PTF(t-1). Must be finite and ``>= 0``.
        min_margin: m_min, unitless margin threshold. Must be finite.
    """

    limit: int
    delta: Decimal
    min_margin: Decimal

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If ``limit < 1``, ``limit`` is not an integer,
                ``delta`` is negative or non-finite, or ``min_margin`` is
                non-finite or not numeric.
        """
        if isinstance(self.limit, bool) or not isinstance(self.limit, int):
            raise ValueError(f"limit must be an int, got {self.limit!r}")
        if self.limit < 1:
            raise ValueError(f"limit must be >= 1, got {self.limit}")
        try:
            delta = _require_finite("delta", self.delta)
            min_margin = _require_finite("min_margin", self.min_margin)
        except TypeError as exc:
            raise ValueError(str(exc)) from None
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "min_margin", min_margin)

    @classmethod
    def from_agent(cls, agent: Agent) -> FailStackParams | None:
        """Build the parameters from an ``Agent``'s fail-stack fields.

        Only attributes are read (no query). ``getattr`` defaults make this
        work before the model fields exist: disabled, L = 6, delta = 15,
        m_min = 0, algorithm DDPG.

        Args:
            agent: Agent instance (or any object with the same attributes).

        Returns:
            None if ``failstack_enabled`` is falsy or ``algorithm_type`` is a
            baseline (RANDOM, MARGINAL_COST, FIXED_MARKUP), else the
            validated parameters.

        Raises:
            ValueError: If the stored parameters are invalid.
        """
        if not getattr(agent, "failstack_enabled", False):
            return None
        if _is_baseline(getattr(agent, "algorithm_type", None)):
            return None
        try:
            delta = _to_decimal(
                "failstack_delta", getattr(agent, "failstack_delta", DEFAULT_DELTA)
            )
            min_margin = _to_decimal(
                "failstack_min_margin",
                getattr(agent, "failstack_min_margin", DEFAULT_MIN_MARGIN),
            )
        except TypeError as exc:
            raise ValueError(str(exc)) from None
        return cls(
            limit=getattr(agent, "failstack_limit", DEFAULT_LIMIT),
            delta=delta,
            min_margin=min_margin,
        )


def _is_baseline(algorithm_type: Any) -> bool:
    """Whether ``algorithm_type`` is one of the baseline algorithm types.

    The enum and the baseline set are imported lazily so this module has no
    module-level dependency on ``abmem.models``.

    Args:
        algorithm_type: An ``AlgorithmType``, its integer value, or None
            (treated as the model default DDPG, not a baseline).

    Returns:
        True for RANDOM, MARGINAL_COST and FIXED_MARKUP.
    """
    if algorithm_type is None:
        return False
    from abmem.models.enums import AlgorithmType
    from abmem.services.algorithms.registry import BASELINE_TYPES

    return AlgorithmType(int(algorithm_type)) in BASELINE_TYPES


@dataclass
class FailStackState:
    """Consecutive-failure counters of one agent.

    Attributes:
        counters: One counter per plant, in ``Plant.id`` order.
    """

    counters: list[int]

    @classmethod
    def zeros(cls, n_plants: int) -> FailStackState:
        """Create a state with all counters at 0.

        Args:
            n_plants: Number of plants, ``>= 1``.

        Returns:
            A new state with ``n_plants`` zero counters.

        Raises:
            ValueError: If ``n_plants < 1``.
        """
        if isinstance(n_plants, bool) or not isinstance(n_plants, int) or n_plants < 1:
            raise ValueError(f"n_plants must be an int >= 1, got {n_plants!r}")
        return cls(counters=[0] * n_plants)


def margin(outcome: OfferOutcome, profit: Decimal) -> Decimal | None:
    """Profit margin ``g / (p_acc * q_acc)`` of one offer (unitless).

    Args:
        outcome: Cleared offer outcome.
        profit: Profit g of the offer in USD (``offer_profit(outcome)``).

    Returns:
        The margin, or None if the accepted revenue is 0 (which includes
        rejected offers).

    Raises:
        TypeError: If ``profit`` is not a ``Decimal``.
        ValueError: If ``profit`` is non-finite.
    """
    profit = _require_finite("profit", profit)
    if not outcome.accepted:
        return None
    assert outcome.acceptance_price is not None
    revenue = outcome.acceptance_price * outcome.accepted_quantity
    if revenue == 0:
        return None
    return profit / revenue


def is_failure(outcome: OfferOutcome, profit: Decimal, min_margin: Decimal) -> bool:
    """Failure indicator ``f_i(t)`` of one offer.

    Args:
        outcome: Cleared offer outcome.
        profit: Profit g of the offer in USD.
        min_margin: Threshold m_min (unitless).

    Returns:
        True if the offer was rejected, or its margin is below
        ``min_margin``. If the accepted revenue is 0, True iff
        ``profit < 0`` (OQ-13).

    Raises:
        TypeError: If ``profit`` or ``min_margin`` is not a ``Decimal``.
        ValueError: If ``profit`` or ``min_margin`` is non-finite.
    """
    min_margin = _require_finite("min_margin", min_margin)
    profit = _require_finite("profit", profit)
    if not outcome.accepted:
        return True
    m = margin(outcome, profit)
    if m is None:
        return profit < 0
    return m < min_margin


def update_counters(state: FailStackState, failures: Sequence[bool]) -> None:
    """Update the consecutive-failure counters in place.

    Args:
        state: Counters to update.
        failures: One failure flag per plant.

    Raises:
        ValueError: If ``len(failures) != len(state.counters)``.
    """
    if len(failures) != len(state.counters):
        raise ValueError(
            f"got {len(failures)} failure flags for {len(state.counters)} counters"
        )
    for i, failed in enumerate(failures):
        state.counters[i] = state.counters[i] + 1 if failed else 0


def _validate_bounds(low: Decimal, high: Decimal) -> tuple[Decimal, Decimal]:
    """Validate the market price bounds.

    Args:
        low: Lower bid bound.
        high: Upper bid bound.

    Returns:
        ``(low, high)`` as ``Decimal``.

    Raises:
        ValueError: If a bound is non-finite or ``high < low``.
    """
    try:
        low = _require_finite("low", low)
        high = _require_finite("high", high)
    except TypeError as exc:
        raise ValueError(str(exc)) from None
    if high < low:
        raise ValueError(f"high must be >= low, got [{low}, {high}]")
    return low, high


def intervention_price(
    ptf_prev: Decimal,
    delta: Decimal,
    low: Decimal,
    high: Decimal,
    rng: np.random.Generator,
) -> Decimal:
    """Draw an intervention price from ``U[max(low, PTF - delta), PTF)``.

    The draw is rounded down to ``PRICE_QUANTUM`` (so it stays strictly
    below PTF(t-1)) and then clipped to ``[low, high]``.

    Args:
        ptf_prev: PTF(t-1) in USD/MWh.
        delta: Interval width in USD/MWh, ``>= 0``.
        low: Lower bid bound.
        high: Upper bid bound.
        rng: Generator seeded per agent, the only source of randomness.

    Returns:
        The intervention price. If the interval is empty (``b <= a``) the
        result is ``min(max(ptf_prev, low), high)``.

    Raises:
        ValueError: If an input is non-finite, ``delta < 0`` or
            ``high < low``.
    """
    low, high = _validate_bounds(low, high)
    try:
        ptf_prev = _require_finite("ptf_prev", ptf_prev)
        delta = _require_finite("delta", delta)
    except TypeError as exc:
        raise ValueError(str(exc)) from None
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta}")
    a = max(low, ptf_prev - delta)
    b = ptf_prev
    if b <= a:
        return min(max(ptf_prev, low), high)
    u = Decimal(float(rng.random()))
    raw = a + u * (b - a)
    price = raw.quantize(PRICE_QUANTUM, rounding=ROUND_FLOOR)
    return min(max(price, low), high)


def apply_interventions(
    state: FailStackState,
    params: FailStackParams,
    prices: Sequence[Decimal],
    ptf_prev: Decimal | None,
    low: Decimal,
    high: Decimal,
    rng: np.random.Generator,
) -> tuple[list[Decimal], list[bool]]:
    """Replace the prices of plants whose counter reached the limit.

    Plants are processed in order, so the generator is consumed
    deterministically.

    Args:
        state: Counters, modified in place (intervened counters reset to 0).
        params: Fail-stack parameters.
        prices: Prices proposed by the policy, one per plant.
        ptf_prev: PTF(t-1), or None if no market price is known yet.
        low: Lower bid bound.
        high: Upper bid bound.
        rng: Generator seeded per agent.

    Returns:
        ``(applied_prices, intervened_flags)``. If ``ptf_prev`` is None the
        prices are returned unchanged, all flags are False and the counters
        are not modified.

    Raises:
        ValueError: If ``len(prices) != len(state.counters)`` or an input
            is invalid.
    """
    if len(prices) != len(state.counters):
        raise ValueError(f"got {len(prices)} prices for {len(state.counters)} counters")
    applied = list(prices)
    flags = [False] * len(applied)
    if ptf_prev is None:
        return applied, flags
    for i, count in enumerate(state.counters):
        if count >= params.limit:
            applied[i] = intervention_price(ptf_prev, params.delta, low, high, rng)
            state.counters[i] = 0
            flags[i] = True
    return applied, flags
