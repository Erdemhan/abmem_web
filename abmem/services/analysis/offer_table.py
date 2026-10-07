"""Filtered, paginated offer table of a run (PLAN 2.11, spec watch_analysis_ui).

:func:`parse_offer_filter` validates the query string of the offers endpoint.
:func:`offer_page` returns one page of offers ordered by period, agent and
offer id (the order of the Offers CSV). One page costs a constant number of
queries: the agent check (only with an agent filter), the count and the page.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from django.core.paginator import Paginator

PAGE_SIZES: Final[tuple[int, ...]] = (10, 25, 50, 100)
"""Allowed values of ``page_size``."""

DEFAULT_PAGE_SIZE: Final[int] = 10

ACCEPTED_CHOICES: Final[tuple[str, ...]] = ("all", "yes", "no")
"""Allowed values of ``accepted``."""


class OfferFilterError(ValueError):
    """Invalid query parameters of the offers endpoint.

    Attributes:
        errors: Field name -> list of messages.
    """

    def __init__(self, errors: dict[str, list[str]]) -> None:
        """Store the field errors.

        Args:
            errors: Field name -> list of messages.
        """
        super().__init__(" ".join(f"{k}: {' '.join(v)}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class OfferFilter:
    """Validated filter and page of the offers table.

    Attributes:
        agent_id: Agent primary key, or None for all agents.
        period_from: First period (inclusive), or None.
        period_to: Last period (inclusive), or None.
        accepted: ``"all"``, ``"yes"`` or ``"no"``.
        page: 1-based page number.
        page_size: Rows per page, one of :data:`PAGE_SIZES`.
    """

    agent_id: int | None = None
    period_from: int | None = None
    period_to: int | None = None
    accepted: str = "all"
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE


def _optional_int(
    params: Mapping[str, str],
    name: str,
    minimum: int,
    errors: dict[str, list[str]],
) -> int | None:
    """Parse an optional integer parameter.

    Args:
        params: Query parameters.
        name: Parameter name.
        minimum: Smallest allowed value.
        errors: Collects the message of an invalid value.

    Returns:
        The value, or None if the parameter is missing, empty or invalid.
    """
    raw = str(params.get(name, "") or "").strip()
    if raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        errors.setdefault(name, []).append("Enter a whole number.")
        return None
    if value < minimum:
        errors.setdefault(name, []).append(f"Must be at least {minimum}.")
        return None
    return value


def parse_offer_filter(params: Mapping[str, str]) -> OfferFilter:
    """Validate the query parameters of the offers endpoint.

    Args:
        params: Query parameters ``agent``, ``period_from``, ``period_to``,
            ``accepted``, ``page`` and ``page_size``, all optional.

    Returns:
        The filter. Missing parameters take the defaults of
        :class:`OfferFilter`.

    Raises:
        OfferFilterError: A parameter is not a valid number, out of range,
            ``period_from > period_to``, ``accepted`` is not in
            :data:`ACCEPTED_CHOICES` or ``page_size`` not in :data:`PAGE_SIZES`.
    """
    errors: dict[str, list[str]] = {}
    agent_id = _optional_int(params, "agent", 1, errors)
    period_from = _optional_int(params, "period_from", 1, errors)
    period_to = _optional_int(params, "period_to", 1, errors)
    if period_from is not None and period_to is not None and period_from > period_to:
        errors.setdefault("period_to", []).append(
            "Must not be smaller than period_from."
        )
    accepted = str(params.get("accepted", "") or "all").strip().lower()
    if accepted not in ACCEPTED_CHOICES:
        errors.setdefault("accepted", []).append(
            f"Choose one of {', '.join(ACCEPTED_CHOICES)}."
        )
    page = _optional_int(params, "page", 1, errors)
    page_size = _optional_int(params, "page_size", 1, errors)
    if page_size is not None and page_size not in PAGE_SIZES:
        errors.setdefault("page_size", []).append(
            f"Choose one of {', '.join(str(size) for size in PAGE_SIZES)}."
        )
    if errors:
        raise OfferFilterError(errors)
    return OfferFilter(
        agent_id=agent_id,
        period_from=period_from,
        period_to=period_to,
        accepted=accepted,
        page=page or 1,
        page_size=page_size or DEFAULT_PAGE_SIZE,
    )


def _optional_float(value: Any) -> float | None:
    """Return ``float(value)``, or None for None.

    Args:
        value: A number or None.

    Returns:
        The float or None.
    """
    return None if value is None else float(value)


def offer_page(simulation: Any, offer_filter: OfferFilter) -> dict[str, Any]:
    """Return one page of the filtered offers of a run.

    A page number beyond the last page returns the last page (the filter may
    have reduced the number of pages).

    Args:
        simulation: The run (``Simulation`` with its market).
        offer_filter: Validated filter.

    Returns:
        ``{count, page, page_size, num_pages, rows}``. Each row has
        ``period, agent, resource, quantity, accepted_quantity, offer_price,
        acceptance_price, accepted, profit`` (``profit`` is the offer profit
        ``g`` in USD, None for an offer that was not cleared).

    Raises:
        OfferFilterError: ``agent`` is not an agent of the run.
    """
    from abmem.models.models import Offer
    from abmem.services.market.accounting import offer_profit, outcome_from_offer

    market = simulation.market
    offers = Offer.objects.filter(period__market=market)
    if offer_filter.agent_id is not None:
        if not market.agent_set.filter(pk=offer_filter.agent_id).exists():
            raise OfferFilterError({"agent": ["Unknown agent of this run."]})
        offers = offers.filter(agent_id=offer_filter.agent_id)
    if offer_filter.period_from is not None:
        offers = offers.filter(period__periodNumber__gte=offer_filter.period_from)
    if offer_filter.period_to is not None:
        offers = offers.filter(period__periodNumber__lte=offer_filter.period_to)
    if offer_filter.accepted == "yes":
        offers = offers.filter(acceptance=True)
    elif offer_filter.accepted == "no":
        offers = offers.exclude(acceptance=True)
    offers = (
        offers.select_related("period", "agent", "resource")
        .only(
            "id",
            "amount",
            "offerPrice",
            "acceptance",
            "acceptancePrice",
            "acceptanceAmount",
            "period__periodNumber",
            "agent__name",
            "resource__name",
            "resource__omCost",
            "resource__investmentCost",
            "resource__co2Cost",
            "resource__fuelCost",
        )
        .order_by("period__periodNumber", "agent_id", "id")
    )
    paginator = Paginator(offers, offer_filter.page_size)
    page = paginator.get_page(offer_filter.page)
    rows = []
    for offer in page.object_list:
        profit = (
            None
            if offer.acceptanceAmount is None
            else float(offer_profit(outcome_from_offer(offer)))
        )
        rows.append(
            {
                "period": offer.period.periodNumber,
                "agent": offer.agent.name,
                "resource": offer.resource.name,
                "quantity": int(offer.amount),
                "accepted_quantity": _optional_float(offer.acceptanceAmount),
                "offer_price": float(offer.offerPrice),
                "acceptance_price": _optional_float(offer.acceptancePrice),
                "accepted": bool(offer.acceptance),
                "profit": profit,
            }
        )
    return {
        "count": paginator.count,
        "page": page.number,
        "page_size": offer_filter.page_size,
        "num_pages": paginator.num_pages,
        "rows": rows,
    }
