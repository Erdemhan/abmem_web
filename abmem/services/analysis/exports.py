"""CSV exports of a run (PLAN 2.5).

Every export is UTF-8 text with a header row, comma separated, decimal point
``.`` and ``\\r\\n`` line ends (``csv`` module default). Empty cells stand for
NULL. Each function issues one query and writes nothing.

Column order:

* :func:`periods_csv`: ``period, demand, met_demand, ptf, market_volume``,
  one row per period, by period number.
* :func:`offers_csv`: ``period, agent, resource, amount, offer_price,
  accepted, acceptance_amount, acceptance_price``, one row per offer, by
  period number, agent id and offer id. ``accepted`` is ``1``/``0``.
* :func:`step_logs_csv`: ``period, agent, reward, critic_loss, actor_loss,
  mean_q, exploration_scale, intervened_plants, learning_active,
  evaluation_phase``, one row per ``AgentStepLog`` row, by agent id and
  period number. ``intervened_plants`` is the JSON list of per-plant flags,
  the booleans are ``1``/``0``.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abmem.models import Simulation

PERIOD_COLUMNS: tuple[str, ...] = (
    "period",
    "demand",
    "met_demand",
    "ptf",
    "market_volume",
)
OFFER_COLUMNS: tuple[str, ...] = (
    "period",
    "agent",
    "resource",
    "amount",
    "offer_price",
    "accepted",
    "acceptance_amount",
    "acceptance_price",
)
STEP_LOG_COLUMNS: tuple[str, ...] = (
    "period",
    "agent",
    "reward",
    "critic_loss",
    "actor_loss",
    "mean_q",
    "exploration_scale",
    "intervened_plants",
    "learning_active",
    "evaluation_phase",
)


def _cell(value: Any) -> str:
    """Format one CSV cell.

    Args:
        value: Cell value.

    Returns:
        ``''`` for None, ``'1'``/``'0'`` for booleans, ``repr`` for floats
        (shortest round-trip form), ``str`` otherwise (Decimal keeps its
        stored scale).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _write(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Write a header and rows to a CSV string.

    Args:
        header: Column names.
        rows: Row values in column order.

    Returns:
        The CSV text.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow([_cell(value) for value in row])
    return buffer.getvalue()


def periods_csv(sim: Simulation) -> str:
    """Export the periods of a run.

    Args:
        sim: The run.

    Returns:
        CSV text with the columns :data:`PERIOD_COLUMNS`.
    """
    from abmem.models.models import Period

    rows = (
        Period.objects.filter(market__simulation=sim)
        .order_by("periodNumber", "id")
        .values_list("periodNumber", "demand", "metDemand", "ptf", "marketVolume")
    )
    return _write(PERIOD_COLUMNS, rows)


def offers_csv(sim: Simulation) -> str:
    """Export the offers of a run.

    Args:
        sim: The run.

    Returns:
        CSV text with the columns :data:`OFFER_COLUMNS`.
    """
    from abmem.models.models import Offer

    rows = (
        Offer.objects.filter(period__market__simulation=sim)
        .order_by("period__periodNumber", "agent_id", "id")
        .values_list(
            "period__periodNumber",
            "agent__name",
            "resource__name",
            "amount",
            "offerPrice",
            "acceptance",
            "acceptanceAmount",
            "acceptancePrice",
        )
    )
    return _write(OFFER_COLUMNS, rows)


def step_logs_csv(sim: Simulation) -> str:
    """Export the step logs of the agents of a run.

    Args:
        sim: The run.

    Returns:
        CSV text with the columns :data:`STEP_LOG_COLUMNS`.
    """
    from abmem.models import AgentStepLog

    rows = (
        AgentStepLog.objects.filter(agent__market__simulation=sim)
        .order_by("agent_id", "period__periodNumber")
        .values_list(
            "period__periodNumber",
            "agent__name",
            "reward",
            "critic_loss",
            "actor_loss",
            "mean_q",
            "exploration_scale",
            "intervened_plants",
            "learning_active",
            "evaluation_phase",
        )
    )
    return _write(
        STEP_LOG_COLUMNS,
        (
            (*row[:7], json.dumps([bool(flag) for flag in row[7] or []]), *row[8:])
            for row in rows
        ),
    )
