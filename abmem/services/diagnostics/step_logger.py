"""Buffered writer of ``AgentStepLog`` rows (PLAN 1.2, A9, A10).

The runtime adds one row per agent and period. ``flush`` writes all buffered
rows with one ``bulk_create``. Float fields must be finite in the database,
so non-finite values are stored as None and named in ``extra['non_finite']``.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abmem.models import Agent, AgentStepLog
    from abmem.models.models import Period
    from abmem.services.algorithms.base import UpdateStats

logger = logging.getLogger(__name__)


def _finite_or_none(
    name: str, value: float | None, non_finite: list[str]
) -> float | None:
    """Return ``value`` as float, or None if it is None or non-finite.

    Args:
        name: Field name recorded in ``non_finite`` when the value is dropped.
        value: Value to check.
        non_finite: List collecting the names of dropped fields.

    Returns:
        The finite float or None.
    """
    if value is None:
        return None
    number = float(value)
    if math.isfinite(number):
        return number
    non_finite.append(name)
    return None


class StepLogger:
    """Collects ``AgentStepLog`` instances of one period and bulk-writes them."""

    def __init__(self) -> None:
        """Create an empty buffer."""
        self._rows: list[AgentStepLog] = []

    def __len__(self) -> int:
        """Return the number of buffered rows."""
        return len(self._rows)

    def add(
        self,
        agent: Agent,
        period: Period,
        reward: float | None,
        stats: UpdateStats | None,
        exploration_scale: float | None,
        intervened: Sequence[bool],
        learning_active: bool,
        evaluation_phase: bool,
    ) -> None:
        """Buffer one row.

        Args:
            agent: Agent of the step.
            period: Decision period t (saved).
            reward: Reward of period t-1, None if not available.
            stats: Update statistics of the step, None if no gradient step.
            exploration_scale: Exploration scale used for the decision at t.
            intervened: Fail-stack flag per plant for the decision at t.
            learning_active: True if the policy is learning in this step.
            evaluation_phase: True if t lies in the evaluation window.
        """
        from abmem.models import AgentStepLog

        non_finite: list[str] = []
        extra: dict[str, Any] = {}
        if stats is not None:
            for key, value in stats.extra.items():
                number = _finite_or_none(f"extra.{key}", value, non_finite)
                if number is not None:
                    extra[str(key)] = number
        row = AgentStepLog(
            agent=agent,
            period=period,
            reward=_finite_or_none("reward", reward, non_finite),
            critic_loss=_finite_or_none(
                "critic_loss", None if stats is None else stats.critic_loss, non_finite
            ),
            actor_loss=_finite_or_none(
                "actor_loss", None if stats is None else stats.actor_loss, non_finite
            ),
            mean_q=_finite_or_none(
                "mean_q", None if stats is None else stats.mean_q, non_finite
            ),
            exploration_scale=_finite_or_none(
                "exploration_scale", exploration_scale, non_finite
            ),
            intervened_plants=[bool(flag) for flag in intervened],
            learning_active=bool(learning_active),
            evaluation_phase=bool(evaluation_phase),
            extra=extra,
        )
        if non_finite:
            extra["non_finite"] = non_finite
            logger.warning(
                "Non-finite step values of agent %s in period %s stored as None: %s",
                getattr(agent, "pk", None),
                getattr(period, "periodNumber", None),
                ", ".join(non_finite),
            )
        self._rows.append(row)

    def clear(self) -> None:
        """Drop all buffered rows without writing them."""
        self._rows = []

    def flush(self) -> int:
        """Write all buffered rows with one ``bulk_create`` and empty the buffer.

        Returns:
            The number of rows written.
        """
        from abmem.models import AgentStepLog

        rows = self._rows
        self._rows = []
        if not rows:
            return 0
        AgentStepLog.objects.bulk_create(rows)
        return len(rows)
