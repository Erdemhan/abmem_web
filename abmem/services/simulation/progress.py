"""Run progress of a simulation for the watch page (PLAN 2.11, spec watch_analysis_ui).

:func:`compute_run_progress` is a pure function of the run counters and
timestamps. :func:`simulation_progress` loads them for a ``Simulation`` with
one aggregate query over its periods.

A period counts as completed when its ``Period`` row exists, because the
market service inserts the row after the clearing of that period. The run
starts at ``Simulation.created_at``: a run is cloned from its scenario when it
is started (``clone_scenario_for_run``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Final

from abmem.models.enums import SimulationMode, SimulationState

TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {"FINISHED", "FAILED", "CANCELED", "STOPPED"}
)
"""Names of the simulation states in which the run no longer advances."""

PHASE_TRAINING: Final[str] = "training"
PHASE_EVALUATION: Final[str] = "evaluation"


@dataclass(frozen=True)
class RunProgress:
    """Progress of a run.

    Attributes:
        completed: Completed periods, in ``[0, total]``.
        total: Number of periods of the run (``periodNumber``).
        percent: ``100 * completed / total`` rounded to one decimal, 100 for a
            FINISHED run, 0 when ``total`` is not positive.
        phase: ``"training"`` or ``"evaluation"`` for the period in progress
            (the last period for a stopped run), None when the run has no
            evaluation window.
        evaluation_periods: Length of the evaluation window.
        started_at: ISO 8601 start time of the run, or None.
        elapsed_seconds: Seconds since the start (up to the end for a stopped
            run), None without a start time.
        remaining_seconds: Estimated seconds until the run finishes, from the
            mean time per completed period. None for a stopped run, before the
            first period is completed, in PERIODBYPERIOD mode (the run waits
            for the user) and without timestamps.
        finished: True if the run is in a terminal state.
    """

    completed: int
    total: int
    percent: float
    phase: str | None
    evaluation_periods: int
    started_at: str | None
    elapsed_seconds: float | None
    remaining_seconds: float | None
    finished: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the progress as a JSON-serialisable dict.

        Returns:
            The fields of the dataclass.
        """
        return asdict(self)


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    """Return ``end - start`` in seconds, clamped at 0, or None.

    Args:
        start: Start time.
        end: End time.

    Returns:
        Non-negative seconds rounded to 0.1, or None if a bound is missing.
    """
    if start is None or end is None:
        return None
    return round(max(0.0, (end - start).total_seconds()), 1)


def compute_run_progress(
    *,
    total: int,
    completed: int,
    current_period: int,
    evaluation_periods: int,
    state: str,
    period_by_period: bool,
    started_at: datetime | None,
    last_period_at: datetime | None,
    ended_at: datetime | None,
    now: datetime,
) -> RunProgress:
    """Compute the progress of a run from its counters and timestamps.

    Args:
        total: ``periodNumber`` of the run.
        completed: Number of stored periods.
        current_period: ``currentPeriod`` (the period in progress, -1 or 0
            before the first period).
        evaluation_periods: Length of the evaluation window (0 for none).
        state: Name of the ``SimulationState``.
        period_by_period: True in PERIODBYPERIOD mode.
        started_at: Start of the run.
        last_period_at: Creation time of the last stored period.
        ended_at: End of the run, used for a run in a terminal state.
        now: Current time, used for a running run.

    Returns:
        The progress.
    """
    total = max(0, int(total))
    completed = min(max(0, int(completed)), total)
    finished = state in TERMINAL_STATES
    if total <= 0:
        percent = 0.0
    elif state == "FINISHED":
        percent = 100.0
        completed = total
    else:
        percent = round(100.0 * completed / total, 1)

    evaluation_periods = max(0, int(evaluation_periods))
    phase: str | None = None
    if evaluation_periods > 0 and total > 0:
        active = completed if finished else max(int(current_period), completed + 1)
        active = min(max(active, 1), total)
        phase = (
            PHASE_EVALUATION
            if active >= total - evaluation_periods + 1
            else PHASE_TRAINING
        )

    end = (ended_at or last_period_at or now) if finished else now
    elapsed = _seconds(started_at, end)

    remaining: float | None = None
    if not finished and not period_by_period and 0 < completed < total:
        spent = _seconds(started_at, last_period_at)
        if spent is not None:
            remaining = round(spent / completed * (total - completed), 1)
    if not finished and completed >= total > 0 and not period_by_period:
        remaining = 0.0

    return RunProgress(
        completed=completed,
        total=total,
        percent=percent,
        phase=phase,
        evaluation_periods=evaluation_periods,
        started_at=started_at.isoformat() if started_at is not None else None,
        elapsed_seconds=elapsed,
        remaining_seconds=remaining,
        finished=finished,
    )


def simulation_progress(simulation: Any, now: datetime) -> RunProgress:
    """Compute the progress of a stored simulation.

    Args:
        simulation: A ``Simulation`` with its market.
        now: Current time (timezone-aware when ``USE_TZ`` is on).

    Returns:
        The progress. Costs one aggregate query over the periods.
    """
    from django.db.models import Count, Max

    from abmem.models.models import Period

    stats = Period.objects.filter(market__simulation=simulation).aggregate(
        n=Count("id"), last=Max("created_at")
    )
    state = SimulationState.get(getattr(simulation.state, "value", simulation.state))
    mode = SimulationMode.get(getattr(simulation.mode, "value", simulation.mode))
    state_name = state.name if state is not None else ""
    return compute_run_progress(
        total=simulation.periodNumber,
        completed=stats["n"] or 0,
        current_period=simulation.currentPeriod,
        evaluation_periods=simulation.evaluation_periods,
        state=state_name,
        period_by_period=mode == SimulationMode.PERIODBYPERIOD,
        started_at=simulation.created_at,
        last_period_at=stats["last"],
        ended_at=simulation.updated_at if state_name in TERMINAL_STATES else None,
        now=now,
    )
