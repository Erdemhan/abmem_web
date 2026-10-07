"""Retention of HPO evaluation data (K9).

``HPOTrial`` rows and the evaluation ``Simulation``, ``Market`` and ``Agent``
rows are always kept. Periods (and through ``CASCADE`` their offers and agent
step logs) are deleted for every scored evaluation simulation except those of
the current best candidate.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abmem.models import HPOStudy, Simulation


def purge_simulation_details(simulation: Simulation) -> None:
    """Delete the periods, offers and step logs of one evaluation simulation.

    Args:
        simulation: Evaluation run. Its ``Simulation``, ``Market`` and
            ``Agent`` rows are kept.
    """
    from abmem.models.models import Period

    Period.objects.filter(market__simulation=simulation).delete()


def apply_retention(
    study: HPOStudy,
    best_candidate_key: tuple[int, int, int] | None,
    *,
    keep_trial_ids: Collection[int] | None = None,
) -> None:
    """Purge every FINISHED evaluation of ``study`` except the best candidate's.

    Args:
        study: The study.
        best_candidate_key: ``(iteration, phase value, source_index)`` of the
            current best candidate, or None when there is no best yet.
        keep_trial_ids: When given, exactly these trials keep their data and
            ``best_candidate_key`` is ignored. The service passes trial ids
            because several onlookers of one batch can share a key.
    """
    from abmem.models.enums import HPOTrialStatus
    from abmem.models.models import Period

    finished = study.trials.filter(
        status=HPOTrialStatus.FINISHED, simulation__isnull=False
    )
    if keep_trial_ids is not None:
        purge = finished.exclude(id__in=list(keep_trial_ids))
    elif best_candidate_key is not None:
        iteration, phase, source_index = best_candidate_key
        purge = finished.exclude(
            iteration=iteration, phase=phase, source_index=source_index
        )
    else:
        purge = finished
    simulation_ids = list(purge.values_list("simulation_id", flat=True))
    if simulation_ids:
        Period.objects.filter(market__simulation_id__in=simulation_ids).delete()
