"""Current ABC round shown on the HPO watch page."""

from __future__ import annotations

from abmem.models.enums import HPOPhase, HPOTrialStatus
from abmem.views_hpo import round_progress

DONE = int(HPOTrialStatus.FINISHED)
FAILED = int(HPOTrialStatus.FAILED)
RUN = int(HPOTrialStatus.RUNNING)


def _row(iteration: int, phase: HPOPhase, status: int) -> dict:
    return {"iteration": iteration, "phase": int(phase), "status": status}


def test_no_trials_yet() -> None:
    assert round_progress([], max_iter=10) is None


def test_initial_round() -> None:
    rows = [_row(0, HPOPhase.INIT, DONE)] * 2 + [_row(0, HPOPhase.INIT, RUN)] * 8
    assert round_progress(rows, max_iter=10) == {
        "label": "initialisation", "iteration": 0, "max_iter": 10, "done": 2, "total": 10,
    }


def test_latest_round_counts_failed_as_done() -> None:
    rows = (
        [_row(0, HPOPhase.INIT, DONE)] * 5
        + [_row(1, HPOPhase.EMPLOYED, DONE)] * 5
        + [_row(1, HPOPhase.ONLOOKER, FAILED)] * 2
        + [_row(1, HPOPhase.ONLOOKER, RUN)] * 3
    )
    assert round_progress(rows, max_iter=10) == {
        "label": "onlooker bees", "iteration": 1, "max_iter": 10, "done": 7, "total": 10,
    }


def test_final_evaluation() -> None:
    rows = [_row(10, HPOPhase.ONLOOKER, DONE)] * 5 + [_row(10, HPOPhase.FINAL_EVAL, RUN)] * 2
    progress = round_progress(rows, max_iter=10)
    assert progress["label"] == "final evaluation"
    assert (progress["done"], progress["total"]) == (0, 2)
