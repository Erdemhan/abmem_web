"""Tests of the run progress of the watch page (PLAN 2.11, spec watch_analysis_ui)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import Simulation
from abmem.models.enums import SimulationMode, SimulationState
from abmem.services.simulation.progress import (
    PHASE_EVALUATION,
    PHASE_TRAINING,
    compute_run_progress,
)
from tests.factories import make_market, make_period, make_simulation

T0 = datetime(2026, 10, 7, 12, 0, 0, tzinfo=timezone.utc)


def _progress(**overrides):
    kwargs = dict(
        total=10,
        completed=4,
        current_period=5,
        evaluation_periods=0,
        state="STARTED",
        period_by_period=False,
        started_at=T0,
        last_period_at=T0 + timedelta(seconds=40),
        ended_at=None,
        now=T0 + timedelta(seconds=45),
    )
    kwargs.update(overrides)
    return compute_run_progress(**kwargs)


def test_running_run_percent_elapsed_and_remaining() -> None:
    progress = _progress()
    assert progress.completed == 4 and progress.total == 10
    assert progress.percent == 40.0
    assert progress.elapsed_seconds == 45.0
    # 40 s for 4 periods -> 10 s per period, 6 periods left.
    assert progress.remaining_seconds == 60.0
    assert progress.phase is None
    assert progress.finished is False
    assert progress.started_at == T0.isoformat()


def test_no_completed_period_has_no_estimate() -> None:
    progress = _progress(completed=0, current_period=0, last_period_at=None)
    assert progress.percent == 0.0
    assert progress.remaining_seconds is None
    assert progress.elapsed_seconds == 45.0


def test_finished_run_is_100_percent_and_uses_end_time() -> None:
    progress = _progress(
        completed=10,
        current_period=10,
        state="FINISHED",
        last_period_at=T0 + timedelta(seconds=100),
        ended_at=T0 + timedelta(seconds=101),
        now=T0 + timedelta(hours=5),
    )
    assert progress.percent == 100.0
    assert progress.finished is True
    assert progress.elapsed_seconds == 101.0
    assert progress.remaining_seconds is None


def test_failed_run_keeps_partial_percent() -> None:
    progress = _progress(state="FAILED", ended_at=T0 + timedelta(seconds=41))
    assert progress.percent == 40.0
    assert progress.finished is True
    assert progress.remaining_seconds is None
    assert progress.elapsed_seconds == 41.0


def test_period_by_period_has_no_estimate() -> None:
    assert _progress(period_by_period=True).remaining_seconds is None


@pytest.mark.parametrize(
    ("current_period", "completed", "expected"),
    [(1, 0, PHASE_TRAINING), (7, 6, PHASE_TRAINING), (8, 7, PHASE_EVALUATION)],
)
def test_phase_with_evaluation_window(
    current_period: int, completed: int, expected: str
) -> None:
    # 10 periods, evaluation window 8..10.
    progress = _progress(
        evaluation_periods=3, current_period=current_period, completed=completed
    )
    assert progress.phase == expected


def test_zero_total_and_out_of_range_counts_are_clamped() -> None:
    progress = _progress(total=0, completed=3)
    assert progress.percent == 0.0 and progress.completed == 0
    assert progress.remaining_seconds is None
    progress = _progress(completed=-2, last_period_at=None)
    assert progress.completed == 0


def test_clock_skew_never_gives_negative_seconds() -> None:
    progress = _progress(now=T0 - timedelta(seconds=5))
    assert progress.elapsed_seconds == 0.0


# status endpoint --------------------------------------------------------


@pytest.mark.django_db
def test_status_endpoint_returns_progress_fields(client: Client) -> None:
    simulation = make_simulation(name="run", period_number=4, current_period=3)
    simulation.state = SimulationState.STARTED
    simulation.evaluation_periods = 1
    simulation.save()
    market = make_market(simulation)
    make_period(market, 1, ptf=Decimal("40"))
    make_period(market, 2, ptf=Decimal("41"))

    response = client.get(
        reverse("get_simulation_status"), {"simulation_id": simulation.pk}
    )
    assert response.status_code == 200
    body = response.json()
    assert "periods" not in body
    assert body["status"] == "STARTED"
    assert body["period"] == 3 and body["total"] == 4
    progress = body["progress"]
    assert set(progress) >= {
        "completed",
        "total",
        "percent",
        "phase",
        "evaluation_periods",
        "started_at",
        "elapsed_seconds",
        "remaining_seconds",
        "finished",
    }
    assert progress["completed"] == 2
    assert progress["percent"] == 50.0
    assert progress["phase"] == PHASE_TRAINING
    assert progress["evaluation_periods"] == 1
    assert progress["finished"] is False
    assert progress["remaining_seconds"] is not None


@pytest.mark.django_db
def test_status_endpoint_of_a_failed_run(client: Client) -> None:
    simulation = make_simulation(
        name="run", period_number=4, current_period=2, mode=SimulationMode.ONLYRESULT
    )
    simulation.state = SimulationState.FAILED
    simulation.error = "boom"
    simulation.save()
    market = make_market(simulation)
    make_period(market, 1, ptf=Decimal("40"))

    body = client.get(
        reverse("get_simulation_status"), {"simulation_id": simulation.pk}
    ).json()
    assert body["status"] == "FAILED"
    assert body["error"] == "boom"
    assert body["progress"]["finished"] is True
    assert body["progress"]["percent"] == 25.0
    assert body["progress"]["remaining_seconds"] is None


# watch page -------------------------------------------------------------


@pytest.mark.django_db
def test_watch_page_has_progress_chart_and_no_offer_tables(client: Client) -> None:
    simulation: Simulation = make_simulation(name="run", period_number=4)
    make_market(simulation)
    html = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": simulation.pk})
    ).content.decode()
    assert "period-table" not in html
    assert "offers-period" not in html
    for element_id in (
        "progress-chart",
        "progress-percent",
        "progress-bar",
        "progress-period",
        "progress-elapsed",
        "progress-remaining",
        "progress-phase",
        "run-error",
        "nextRow",
    ):
        assert f'id="{element_id}"' in html, element_id
    assert "chartjs.min.js" in html
    assert reverse("get_simulation_status") in html
