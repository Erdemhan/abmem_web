"""Elapsed and remaining time of an HPO study on the watch page."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from abmem.views_hpo import hpo_time_estimate

T0 = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


def test_remaining_time_scales_the_mean_time_per_finished_trial() -> None:
    elapsed, remaining = hpo_time_estimate(T0, None, T0 + timedelta(minutes=10), 5, 25, running=True)
    assert elapsed == pytest.approx(600.0)
    assert remaining == pytest.approx(600.0 / 5 * 20)


@pytest.mark.parametrize(
    ("started", "done", "running"),
    [(None, 5, True), (T0, 0, True), (T0, 5, False)],
)
def test_no_estimate_without_start_finished_trials_or_running_study(started, done, running) -> None:
    _, remaining = hpo_time_estimate(started, None, T0 + timedelta(minutes=10), done, 25, running=running)
    assert remaining is None


def test_finished_study_uses_its_finish_time() -> None:
    elapsed, remaining = hpo_time_estimate(
        T0, T0 + timedelta(hours=2), T0 + timedelta(days=1), 25, 25, running=False
    )
    assert elapsed == pytest.approx(7200.0)
    assert remaining is None


def test_done_beyond_total_gives_zero_remaining() -> None:
    _, remaining = hpo_time_estimate(T0, None, T0 + timedelta(minutes=1), 30, 25, running=True)
    assert remaining == 0.0
