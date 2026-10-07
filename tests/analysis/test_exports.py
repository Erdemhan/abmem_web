"""Tests of the CSV exports (PLAN 2.5)."""

from __future__ import annotations

import csv
import io
import json
from decimal import Decimal

import pytest

from abmem.models import AgentStepLog, Simulation
from abmem.models.models import Offer, Period
from abmem.services.analysis.exports import (
    OFFER_COLUMNS,
    PERIOD_COLUMNS,
    STEP_LOG_COLUMNS,
    offers_csv,
    periods_csv,
    step_logs_csv,
)
from tests.analysis.test_run_analysis import hand_run  # noqa: F401  (fixture)

pytestmark = pytest.mark.django_db


def _parse(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


def test_periods_csv(hand_run: Simulation) -> None:  # noqa: F811
    rows = _parse(periods_csv(hand_run))
    assert tuple(rows[0]) == PERIOD_COLUMNS
    assert len(rows) == 1 + Period.objects.filter(market__simulation=hand_run).count()
    assert rows[1] == ["1", "120.000", "100.000", "40.00", "230"]
    assert rows[2][0] == "2"
    assert float(rows[2][2]) == pytest.approx(110.5)


def test_offers_csv(hand_run: Simulation) -> None:  # noqa: F811
    text = offers_csv(hand_run)
    rows = _parse(text)
    assert tuple(rows[0]) == OFFER_COLUMNS
    assert (
        len(rows)
        == 1 + Offer.objects.filter(period__market__simulation=hand_run).count()
    )
    assert rows[1] == ["1", "A", "R1", "100", "30.00", "1", "100.000", "40.00"]
    # rejected offer without acceptance price -> empty cell
    assert rows[2] == ["1", "A", "R2", "50", "50.00", "0", "0.000", ""]
    assert [r[1] for r in rows[1:]] == ["A", "A", "B", "A", "A", "B"]
    # round trip with DictReader and the decimal point
    records = list(csv.DictReader(io.StringIO(text)))
    assert sum(float(r["acceptance_amount"]) for r in records) == pytest.approx(210.5)


def test_step_logs_csv(hand_run: Simulation) -> None:  # noqa: F811
    assert _parse(step_logs_csv(hand_run)) == [list(STEP_LOG_COLUMNS)]
    agent = hand_run.market.agent_set.get(name="A")
    period = hand_run.market.period_set.get(periodNumber=2)
    AgentStepLog.objects.create(
        agent=agent,
        period=period,
        reward=0.125,
        critic_loss=None,
        actor_loss=-1.5,
        mean_q=2.0,
        exploration_scale=0.1,
        intervened_plants=[True, False],
        learning_active=True,
        evaluation_phase=False,
    )
    rows = _parse(step_logs_csv(hand_run))
    assert len(rows) == 2
    record = dict(zip(rows[0], rows[1]))
    assert record["period"] == "2"
    assert record["agent"] == "A"
    assert float(record["reward"]) == 0.125
    assert record["critic_loss"] == ""
    assert float(record["actor_loss"]) == -1.5
    assert json.loads(record["intervened_plants"]) == [True, False]
    assert record["learning_active"] == "1"
    assert record["evaluation_phase"] == "0"


def test_exports_of_a_run_without_data() -> None:
    from tests.factories import make_market

    market = make_market()
    simulation = market.simulation
    assert _parse(periods_csv(simulation)) == [list(PERIOD_COLUMNS)]
    assert _parse(offers_csv(simulation)) == [list(OFFER_COLUMNS)]
    assert _parse(step_logs_csv(simulation)) == [list(STEP_LOG_COLUMNS)]


def test_decimal_cells_keep_the_decimal_point(
    hand_run: Simulation,
) -> None:  # noqa: F811
    Period.objects.filter(market__simulation=hand_run, periodNumber=1).update(
        ptf=Decimal("12.5")
    )
    rows = _parse(periods_csv(hand_run))
    assert rows[1][3] == "12.50"
