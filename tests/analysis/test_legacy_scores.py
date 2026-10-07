"""Tests of the legacy composite score port (``compute_advanced_scores.py``)."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import pytest

from abmem.services.analysis.legacy_scores import legacy_composite_scores
from tests.analysis.legacy_reference import reference_scores


def _random_rows(seed: int, n_periods: int, agents: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    budgets = {f"A{i}": rng.uniform(-1000, 1000) for i in range(agents)}
    for period in range(1, n_periods + 1):
        ptf = rng.uniform(0, 200)
        for name in budgets:
            budgets[name] += rng.uniform(-5000, 20000)
            for _plant in range(rng.randint(1, 3)):
                price = rng.uniform(0, 220)
                rows.append(
                    {
                        "period": period,
                        "agent": name,
                        "amount": rng.randint(0, 600),
                        "offer_price": price,
                        "market_price": ptf,
                        "accepted": price <= ptf,
                        "budget": budgets[name],
                    }
                )
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_the_legacy_script_on_random_rows(seed: int) -> None:
    rows = _random_rows(seed, n_periods=12, agents=4)
    scores, market = legacy_composite_scores(rows)
    ref_scores, ref_market = reference_scores(pd.DataFrame(rows))
    assert list(scores) == list(ref_scores)
    for name, value in ref_scores.items():
        assert scores[name] == pytest.approx(float(value), rel=1e-12, abs=1e-12)
    assert market == pytest.approx(float(ref_market), rel=1e-12)


def test_hand_computed_single_agent() -> None:
    rows = [
        {
            "period": 1,
            "agent": "X",
            "amount": 100,
            "offer_price": 50.0,
            "market_price": 60.0,
            "accepted": True,
            "budget": 1000.0,
        },
        {
            "period": 2,
            "agent": "X",
            "amount": 100,
            "offer_price": 70.0,
            "market_price": 60.0,
            "accepted": False,
            "budget": 2000.0,
        },
    ]
    scores, market = legacy_composite_scores(rows)
    # G = 1000, capacity 100, T = 2: norm_budget = 1000 / 40000 = 0.025,
    # Q = 100: norm_profit = (1000 / 100) / 200 = 0.05,
    # RMSE = 10: similarity = 0.95, score = 0.01 + 0.02 + 0.19 = 0.22.
    assert scores == {"X": pytest.approx(0.22)}
    # m = 60, o = 60: 0.6 * 0.7 + 0.4 * 1 = 0.82
    assert market == pytest.approx(0.82)


def test_no_accepted_offer_and_clipping() -> None:
    rows = [
        {
            "period": 1,
            "agent": "L",
            "amount": 10,
            "offer_price": 200.0,
            "market_price": 0.0,
            "accepted": False,
            "budget": 0.0,
        },
        {
            "period": 2,
            "agent": "L",
            "amount": 10,
            "offer_price": 200.0,
            "market_price": 0.0,
            "accepted": False,
            "budget": -1e9,
        },
    ]
    scores, market = legacy_composite_scores(rows)
    # norm_budget clipped to -1, norm_profit 0, similarity 0 -> clipped to 0.
    assert scores == {"L": 0.0}
    assert market == pytest.approx(0.6)


def test_zero_capacity_and_zero_offer_prices() -> None:
    rows = [
        {
            "period": 1,
            "agent": "Z",
            "amount": 0,
            "offer_price": 0.0,
            "market_price": 0.0,
            "accepted": True,
            "budget": 5.0,
        },
    ]
    scores, market = legacy_composite_scores(rows)
    assert scores == {"Z": pytest.approx(0.2)}
    assert market == pytest.approx(0.6)


def test_empty_rows_raise() -> None:
    with pytest.raises(ValueError):
        legacy_composite_scores([])
