"""Tests of the live budget and offer series of the watch page."""

from __future__ import annotations

import math
from collections import defaultdict
from decimal import Decimal

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from abmem.models import Simulation
from abmem.models.enums import SimulationState
from abmem.services.analysis.run_analysis import build_run_analysis
from abmem.services.simulation import simulation_service
from abmem.services.simulation.live_series import live_series
from tests.analysis.test_run_analysis import hand_run  # noqa: F401  (fixture)
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_resource,
    make_simulation,
)
from tests.runtime.conftest import (  # noqa: F401  (pytest fixtures)
    fast_demand,
    make_run,
    resources,
)

pytestmark = pytest.mark.django_db

WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE")


def _url(simulation: Simulation) -> str:
    return reverse("simulation_live_series", args=[simulation.pk])


@pytest.fixture
def baseline_run(fast_demand) -> Simulation:  # noqa: F811
    """A real FINISHED 20-period run with three baseline agents."""
    r = resources(3)
    simulation = make_run(
        [
            {
                "plants": [(r[0], 400), (r[1], 200)],
                "algorithm": "MARGINAL_COST",
                "budget": Decimal("1234.56"),
            },
            {
                "plants": [(r[2], 300)],
                "algorithm": "RANDOM",
                "budget": Decimal("-50.25"),
            },
            {"plants": [(r[0], 250)], "algorithm": "RANDOM"},
        ],
        period_number=20,
        seed=3,
    )
    simulation_service.run(simulation.id, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    return simulation


# shape -------------------------------------------------------------------


def test_endpoint_shape_and_values(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    response = client.get(_url(hand_run))
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"after", "periods", "ptf", "agents"}
    assert body["after"] == 0
    assert body["periods"] == [1, 2]
    assert body["ptf"] == [40.0, 55.0]
    a, b = body["agents"]
    assert set(a) == {
        "id",
        "name",
        "budget",
        "budget_before",
        "mean_bid",
        "offers",
        "accepted_offers",
    }
    assert [a["name"], b["name"]] == ["A", "B"]
    assert a["id"] < b["id"]
    assert a["budget"] == pytest.approx([3900.0, 9272.5])
    assert a["budget_before"] == pytest.approx(1000.0)
    assert b["budget"] == pytest.approx([300.0, 100.0])
    assert b["budget_before"] == pytest.approx(500.0)
    assert a["mean_bid"] == pytest.approx([40.0, 37.5])
    assert b["mean_bid"] == pytest.approx([45.0, 60.0])
    assert a["offers"] == [2, 2] and a["accepted_offers"] == [1, 2]
    assert b["offers"] == [1, 1] and b["accepted_offers"] == [0, 0]


def test_incremental_after(client: Client, hand_run: Simulation) -> None:  # noqa: F811
    body = client.get(_url(hand_run), {"after": 1}).json()
    assert body["after"] == 1
    assert body["periods"] == [2]
    assert body["ptf"] == [55.0]
    a, b = body["agents"]
    assert a["budget"] == pytest.approx([9272.5])
    assert a["budget_before"] == pytest.approx(3900.0)
    assert a["mean_bid"] == pytest.approx([37.5])
    assert b["budget"] == pytest.approx([100.0])

    empty = client.get(_url(hand_run), {"after": 2}).json()
    assert empty["periods"] == [] and empty["ptf"] == []
    assert [agent["budget"] for agent in empty["agents"]] == [[], []]
    assert [agent["budget_before"] for agent in empty["agents"]] == [None, None]


@pytest.mark.parametrize("value", ["x", "1.5", "-1"])
def test_invalid_after_returns_400(
    client: Client, hand_run: Simulation, value: str  # noqa: F811
) -> None:
    response = client.get(_url(hand_run), {"after": value})
    assert response.status_code == 400
    assert "after" in response.json()["error"]


# states ------------------------------------------------------------------


def test_run_without_periods_gives_empty_series(client: Client) -> None:
    simulation = make_simulation(name="fresh", period_number=5)
    market = make_market(simulation)
    make_agent_with_plants(market, [(make_resource(name="R1"), 100)], name="A")
    body = client.get(_url(simulation)).json()
    assert body["periods"] == [] and body["ptf"] == []
    assert len(body["agents"]) == 1
    assert body["agents"][0]["budget"] == []
    assert body["agents"][0]["mean_bid"] == []
    assert body["agents"][0]["budget_before"] is None


@pytest.mark.parametrize("state", [SimulationState.STARTED, SimulationState.FAILED])
def test_unfinished_runs_are_served(
    client: Client, hand_run: Simulation, state: SimulationState  # noqa: F811
) -> None:
    hand_run.state = state
    hand_run.save()
    response = client.get(_url(hand_run))
    assert response.status_code == 200
    assert response.json()["periods"] == [1, 2]


def test_scenario_template_and_unknown_simulation_return_404(client: Client) -> None:
    proxy = make_simulation(name="tmpl")
    proxy.proxy = True
    proxy.save()
    make_market(proxy)
    assert client.get(_url(proxy)).status_code == 404
    missing = reverse("simulation_live_series", args=[proxy.pk + 1000])
    assert client.get(missing).status_code == 404


def test_post_is_rejected(client: Client, hand_run: Simulation) -> None:  # noqa: F811
    assert client.post(_url(hand_run)).status_code == 405


# agreement with the analysis tab ------------------------------------------


def _mean_bids(analysis) -> dict[str, dict[int, float]]:
    prices: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in analysis.offers:
        prices[row.agent][row.period].append(row.offer_price)
    return {
        agent: {
            period: math.fsum(values) / len(values) for period, values in by.items()
        }
        for agent, by in prices.items()
    }


def test_full_series_matches_the_analysis(baseline_run: Simulation) -> None:
    analysis = build_run_analysis(baseline_run)
    series = live_series(baseline_run, 0)
    assert series.periods == [row.period for row in analysis.periods]
    assert series.ptf == [row.ptf for row in analysis.periods]
    means = _mean_bids(analysis)
    for agent in series.agents:
        expected = analysis.budgets[agent.name]
        assert agent.budget_before == expected[0]
        assert agent.budget == expected[1:]
        assert agent.mean_bid == pytest.approx(
            [means[agent.name][period] for period in series.periods], rel=1e-12
        )
        accepted = defaultdict(int)
        for row in analysis.offers:
            if row.agent == agent.name and row.accepted:
                accepted[row.period] += 1
        assert agent.accepted_offers == [accepted[p] for p in series.periods]


def test_incremental_chunks_concatenate_to_the_full_series(
    baseline_run: Simulation,
) -> None:
    full = live_series(baseline_run, 0)
    periods: list[int] = []
    ptf: list[float | None] = []
    budgets: dict[str, list[float]] = defaultdict(list)
    bids: dict[str, list[float | None]] = defaultdict(list)
    for after in (0, 3, 4, 11, 19, 20):
        upper = {0: 3, 3: 4, 4: 11, 11: 19, 19: 20, 20: 20}[after]
        chunk = live_series(baseline_run, after)
        chunk_periods = [p for p in chunk.periods if p <= upper]
        periods += chunk_periods
        ptf += chunk.ptf[: len(chunk_periods)]
        for agent in chunk.agents:
            budgets[agent.name] += agent.budget[: len(chunk_periods)]
            bids[agent.name] += agent.mean_bid[: len(chunk_periods)]
    assert periods == full.periods
    assert ptf == full.ptf
    for agent in full.agents:
        assert budgets[agent.name] == agent.budget
        assert bids[agent.name] == agent.mean_bid


# queries -----------------------------------------------------------------


def _endpoint_queries(client: Client, simulation: Simulation, after: int) -> list[str]:
    with CaptureQueriesContext(connection) as ctx:
        response = client.get(_url(simulation), {"after": after})
    assert response.status_code == 200
    return [query["sql"] for query in ctx.captured_queries]


def test_query_count_is_constant_and_read_only(
    client: Client, hand_run: Simulation, baseline_run: Simulation  # noqa: F811
) -> None:
    small = _endpoint_queries(client, hand_run, 0)
    large = _endpoint_queries(client, baseline_run, 0)
    incremental = _endpoint_queries(client, baseline_run, 10)
    assert len(small) == len(large) == len(incremental) == 4
    for sql in small + large + incremental:
        assert not sql.lstrip().upper().startswith(WRITE_PREFIXES), sql
    # No new period: the resource query is skipped.
    assert len(_endpoint_queries(client, baseline_run, 20)) == 3


# watch page --------------------------------------------------------------


def test_watch_page_has_the_live_charts(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    html = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": hand_run.pk})
    ).content.decode()
    for element_id in (
        "live-charts",
        "live-charts-note",
        "live-chart-budgets",
        "live-chart-offers",
    ):
        assert f'id="{element_id}"' in html, element_id
    assert _url(hand_run) in html
    assert "chart-tools.js" in html
    assert "chartjs-plugin-zoom.min.js" in html
    assert "update('none')" in html


def test_watch_page_of_a_template_has_no_live_charts(client: Client) -> None:
    proxy = make_simulation(name="tmpl")
    proxy.proxy = True
    proxy.save()
    make_market(proxy)
    html = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": proxy.pk})
    ).content.decode()
    assert 'id="live-charts"' not in html
    assert "/live-series/" not in html
    assert 'var LIVE_URL = "" || null;' in html
