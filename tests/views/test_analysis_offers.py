"""Tests of the offers table endpoint of the analysis tab (PLAN 2.11)."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
from decimal import Decimal

import pytest
from django.db import connection
from django.test import Client
from django.test.utils import CaptureQueriesContext
from django.urls import reverse

from abmem.models import Agent, Simulation
from abmem.models.enums import SimulationState
from abmem.services.analysis.offer_table import (
    DEFAULT_PAGE_SIZE,
    OfferFilterError,
    parse_offer_filter,
)
from tests.analysis.test_run_analysis import hand_run  # noqa: F401  (fixture)
from tests.factories import (
    make_agent_with_plants,
    make_market,
    make_offer,
    make_period,
    make_resource,
    make_simulation,
)

pytestmark = pytest.mark.django_db


def _get(client: Client, sim: Simulation, **params):
    return client.get(reverse("simulation_analysis_offers", args=[sim.pk]), params)


def _agent_id(sim: Simulation, name: str) -> int:
    return Agent.objects.get(market__simulation=sim, name=name).pk


# parsing ----------------------------------------------------------------


def test_parse_defaults() -> None:
    offer_filter = parse_offer_filter({})
    assert offer_filter.page == 1
    assert offer_filter.page_size == DEFAULT_PAGE_SIZE == 10
    assert offer_filter.accepted == "all"
    assert offer_filter.agent_id is None


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"page_size": "20"}, "page_size"),
        ({"page_size": "abc"}, "page_size"),
        ({"page": "0"}, "page"),
        ({"page": "x"}, "page"),
        ({"agent": "x"}, "agent"),
        ({"period_from": "-1"}, "period_from"),
        ({"period_from": "5", "period_to": "2"}, "period_to"),
        ({"accepted": "maybe"}, "accepted"),
    ],
)
def test_parse_rejects_invalid(params: dict[str, str], field: str) -> None:
    with pytest.raises(OfferFilterError) as info:
        parse_offer_filter(params)
    assert field in info.value.errors


# endpoint ---------------------------------------------------------------


def test_default_page_lists_offers_in_order(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    response = _get(client, hand_run)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 6
    assert body["page"] == 1 and body["page_size"] == 10 and body["num_pages"] == 1
    assert [(r["period"], r["agent"], r["resource"]) for r in body["rows"]] == [
        (1, "A", "R1"),
        (1, "A", "R2"),
        (1, "B", "R3"),
        (2, "A", "R1"),
        (2, "A", "R2"),
        (2, "B", "R3"),
    ]
    first = body["rows"][0]
    assert first == {
        "period": 1,
        "agent": "A",
        "resource": "R1",
        "quantity": 100,
        "accepted_quantity": 100.0,
        "offer_price": 30.0,
        "acceptance_price": 40.0,
        "accepted": True,
        "profit": 2900.0,
    }
    assert body["rows"][2]["acceptance_price"] is None
    assert body["rows"][2]["profit"] == -200.0


def test_filters(client: Client, hand_run: Simulation) -> None:  # noqa: F811
    b = _agent_id(hand_run, "B")
    body = _get(client, hand_run, agent=b).json()
    assert body["count"] == 2 and {r["agent"] for r in body["rows"]} == {"B"}

    body = _get(client, hand_run, period_from=2).json()
    assert body["count"] == 3 and {r["period"] for r in body["rows"]} == {2}
    body = _get(client, hand_run, period_to=1).json()
    assert body["count"] == 3 and {r["period"] for r in body["rows"]} == {1}
    body = _get(client, hand_run, period_from=1, period_to=1, accepted="yes").json()
    assert body["count"] == 1 and body["rows"][0]["resource"] == "R1"

    body = _get(client, hand_run, accepted="yes").json()
    assert body["count"] == 3 and all(r["accepted"] for r in body["rows"])
    body = _get(client, hand_run, accepted="no").json()
    assert body["count"] == 3 and not any(r["accepted"] for r in body["rows"])
    body = _get(client, hand_run, accepted="all").json()
    assert body["count"] == 6


def test_unknown_agent_is_400(
    client: Client, hand_run: Simulation
) -> None:  # noqa: F811
    other = make_simulation(name="other")
    market = make_market(other)
    stranger = make_agent_with_plants(market, [(make_resource(), 10)], name="X")
    response = _get(client, hand_run, agent=stranger.pk)
    assert response.status_code == 400
    assert "agent" in response.json()["errors"]


@pytest.mark.parametrize(
    "params",
    [{"page_size": "7"}, {"page": "0"}, {"period_from": "3", "period_to": "1"}],
)
def test_invalid_parameters_are_400(
    client: Client, hand_run: Simulation, params: dict[str, str]  # noqa: F811
) -> None:
    response = _get(client, hand_run, **params)
    assert response.status_code == 400
    assert response.json()["errors"]


def test_unfinished_run_is_409_and_template_is_404(client: Client) -> None:
    running = make_simulation(name="running")
    running.state = SimulationState.STARTED
    running.save()
    make_market(running)
    assert _get(client, running).status_code == 409
    template = make_simulation(name="tmpl")
    template.proxy = True
    template.save()
    assert _get(client, template).status_code == 404


@pytest.fixture
def big_run() -> Simulation:
    """A FINISHED run with 2 agents, 30 periods and 120 offers."""
    simulation = make_simulation(name="big", period_number=30, current_period=30)
    simulation.state = SimulationState.FINISHED
    simulation.save()
    market = make_market(simulation)
    r1 = make_resource(name="BR1")
    r2 = make_resource(name="BR2")
    a = make_agent_with_plants(market, [(r1, 10), (r2, 10)], name="A")
    b = make_agent_with_plants(market, [(r1, 10), (r2, 10)], name="B")
    for t in range(1, 31):
        period = make_period(market, t, ptf=Decimal("40"))
        for agent in (a, b):
            for resource in (r1, r2):
                accepted = (t + agent.pk + resource.pk) % 2 == 0
                make_offer(
                    period,
                    agent,
                    resource,
                    10,
                    Decimal("35"),
                    acceptance=accepted,
                    acceptance_price=Decimal("40") if accepted else None,
                    acceptance_amount=10 if accepted else 0,
                    save=True,
                )
    return simulation


@pytest.mark.parametrize("page_size", [10, 25, 50, 100])
def test_allowed_page_sizes(
    client: Client, big_run: Simulation, page_size: int
) -> None:
    body = _get(client, big_run, page_size=page_size).json()
    assert body["count"] == 120
    assert body["page_size"] == page_size
    assert len(body["rows"]) == min(page_size, 120)
    assert body["num_pages"] == -(-120 // page_size)


def test_pagination_and_last_page_clamp(client: Client, big_run: Simulation) -> None:
    body = _get(client, big_run, page=2).json()
    assert body["page"] == 2 and body["num_pages"] == 12
    assert [r["period"] for r in body["rows"]][:1] == [3]
    body = _get(client, big_run, page=999).json()
    assert body["page"] == 12 and len(body["rows"]) == 10


def test_query_count_is_constant_per_page(client: Client, big_run: Simulation) -> None:
    agent = _agent_id(big_run, "A")
    counts = []
    for params in (
        {"page_size": 10},
        {"page_size": 100},
        {"page_size": 100, "agent": agent},
    ):
        with CaptureQueriesContext(connection) as queries:
            assert _get(client, big_run, **params).status_code == 200
        counts.append(len(queries))
    assert counts[0] == counts[1]
    # The agent filter adds one existence check.
    assert counts[2] == counts[1] + 1
    assert counts[1] <= 5


# page -------------------------------------------------------------------


def test_analysis_page_has_offers_card_and_zoom_assets(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    html = client.get(
        reverse("simulation_analysis", args=[hand_run.pk])
    ).content.decode()
    assert reverse("simulation_analysis_offers", args=[hand_run.pk]) in html
    assert reverse("simulation_export_csv", args=[hand_run.pk, "offers"]) in html
    for element_id in (
        "an-offer-agent",
        "an-offer-from",
        "an-offer-to",
        "an-offer-accepted",
        "an-offer-size",
        "an-offer-first",
        "an-offer-prev",
        "an-offer-next",
        "an-offer-last",
        "an-offer-page",
        "an-offers",
        "an-chart-modal",
    ):
        assert f'id="{element_id}"' in html, element_id
    assert '<option value="10" selected>10</option>' in html
    for script in ("hammer.min.js", "chartjs-plugin-zoom.min.js", "chart-tools.js"):
        assert script in html, script
    for key in ("market", "budgets", "offers", "panel", "learning", "scores", "legacy"):
        assert f'data-zoom-reset="{key}"' in html, key
        assert f'data-expand="{key}"' in html, key


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    "relative",
    [
        "chart-tools.js",
        "plugins/hammer.min.js",
        "plugins/chartjs-plugin-zoom.min.js",
    ],
)
def test_static_chart_scripts_pass_node_check(relative: str) -> None:
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "abmem"
        / "static"
        / "assets"
        / "js"
        / relative
    )
    result = subprocess.run(
        [shutil.which("node"), "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_vendored_plugins_keep_their_license_headers() -> None:
    plugins = (
        pathlib.Path(__file__).resolve().parents[2]
        / "abmem"
        / "static"
        / "assets"
        / "js"
        / "plugins"
    )
    zoom = (plugins / "chartjs-plugin-zoom.min.js").read_text(encoding="utf-8")
    hammer = (plugins / "hammer.min.js").read_text(encoding="utf-8")
    assert "chartjs-plugin-zoom v1.2.1" in zoom[:200] and "MIT" in zoom[:300]
    assert "Hammer.JS" in hammer[:200] and "MIT" in hammer[:300]


NODE_HELPER_CHECK = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const base = process.argv[2];
const ctx = { console, setTimeout, clearTimeout };
ctx.window = ctx; ctx.self = ctx; ctx.globalThis = ctx;
ctx.document = { createElement: () => ({ getContext: () => null, style: {} }), addEventListener() {} };
vm.createContext(ctx);
for (const f of ['plugins/chartjs.min.js', 'plugins/hammer.min.js',
                 'plugins/chartjs-plugin-zoom.min.js', 'chart-tools.js']) {
  vm.runInContext(fs.readFileSync(path.join(base, f), 'utf8'), ctx, { filename: f });
}
const plugin = { id: 'p', beforeDraw() {} };
const cfg = { data: { datasets: [{ data: [{ x: 1, y: 2 }] }] }, plugins: [plugin] };
const copy = ctx.ChartTools.copyConfig(cfg);
console.log(JSON.stringify({
  version: ctx.Chart.version,
  zoom: ctx.ChartTools.zoomAvailable(),
  deep: copy.data.datasets[0].data[0] !== cfg.data.datasets[0].data[0],
  equal: JSON.stringify(copy) === JSON.stringify(cfg),
  sharedFunction: copy.plugins[0].beforeDraw === plugin.beforeDraw,
}));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_zoom_plugin_registers_with_bundled_chartjs(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "check.js"
    script.write_text(NODE_HELPER_CHECK, encoding="utf-8")
    base = (
        pathlib.Path(__file__).resolve().parents[2]
        / "abmem"
        / "static"
        / "assets"
        / "js"
    )
    result = subprocess.run(
        [shutil.which("node"), str(script), str(base)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report == {
        "version": "3.7.1",
        "zoom": True,
        "deep": True,
        "equal": True,
        "sharedFunction": True,
    }
