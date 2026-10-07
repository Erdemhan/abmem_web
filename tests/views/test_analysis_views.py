"""Tests of the run analysis views (PLAN 2.5, spec run_analysis)."""

from __future__ import annotations

import csv
import io
import json
import pathlib
import re
import shutil
import subprocess

import pytest
from django.test import Client
from django.urls import reverse

from abmem.models import Simulation
from abmem.models.enums import SimulationState
from tests.analysis.test_run_analysis import hand_run  # noqa: F401  (fixture)
from tests.factories import make_market, make_simulation

pytestmark = pytest.mark.django_db

NODE = shutil.which("node")


def test_analysis_page_renders_for_a_finished_run(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    response = client.get(reverse("simulation_analysis", args=[hand_run.pk]))
    assert response.status_code == 200
    html = response.content.decode()
    assert 'id="analysis-config"' in html
    assert reverse("simulation_analysis_data", args=[hand_run.pk]) in html
    for kind in ("periods", "offers", "steps"):
        assert reverse("simulation_export_csv", args=[hand_run.pk, kind]) in html
    assert "zip-store.js" in html
    assert "chartjs.min.js" in html
    assert "Download all charts" in html
    assert "alert(" not in html and "confirm(" not in html
    for key in ("market", "budgets", "offers", "panel", "learning", "scores", "legacy"):
        assert f'id="chart-{key}"' in html
        assert f'data-chart="{key}"' in html


def test_analysis_data_endpoint(
    client: Client, hand_run: Simulation
) -> None:  # noqa: F811
    response = client.get(reverse("simulation_analysis_data", args=[hand_run.pk]))
    assert response.status_code == 200
    data = response.json()
    assert set(data) >= {
        "settings",
        "periods",
        "offers",
        "budgets",
        "step_logs",
        "agents",
        "corrected",
        "legacy",
    }
    assert data["settings"]["seed"] == 123
    assert data["budgets"]["A"] == pytest.approx([1000.0, 3900.0, 9272.5])
    assert data["legacy"]["price_cap"] == 200.0
    assert data["corrected"]["evaluation_window"]["first_period"] == 2


@pytest.mark.parametrize("state", [SimulationState.STARTED, SimulationState.FAILED])
def test_unfinished_run_returns_409(
    client: Client, hand_run: Simulation, state: SimulationState  # noqa: F811
) -> None:
    hand_run.state = state
    hand_run.save()
    page = client.get(reverse("simulation_analysis", args=[hand_run.pk]))
    assert page.status_code == 409
    assert "FINISHED" in page.content.decode()
    assert 'id="analysis-config"' not in page.content.decode()
    data = client.get(reverse("simulation_analysis_data", args=[hand_run.pk]))
    assert data.status_code == 409
    assert "FINISHED" in data.json()["error"]
    export = client.get(reverse("simulation_export_csv", args=[hand_run.pk, "offers"]))
    assert export.status_code == 409


def test_purged_run_returns_409(client: Client) -> None:
    simulation = make_simulation(name="purged", period_number=4)
    simulation.state = SimulationState.FINISHED
    simulation.save()
    make_market(simulation)
    for name in ("simulation_analysis", "simulation_analysis_data"):
        response = client.get(reverse(name, args=[simulation.pk]))
        assert response.status_code == 409
    assert (
        "no stored periods"
        in client.get(reverse("simulation_analysis_data", args=[simulation.pk])).json()[
            "error"
        ]
    )


def test_unknown_and_proxy_simulations_return_404(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    for name in ("simulation_analysis", "simulation_analysis_data"):
        assert client.get(reverse(name, args=[999999])).status_code == 404
    assert (
        client.get(
            reverse("simulation_export_csv", args=[999999, "offers"])
        ).status_code
        == 404
    )
    hand_run.proxy = True
    hand_run.save()
    assert (
        client.get(reverse("simulation_analysis", args=[hand_run.pk])).status_code
        == 404
    )


@pytest.mark.parametrize(
    "kind,header",
    [
        ("periods", "period,demand,met_demand,ptf,market_volume"),
        (
            "offers",
            "period,agent,resource,amount,offer_price,accepted,acceptance_amount,acceptance_price",
        ),
        (
            "steps",
            "period,agent,reward,critic_loss,actor_loss,mean_q,exploration_scale,intervened_plants,learning_active,evaluation_phase",
        ),
    ],
)
def test_csv_export(
    client: Client, hand_run: Simulation, kind: str, header: str
) -> None:  # noqa: F811
    response = client.get(reverse("simulation_export_csv", args=[hand_run.pk, kind]))
    assert response.status_code == 200
    assert response["Content-Type"] == "text/csv; charset=utf-8"
    assert (
        response["Content-Disposition"]
        == f'attachment; filename="simulation_{hand_run.pk}_{kind}.csv"'
    )
    text = response.content.decode("utf-8")
    assert text.splitlines()[0] == header
    rows = list(csv.reader(io.StringIO(text)))
    expected = {"periods": 2, "offers": 6, "steps": 0}[kind]
    assert len(rows) == 1 + expected


def test_unknown_csv_kind_returns_404(
    client: Client, hand_run: Simulation
) -> None:  # noqa: F811
    response = client.get(f"/dev/simulation/{hand_run.pk}/export/agents.csv")
    assert response.status_code == 404


def test_analysis_views_are_get_only(
    client: Client, hand_run: Simulation
) -> None:  # noqa: F811
    response = client.post(reverse("simulation_analysis_data", args=[hand_run.pk]))
    assert response.status_code == 405


# navigation -------------------------------------------------------------


def test_run_list_shows_seed_and_analysis_button(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    html = client.get(reverse("simulation_list")).content.decode()
    assert ">Seed<" in html
    assert '<h6 class="mb-0 text-sm">123</h6>' in html
    assert reverse("simulation_analysis", args=[hand_run.pk]) in html
    hand_run.state = SimulationState.STARTED
    hand_run.save()
    html = client.get(reverse("simulation_list")).content.decode()
    assert reverse("simulation_analysis", args=[hand_run.pk]) not in html


def test_watch_page_shows_seed_and_analysis_link(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    html = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": hand_run.pk})
    ).content.decode()
    assert '<span id="seed">123</span>' in html
    assert 'id="analysis-button"' in html
    assert reverse("simulation_analysis", args=[hand_run.pk]) in html


def test_watch_page_of_a_scenario_has_no_analysis_link(client: Client) -> None:
    simulation = make_simulation(name="tmpl")
    simulation.proxy = True
    simulation.save()
    html = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": simulation.pk})
    ).content.decode()
    assert 'id="analysis-button"' not in html


# inline scripts ---------------------------------------------------------


def _inline_scripts(html: str) -> list[str]:
    return [
        body
        for attrs, body in re.findall(
            r"<script([^>]*)>(.*?)</script>", html, flags=re.S
        )
        if "src=" not in attrs and "application/json" not in attrs and body.strip()
    ]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_inline_scripts_pass_node_check(
    client: Client, hand_run: Simulation, tmp_path: pathlib.Path  # noqa: F811
) -> None:
    proxy = make_simulation(name="tmpl")
    proxy.proxy = True
    proxy.save()
    make_market(proxy)
    pages = {
        "analysis": reverse("simulation_analysis", args=[hand_run.pk]),
        "list_proxy": reverse("simulation_list_proxy"),
        "watch": reverse("watch_simulation", kwargs={"simulation_id": hand_run.pk}),
    }
    for name, url in pages.items():
        scripts = _inline_scripts(client.get(url).content.decode())
        assert scripts, name
        for index, body in enumerate(scripts):
            path = tmp_path / f"{name}_{index}.js"
            path.write_text(body, encoding="utf-8")
            result = subprocess.run(
                [NODE, "--check", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            assert result.returncode == 0, f"{name} script {index}: {result.stderr}"
    zip_store = (
        pathlib.Path(__file__).resolve().parents[2]
        / "abmem"
        / "static"
        / "assets"
        / "js"
        / "zip-store.js"
    )
    result = subprocess.run(
        [NODE, "--check", str(zip_store)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_analysis_config_is_json(
    client: Client, hand_run: Simulation
) -> None:  # noqa: F811
    html = client.get(
        reverse("simulation_analysis", args=[hand_run.pk])
    ).content.decode()
    match = re.search(
        r'<script id="analysis-config" type="application/json">(.*?)</script>',
        html,
        flags=re.S,
    )
    assert match
    config = json.loads(match.group(1))
    assert config == {
        "data_url": reverse("simulation_analysis_data", args=[hand_run.pk]),
        "simulation_id": hand_run.pk,
        "offers_url": reverse("simulation_analysis_offers", args=[hand_run.pk]),
        "offer_agents": [
            {"id": agent.pk, "name": agent.name}
            for agent in hand_run.market.agent_set.order_by("id")
        ],
    }
