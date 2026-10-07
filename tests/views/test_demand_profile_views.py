"""Demand profile in the wizard, the preview endpoint, YAML and the JS component (PLAN 2.8)."""

from __future__ import annotations

import copy
import json
import pathlib
import shutil
import subprocess
from typing import Any, Iterator

import numpy as np
import pytest
import yaml
from django.test import Client
from django.urls import reverse

from abmem import scenario_yaml, views_scenario
from abmem.models import Simulation
from abmem.models.enums import EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from abmem.services.market.demand import DemandSeries
from abmem.services.market.demand_generator import DemandProfile, generate_demand
from tests.factories import make_resource
from tests.views.test_scenario_wizard import _create, _new_agent, _payload, _post
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

NODE = shutil.which("node")
JS_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "abmem"
    / "static"
    / "assets"
    / "js"
    / "demand-profile.js"
)

PROFILE: dict[str, Any] = {
    "version": 1,
    "seed": 314,
    "day_start_hour": 6,
    "day_end_hour": 22,
    "autocorrelation": 0.7,
    "morning_ramp_hours": 0,
    "evening_ramp_hours": 0,
    "seasons": None,
    "day": {
        "distribution": "LOGNORMAL",
        "mean": 900.0,
        "std": 120.0,
        "min": 600.0,
        "max": 1300.0,
    },
    "night": {"distribution": "UNIFORM", "min": 400.0, "max": 650.0},
}


@pytest.fixture
def fossil() -> Resource:
    return make_resource(name="dp-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


@pytest.fixture(autouse=True)
def _fresh_demand_cache() -> Iterator[None]:
    views_scenario.load_demand_series.cache_clear()
    yield
    views_scenario.load_demand_series.cache_clear()


def _fake_file(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    series = DemandSeries(values=np.array(values), source="test")
    monkeypatch.setattr(
        DemandSeries, "from_excel", staticmethod(lambda *a, **k: series)
    )


def _preview(client: Client, body: dict[str, Any]) -> Any:
    return client.post(
        reverse("demand_preview"), json.dumps(body), content_type="application/json"
    )


# wizard round trip ---------------------------------------------------------------


def test_create_stores_the_normalised_profile(client: Client, fossil: Resource) -> None:
    raw = copy.deepcopy(PROFILE)
    del raw["version"]
    raw["seed"] = "314"
    simulation = _create(client, _payload([_new_agent(fossil)], demand_profile=raw))
    assert simulation.demand_profile == PROFILE


def test_create_without_profile_is_the_file(client: Client, fossil: Resource) -> None:
    simulation = _create(client, _payload([_new_agent(fossil)]))
    assert simulation.demand_profile is None
    config = wizard_config(client.get(reverse("create_simulation")).content.decode())
    assert config["initial"]["demand_profile"] is None
    assert config["demand_preview_url"] == reverse("demand_preview")


def test_update_round_trip(client: Client, fossil: Resource) -> None:
    simulation = _create(client, _payload([_new_agent(fossil)], demand_profile=PROFILE))
    url = reverse("update_simulation", args=[simulation.id])
    html = client.get(url).content.decode()
    assert wizard_config(html)["initial"]["demand_profile"] == PROFILE
    assert 'id="demand-component"' in html and 'id="review-demand-chart"' in html
    assert "assets/js/demand-profile.js" in html and "chartjs.min.js" in html

    agents = [
        {"mode": "scenario", "agent_id": a.id, **_new_agent(fossil, name=a.name)}
        for a in simulation.market.agent_set.all()
    ]
    for agent in agents:
        agent["mode"] = "scenario"
    # A payload without the key keeps the stored profile.
    response = _post(client, url, _payload(agents))
    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert simulation.demand_profile == PROFILE
    # null switches back to the file.
    response = _post(client, url, _payload(agents, demand_profile=None))
    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert simulation.demand_profile is None


def test_invalid_profile_is_rejected_with_field_keys(
    client: Client, fossil: Resource
) -> None:
    bad = copy.deepcopy(PROFILE)
    bad["day"]["std"] = 0
    bad["day_end_hour"] = 6
    before = Simulation.objects.count()
    response = _post(
        client,
        reverse("create_simulation"),
        _payload([_new_agent(fossil)], demand_profile=bad),
    )
    assert response.status_code == 400
    errors = response.json()["errors"]
    assert "demand_profile.day.std" in errors
    assert "demand_profile.day_end_hour" in errors
    assert Simulation.objects.count() == before


def test_clone_for_run_copies_the_profile(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)], demand_profile=PROFILE))
    run = library_service.clone_scenario_for_run(scenario, seed=5)
    assert run.demand_profile == PROFILE
    duplicate = library_service.duplicate_scenario(scenario)
    assert duplicate.demand_profile == PROFILE


# preview endpoint ----------------------------------------------------------------


def test_generated_preview_matches_the_generator(client: Client) -> None:
    response = _preview(client, {"periods": 48, "profile": PROFILE})
    assert response.status_code == 200, response.content
    body = response.json()
    expected = generate_demand(DemandProfile.from_dict(PROFILE), 48)
    assert body["source"] == "generated" and body["periods"] == 48
    assert body["values"] == [float(v) for v in expected]
    assert body["night_mask"] == [not (6 <= t % 24 < 22) for t in range(48)]
    stats = body["stats"]
    assert stats["max"] == pytest.approx(float(expected.max()))
    assert stats["std"] == pytest.approx(float(expected.std()))
    assert stats["day"]["count"] == 32 and stats["night"]["count"] == 16
    assert stats["night"]["max"] <= 650.0 and stats["day"]["min"] >= 600.0


def test_preview_prefix_matches_a_longer_horizon(client: Client) -> None:
    short = _preview(client, {"periods": 168, "profile": PROFILE}).json()["values"]
    long = _preview(client, {"periods": 336, "profile": PROFILE}).json()["values"]
    assert long[:168] == short


def test_generated_preview_by_get(client: Client) -> None:
    response = client.get(
        reverse("demand_preview"), {"periods": "24", "profile": json.dumps(PROFILE)}
    )
    assert response.status_code == 200
    assert response.json()["source"] == "generated"


def test_file_preview(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_file(monkeypatch, [10.0, 30.0, 20.0, 1000.0])
    response = client.get(reverse("demand_preview"), {"periods": "3", "source": "file"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "file"
    assert body["values"] == [10.0, 30.0, 20.0]
    assert body["available_periods"] == 4
    assert body["night_mask"] == [True, True, True]  # hours 0-2, default day 08-20
    assert body["stats"]["mean"] == 20.0 and body["stats"]["day"] is None
    body = _preview(
        client, {"periods": 3, "source": "file", "day_start_hour": 0, "day_end_hour": 2}
    ).json()
    assert body["night_mask"] == [False, False, True]


@pytest.mark.parametrize(
    "body, key",
    [
        ({"periods": 0, "profile": PROFILE}, "periods"),
        ({"periods": "abc", "source": "file"}, "periods"),
        ({"periods": 20000, "profile": PROFILE}, "periods"),
        ({"periods": 24, "source": "generated"}, "demand_profile"),
        ({"periods": 24, "source": "other"}, "source"),
        (
            {"periods": 24, "profile": dict(PROFILE, autocorrelation=1)},
            "demand_profile.autocorrelation",
        ),
        (
            {
                "periods": 24,
                "profile": dict(
                    PROFILE, night={"distribution": "NORMAL", "min": 1, "max": 2}
                ),
            },
            "demand_profile.night.mean",
        ),
    ],
)
def test_preview_rejects_bad_requests(
    client: Client, body: dict[str, Any], key: str
) -> None:
    response = _preview(client, body)
    assert response.status_code == 400
    assert key in response.json()["errors"]


def test_file_preview_beyond_the_data(
    client: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_file(monkeypatch, [10.0, 20.0])
    response = _preview(client, {"periods": 3, "source": "file"})
    assert response.status_code == 400
    assert "covers 2 periods" in response.json()["errors"]["periods"][0]


def test_preview_rejects_invalid_json(client: Client) -> None:
    response = client.post(
        reverse("demand_preview"), "{", content_type="application/json"
    )
    assert response.status_code == 400
    response = client.get(reverse("demand_preview"), {"periods": "3", "profile": "{"})
    assert response.status_code == 400


# YAML ----------------------------------------------------------------------------


def test_yaml_round_trip_with_demand(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)], demand_profile=PROFILE))
    text = client.get(
        reverse("export_scenario_yaml", args=[scenario.id])
    ).content.decode()
    document = yaml.safe_load(text)
    assert document["scenario"]["demand"] == PROFILE
    preview = client.post(reverse("import_scenario_preview"), {"file": _upload(text)})
    assert preview.status_code == 200, preview.content
    assert preview.json()["scenario"]["demand"] == PROFILE
    response = client.post(reverse("import_scenario_confirm"), {"file": _upload(text)})
    assert response.status_code == 200, response.content
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.demand_profile == PROFILE


def test_yaml_without_demand_is_the_file(client: Client, fossil: Resource) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)]))
    document = scenario_yaml.scenario_document(scenario)
    assert "demand" not in document["scenario"]
    text = yaml.safe_dump(
        json.loads(json.dumps(document, default=str)), sort_keys=False
    )
    response = client.post(reverse("import_scenario_confirm"), {"file": _upload(text)})
    assert response.status_code == 200, response.content
    assert (
        Simulation.objects.get(id=response.json()["simulation_id"]).demand_profile
        is None
    )


def test_yaml_with_invalid_demand_writes_nothing(
    client: Client, fossil: Resource
) -> None:
    scenario = _create(client, _payload([_new_agent(fossil)], demand_profile=PROFILE))
    document = yaml.safe_load(
        client.get(reverse("export_scenario_yaml", args=[scenario.id])).content.decode()
    )
    document["scenario"]["demand"]["night"]["max"] = 100.0
    del document["scenario"]["demand"]["version"]
    text = yaml.safe_dump(document, sort_keys=False)
    before = Simulation.objects.count()
    for url in ("import_scenario_preview", "import_scenario_confirm"):
        response = client.post(reverse(url), {"file": _upload(text)})
        assert response.status_code == 400
        assert "scenario.demand.night.max" in response.json()["errors"]
    assert Simulation.objects.count() == before


def _upload(text: str) -> Any:
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile(
        "scenario.yaml", text.encode("utf-8"), content_type="application/x-yaml"
    )


# JS component --------------------------------------------------------------------

_JS_HARNESS = r"""
const fs = require('fs');
global.window = {};
eval(fs.readFileSync(process.argv[2], 'utf8'));
const api = window.DemandProfileForm;
const cases = JSON.parse(process.argv[3]);
const out = cases.map(p => api.validateProfile(p));
out.push(api.nightText(8, 20), api.nightText(0, 6), api.nightText(18, 24));
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_component_script_passes_node_check() -> None:
    result = subprocess.run(
        [NODE, "--check", str(JS_PATH)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_client_validation_mirrors_the_server(tmp_path: pathlib.Path) -> None:
    from django.core.exceptions import ValidationError

    cases: list[dict[str, Any]] = [copy.deepcopy(PROFILE)]
    for change in (
        {"seed": -1},
        {"seed": 2**31},
        {"day_start_hour": 24},
        {"day_end_hour": 6},
        {"day_start_hour": 0, "day_end_hour": 24},
        {"autocorrelation": 1},
        {
            "day": {
                "distribution": "NORMAL",
                "mean": 2000.0,
                "std": 10.0,
                "min": 600.0,
                "max": 1300.0,
            }
        },
        {
            "day": {
                "distribution": "LOGNORMAL",
                "mean": 900.0,
                "std": 0.0,
                "min": 600.0,
                "max": 1300.0,
            }
        },
        {"night": {"distribution": "UNIFORM", "min": 0.0, "max": 0.0}},
        {"night": {"distribution": "GAMMA", "min": 1.0, "max": 2.0}},
        {"night": {"distribution": "NORMAL", "min": 1.0, "max": 2.0}},
    ):
        cases.append({**copy.deepcopy(PROFILE), **change})
    harness = tmp_path / "harness.js"
    harness.write_text(_JS_HARNESS, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(harness), str(JS_PATH), json.dumps(cases)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output[-3:] == [
        "Night: 20:00-08:00 (12 h)",
        "Night: 06:00-24:00 (18 h)",
        "Night: 00:00-18:00 (18 h)",
    ]
    for case, client_errors in zip(cases, output[:-3]):
        try:
            DemandProfile.from_dict(case)
            server_errors: dict[str, list[str]] = {}
        except ValidationError as exc:
            server_errors = exc.message_dict
        assert set(client_errors) == set(server_errors), (
            case,
            client_errors,
            server_errors,
        )
        for key, message in client_errors.items():
            assert server_errors[key] == [message]
