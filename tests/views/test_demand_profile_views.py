"""Demand profile in the wizard, the preview endpoint, YAML and the JS component (PLAN 2.8)."""

from __future__ import annotations

import copy
import json
import pathlib
import re
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


# ramps and summer/winter split ---------------------------------------------------

SEASONAL: dict[str, Any] = {
    **copy.deepcopy(PROFILE),
    "morning_ramp_hours": 2,
    "evening_ramp_hours": 3,
    "seasons": {
        "summer_end_period": 48,
        "transition_days": 1,
        "winter": {
            "day_start_hour": 8,
            "day_end_hour": 18,
            "morning_ramp_hours": 1,
            "evening_ramp_hours": 0,
            "day": {
                "distribution": "NORMAL",
                "mean": 1100.0,
                "std": 100.0,
                "min": 800.0,
                "max": 1500.0,
            },
            "night": {"distribution": "UNIFORM", "min": 500.0, "max": 800.0},
        },
    },
}


def _seasonal(**changes: Any) -> dict[str, Any]:
    """Return SEASONAL with ``changes`` applied ("a.b.c" keys address nested fields)."""
    profile = copy.deepcopy(SEASONAL)
    for path, value in changes.items():
        target = profile
        *parents, last = path.split(".")
        for name in parents:
            target = target[name]
        target[last] = value
    return profile


def test_seasonal_profile_is_stored_unchanged() -> None:
    assert DemandProfile.from_dict(SEASONAL).to_dict() == SEASONAL


def test_seasonal_preview_has_boundary_and_season_stats(client: Client) -> None:
    response = _preview(client, {"periods": 96, "profile": SEASONAL})
    assert response.status_code == 200, response.content
    body = response.json()
    profile = DemandProfile.from_dict(SEASONAL)
    expected = generate_demand(profile, 96)
    assert body["values"] == [float(v) for v in expected]
    assert body["night_mask"] == [bool(flag) for flag in profile.night_mask(96)]
    assert body["season_boundary"] == 48
    winter = profile.seasons.winter_weight(96) >= 0.5
    assert int(winter.sum()) == 48 and not winter[:48].any()
    stats = body["stats"]
    assert stats["summer"]["count"] == 48 and stats["winter"]["count"] == 48
    assert stats["summer"]["max"] == pytest.approx(float(expected[:48].max()))
    assert stats["winter"]["mean"] == pytest.approx(float(expected[48:].mean()))
    # Hour 6 is a summer day hour (06:00-22:00) but a winter night hour (18:00-08:00).
    assert body["night_mask"][6] is False and body["night_mask"][54] is True


def test_seasonal_preview_within_summer_has_an_empty_winter(client: Client) -> None:
    body = _preview(client, {"periods": 24, "profile": SEASONAL}).json()
    assert body["season_boundary"] == 48
    assert body["stats"]["winter"] is None and body["stats"]["summer"]["count"] == 24


def test_preview_without_seasons_has_no_season_keys(client: Client) -> None:
    body = _preview(client, {"periods": 24, "profile": PROFILE}).json()
    assert "season_boundary" not in body
    assert "summer" not in body["stats"] and "winter" not in body["stats"]


def test_preview_with_ramps_matches_the_generator(client: Client) -> None:
    profile = dict(copy.deepcopy(PROFILE), morning_ramp_hours=4, evening_ramp_hours=4)
    body = _preview(client, {"periods": 48, "profile": profile}).json()
    expected = generate_demand(DemandProfile.from_dict(profile), 48)
    assert body["values"] == [float(v) for v in expected]
    assert body["night_mask"] == [not (6 <= t % 24 < 22) for t in range(48)]


@pytest.mark.parametrize(
    "profile, key",
    [
        (_seasonal(**{"seasons.winter.night.max": 100.0}), "seasons.winter.night.max"),
        (
            _seasonal(**{"seasons.winter.day_end_hour": 8}),
            "seasons.winter.day_end_hour",
        ),
        (_seasonal(**{"seasons.transition_days": 5}), "seasons.transition_days"),
        (_seasonal(**{"seasons.summer_end_period": 0}), "seasons.summer_end_period"),
        (_seasonal(**{"seasons": []}), "seasons"),
        (_seasonal(evening_ramp_hours=15), "evening_ramp_hours"),
        (
            _seasonal(
                **{
                    "seasons.winter.morning_ramp_hours": 12,
                    "seasons.winter.evening_ramp_hours": 12,
                }
            ),
            "seasons.winter.evening_ramp_hours",
        ),
    ],
)
def test_preview_rejects_bad_seasonal_profiles(
    client: Client, profile: dict[str, Any], key: str
) -> None:
    response = _preview(client, {"periods": 24, "profile": profile})
    assert response.status_code == 400
    assert f"demand_profile.{key}" in response.json()["errors"]


def test_wizard_round_trip_of_a_seasonal_profile(
    client: Client, fossil: Resource
) -> None:
    simulation = _create(
        client, _payload([_new_agent(fossil)], demand_profile=SEASONAL)
    )
    assert simulation.demand_profile == SEASONAL
    url = reverse("update_simulation", args=[simulation.id])
    html = client.get(url).content.decode()
    assert wizard_config(html)["initial"]["demand_profile"] == SEASONAL
    for element in (
        "demand-seasons_enabled",
        "demand-summer_end_period",
        "demand-transition_days",
        "demand-season-text",
        "demand-season-warning",
        "demand-morning_ramp_hours",
        "demand-evening_ramp_hours",
        "demand-winter-day_start_hour",
        "demand-winter-evening_ramp_hours",
        "demand-winter-night-std",
        "demand-winter-feedback",
        "demand-seasons-feedback",
    ):
        assert f'id="{element}"' in html, element

    agents = [
        {"mode": "scenario", "agent_id": a.id, **_new_agent(fossil, name=a.name)}
        for a in simulation.market.agent_set.all()
    ]
    for agent in agents:
        agent["mode"] = "scenario"
    edited = _seasonal(
        morning_ramp_hours=0,
        **{"seasons.summer_end_period": 100, "seasons.transition_days": 0},
    )
    response = _post(client, url, _payload(agents, demand_profile=edited))
    assert response.status_code == 200, response.content
    simulation.refresh_from_db()
    assert simulation.demand_profile == edited
    run = library_service.clone_scenario_for_run(simulation, seed=3)
    assert run.demand_profile == edited


def test_wizard_rejects_bad_winter_settings(client: Client, fossil: Resource) -> None:
    bad = _seasonal(
        **{
            "seasons.winter.night.std": 0,
            "seasons.winter.night.mean": 600.0,
            "seasons.winter.night.distribution": "NORMAL",
            "seasons.transition_days": 5,
        }
    )
    before = Simulation.objects.count()
    response = _post(
        client,
        reverse("create_simulation"),
        _payload([_new_agent(fossil)], demand_profile=bad),
    )
    assert response.status_code == 400
    errors = response.json()["errors"]
    assert "demand_profile.seasons.winter.night.std" in errors
    assert "demand_profile.seasons.transition_days" in errors
    assert Simulation.objects.count() == before


def test_profile_without_new_keys_reads_as_off() -> None:
    old = copy.deepcopy(PROFILE)
    for key in ("morning_ramp_hours", "evening_ramp_hours", "seasons"):
        del old[key]
    assert DemandProfile.from_dict(old).to_dict() == PROFILE


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_client_validation_mirrors_the_server_for_seasons(
    tmp_path: pathlib.Path,
) -> None:
    from django.core.exceptions import ValidationError

    winter = SEASONAL["seasons"]["winter"]
    cases: list[dict[str, Any]] = [
        copy.deepcopy(SEASONAL),
        _seasonal(morning_ramp_hours=13),
        _seasonal(morning_ramp_hours=1.5),
        _seasonal(morning_ramp_hours=None, evening_ramp_hours=""),
        _seasonal(morning_ramp_hours=10, evening_ramp_hours=7),
        _seasonal(**{"seasons.summer_end_period": 0}),
        _seasonal(**{"seasons.summer_end_period": None}),
        _seasonal(**{"seasons.summer_end_period": 2_000_000}),
        _seasonal(**{"seasons.transition_days": 91}),
        _seasonal(**{"seasons.transition_days": 5}),
        _seasonal(**{"seasons.transition_days": "x"}),
        _seasonal(**{"seasons.winter.day_end_hour": 8}),
        _seasonal(**{"seasons.winter.day_start_hour": "x"}),
        _seasonal(**{"seasons.winter.evening_ramp_hours": 13}),
        _seasonal(
            **{
                "seasons.winter.morning_ramp_hours": 12,
                "seasons.winter.evening_ramp_hours": 12,
            }
        ),
        _seasonal(**{"seasons.winter.night.max": 100.0}),
        _seasonal(**{"seasons.winter.day.std": 0.0}),
        _seasonal(**{"seasons.winter.day.mean": None}),
        _seasonal(**{"seasons.winter.day.distribution": "GAMMA"}),
        _seasonal(**{"seasons.winter": None}),
        _seasonal(seasons=[]),
        _seasonal(seasons={"summer_end_period": 48, "winter": winter}),
    ]
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
    assert output[0] == {}
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
            assert server_errors[key] == [message], (key, message, server_errors[key])


# Editor behaviour in node: a minimal DOM built from the rendered component markup.

_DOM_HARNESS = r"""
const fs = require('fs');
const nodes = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));

class ClassList {
    constructor(names) { this.set = new Set(names); }
    add(name) { this.set.add(name); }
    remove(name) { this.set.delete(name); }
    contains(name) { return this.set.has(name); }
    toggle(name, force) {
        const on = force === undefined ? !this.set.has(name) : Boolean(force);
        if (on) { this.set.add(name); } else { this.set.delete(name); }
        return on;
    }
}

class Element {
    constructor(tag, attrs) {
        this.tagName = tag.toUpperCase();
        this.attrs = Object.assign({}, attrs || {});
        this.id = this.attrs.id || '';
        this.classList = new ClassList((this.attrs.class || '').split(/\s+/).filter(Boolean));
        this.hidden = 'hidden' in this.attrs;
        this.disabled = 'disabled' in this.attrs;
        this.checked = 'checked' in this.attrs;
        this.children = [];
        this.parent = null;
        this.listeners = {};
        this._value = this.attrs.value !== undefined ? this.attrs.value : '';
        this.textContent = this.attrs.__text || '';
    }
    get name() { return this.attrs.name || ''; }
    get options() { return this.children.filter(child => child.tagName === 'OPTION'); }
    get value() {
        if (this.tagName !== 'SELECT') { return this._value; }
        if (this._selected === undefined) { this._selected = this.options.length ? this.options[0].value : ''; }
        return this._selected;
    }
    set value(text) {
        text = String(text);
        if (this.tagName !== 'SELECT') { this._value = text; return; }
        this._selected = this.options.some(option => option.value === text) ? text : '';
    }
    getAttribute(name) { return this.attrs[name] === undefined ? null : this.attrs[name]; }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    appendChild(child) { child.parent = this; this.children.push(child); return child; }
    remove(index) { const option = this.options[index]; this.children.splice(this.children.indexOf(option), 1); }
    addEventListener(type, callback) { (this.listeners[type] = this.listeners[type] || []).push(callback); }
    dispatch(type) {
        const event = { type: type, target: this };
        for (let node = this; node; node = node.parent) {
            (node.listeners[type] || []).forEach(callback => callback(event));
        }
    }
    descendants() {
        const out = [];
        const walk = node => node.children.forEach(child => { out.push(child); walk(child); });
        walk(this);
        return out;
    }
    matches(selector) {
        return selector.split(',').some(part => {
            const match = /^\s*([a-z]*)(?:\[name="([^"]*)"\])?(?:\.([\w-]+))?(:checked)?\s*$/.exec(part);
            if (!match) { throw new Error('unsupported selector ' + part); }
            if (match[1] && this.tagName !== match[1].toUpperCase()) { return false; }
            if (match[2] !== undefined && this.name !== match[2]) { return false; }
            if (match[3] && !this.classList.contains(match[3])) { return false; }
            if (match[4] && !this.checked) { return false; }
            return true;
        });
    }
    querySelectorAll(selector) { return this.descendants().filter(node => node.matches(selector)); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

const root = new Element('div', {});
const all = [];
nodes.forEach(spec => {
    const element = new Element(spec.tag, spec.attrs);
    all.push(element);
    (spec.parent === null ? root : all[spec.parent]).appendChild(element);
});
global.document = {
    getElementById: id => all.find(node => node.id === id) || null,
    createElement: tag => new Element(tag, {}),
};
global.window = {};
eval(fs.readFileSync(process.argv[2], 'utf8'));
const api = window.DemandProfileForm;
const scenario = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const byId = id => document.getElementById(id);
const out = {};

const form = api.init(byId('demand-component'), { library: [{ id: 7, name: 'Lib' }] });
out.initialSource = form.getSource();
out.initialProfile = form.getProfile();

form.setHorizon(100, 'run');
form.setProfile(scenario.seasonal, null);
out.seasonalSource = form.getSource();
out.seasonalProfile = form.getProfile();
out.shapeTitle = byId('demand-shape-title').textContent;
out.winterHidden = byId('demand-winter').hidden;
out.seasonFieldsHidden = byId('demand-season-fields').hidden;
out.seasonText = byId('demand-season-text').textContent;
out.warningHidden = byId('demand-season-warning').hidden;
out.winterNightText = byId('demand-winter-night-text').textContent;
out.previewBody = form.previewRequest(100);
form.setHorizon(40, 'run');
out.wholeRunText = byId('demand-season-text').textContent;
out.wholeRunWarning = byId('demand-season-warning').hidden ? null : byId('demand-season-warning').textContent;

out.shown = form.showErrors(scenario.errors);
out.feedback = {};
scenario.feedbackIds.forEach(id => { out.feedback[id] = byId(id).textContent; });
out.invalid = scenario.invalidIds.filter(id => byId(id).classList.contains('is-invalid'));
out.sourceFeedback = byId('demand-source-feedback').textContent;

form.setProfile(scenario.old, null);
out.oldProfile = form.getProfile();
out.oldTitle = byId('demand-shape-title').textContent;
out.oldWinterHidden = byId('demand-winter').hidden;

const toggle = byId('demand-seasons_enabled');
toggle.checked = true;
toggle.dispatch('change');
out.switchedProfile = form.getProfile();
out.switchedTitle = byId('demand-shape-title').textContent;
toggle.checked = false;
toggle.dispatch('change');
out.switchedOffSeasons = form.getProfile().seasons;

form.setProfile(scenario.seasonal, 7);
out.librarySource = form.getSource();
out.libraryId = form.getLibraryId();
out.locked = scenario.lockedIds.filter(id => byId(id).disabled);
byId('demand-library-edit').dispatch('click');
out.afterEditCopySource = form.getSource();
out.unlocked = scenario.lockedIds.filter(id => !byId(id).disabled);

form.setProfile(null);
out.fileProfile = form.getProfile();
process.stdout.write(JSON.stringify(out));
"""


def _component_nodes(html: str) -> list[dict[str, Any]]:
    """Parse rendered markup into ``[{tag, attrs, parent}]`` for the DOM harness.

    Args:
        html: Rendered component markup.

    Returns:
        One entry per element in document order. ``parent`` is the index of
        the parent element or None. Script and option text is kept in the
        attribute ``__text``.
    """
    from html.parser import HTMLParser

    void = {"input", "br", "hr", "img", "meta", "link"}
    nodes: list[dict[str, Any]] = []
    stack: list[int] = []

    class Parser(HTMLParser):
        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            nodes.append(
                {
                    "tag": tag,
                    "attrs": {k: "" if v is None else v for k, v in attrs},
                    "parent": stack[-1] if stack else None,
                }
            )
            if tag not in void:
                stack.append(len(nodes) - 1)

        def handle_startendtag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            self.handle_starttag(tag, attrs)
            if tag not in void:
                stack.pop()

        def handle_endtag(self, tag: str) -> None:
            if tag not in void and stack:
                stack.pop()

        def handle_data(self, data: str) -> None:
            if stack and nodes[stack[-1]]["tag"] in {"script", "option"}:
                nodes[stack[-1]]["attrs"]["__text"] = data

    Parser().feed(html)
    for node in nodes:
        if node["tag"] == "option" and "value" not in node["attrs"]:
            node["attrs"]["value"] = node["attrs"].get("__text", "").strip()
    return nodes


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_editor_reads_writes_and_locks_the_new_fields(tmp_path: pathlib.Path) -> None:
    from django.template.loader import render_to_string

    html = render_to_string(
        "components/demand_profile_form.html",
        {"prefix": "demand", "initial_profile": None, "show_library": True},
    )
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "duplicate ids in the editor markup"
    nodes_path = tmp_path / "nodes.json"
    nodes_path.write_text(json.dumps(_component_nodes(html)), encoding="utf-8")
    old = copy.deepcopy(PROFILE)
    for key in ("morning_ramp_hours", "evening_ramp_hours", "seasons"):
        del old[key]
    errors = {
        "demand_profile.seasons.winter.night.std": ["winter night std"],
        "demand_profile.seasons.transition_days": ["transition"],
        "demand_profile.seasons.summer_end_period": ["summer end"],
        "demand_profile.seasons": ["seasons object"],
        "demand_profile.seasons.winter": ["winter object"],
        "demand_profile.seasons.winter.evening_ramp_hours": ["winter evening"],
        "demand_profile.morning_ramp_hours": ["morning"],
    }
    feedback_ids = [
        "demand-winter-night-std-feedback",
        "demand-transition_days-feedback",
        "demand-summer_end_period-feedback",
        "demand-seasons-feedback",
        "demand-winter-feedback",
        "demand-winter-evening_ramp_hours-feedback",
        "demand-morning_ramp_hours-feedback",
    ]
    invalid_ids = [
        "demand-winter-night-std",
        "demand-transition_days",
        "demand-summer_end_period",
        "demand-winter-evening_ramp_hours",
        "demand-morning_ramp_hours",
    ]
    locked_ids = [
        "demand-seasons_enabled",
        "demand-summer_end_period",
        "demand-transition_days",
        "demand-morning_ramp_hours",
        "demand-evening_ramp_hours",
        "demand-winter-day_start_hour",
        "demand-winter-morning_ramp_hours",
        "demand-winter-night-distribution",
        "demand-winter-day-mean",
    ]
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "seasonal": SEASONAL,
                "old": old,
                "errors": errors,
                "feedbackIds": feedback_ids,
                "invalidIds": invalid_ids,
                "lockedIds": locked_ids,
            }
        ),
        encoding="utf-8",
    )
    harness = tmp_path / "dom.js"
    harness.write_text(_DOM_HARNESS, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(harness), str(JS_PATH), str(nodes_path), str(scenario_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)

    assert out["initialSource"] == "file" and out["initialProfile"] is None
    assert out["seasonalSource"] == "generated"
    assert out["seasonalProfile"] == SEASONAL
    assert out["shapeTitle"] == "Summer"
    assert out["winterHidden"] is False and out["seasonFieldsHidden"] is False
    assert out["seasonText"] == "Summer: periods 1..48, Winter: periods 49..100"
    assert out["warningHidden"] is True
    assert out["winterNightText"] == "Night: 18:00-08:00 (14 h)"
    assert out["previewBody"] == {
        "periods": 100,
        "source": "generated",
        "profile": SEASONAL,
    }
    assert out["wholeRunText"] == "Summer: periods 1..40, Winter: none"
    assert out["wholeRunWarning"] == "The whole run is summer."

    assert out["shown"] == len(errors)
    assert out["feedback"] == dict(
        zip(feedback_ids, [messages[0] for messages in errors.values()])
    )
    assert out["invalid"] == invalid_ids
    assert out["sourceFeedback"] == ""

    assert out["oldProfile"] == PROFILE
    assert out["oldTitle"] == "Day/Night" and out["oldWinterHidden"] is True
    switched = out["switchedProfile"]
    assert out["switchedTitle"] == "Summer"
    assert switched["seasons"]["summer_end_period"] == 20  # half of the horizon 40
    assert switched["seasons"]["transition_days"] == 0
    shape_keys = (
        "day_start_hour",
        "day_end_hour",
        "morning_ramp_hours",
        "evening_ramp_hours",
        "day",
        "night",
    )
    assert switched["seasons"]["winter"] == {key: PROFILE[key] for key in shape_keys}
    assert out["switchedOffSeasons"] is None

    assert out["librarySource"] == "library" and out["libraryId"] == 7
    assert out["locked"] == locked_ids
    assert out["afterEditCopySource"] == "generated"
    assert out["unlocked"] == locked_ids
    assert out["fileProfile"] is None
