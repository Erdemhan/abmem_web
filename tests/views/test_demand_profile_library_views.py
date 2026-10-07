"""Demand profile library pages, YAML and wizard integration (PLAN 2.9)."""

from __future__ import annotations

import copy
import json
import pathlib
import re
import shutil
import subprocess
from typing import Any

import pytest
import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from abmem import demand_profile_yaml, scenario_yaml
from abmem.models import DemandProfileTemplate, Simulation
from abmem.models.enums import EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from abmem.services.market import demand_library
from abmem.views_demand import profile_row, segment_summary
from tests.factories import make_resource
from tests.views.test_scenario_wizard import _create, _new_agent, _payload, _post
from tests.views.wizard_helpers import wizard_config

pytestmark = pytest.mark.django_db

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
OTHER = dict(copy.deepcopy(PROFILE), seed=999)


@pytest.fixture
def fossil() -> Resource:
    return make_resource(name="dpl-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


@pytest.fixture
def template() -> DemandProfileTemplate:
    return demand_library.create_profile("Winter", PROFILE, "cold days")


def _json_post(client: Client, url: str, body: Any, **headers: str) -> Any:
    return client.post(
        url, json.dumps(body), content_type="application/json", **headers
    )


def _upload(text: str, name: str = "profile.yaml") -> SimpleUploadedFile:
    return SimpleUploadedFile(
        name, text.encode("utf-8"), content_type="application/x-yaml"
    )


def _yaml(document: dict[str, Any]) -> str:
    return yaml.safe_dump(document, sort_keys=False)


def _document(name: str = "Imported", demand: Any = None) -> dict[str, Any]:
    return {
        "format_version": 1,
        "kind": "demand_profile",
        "name": name,
        "description": "from a file",
        "demand": copy.deepcopy(PROFILE) if demand is None else demand,
    }


# list page, dashboard, sidebar ------------------------------------------------


def test_list_page_shows_profiles_and_usage(
    client: Client, template: DemandProfileTemplate, fossil: Resource
) -> None:
    _create(client, _payload([_new_agent(fossil)], demand_profile_source=template.id))
    run_source = library_service.clone_scenario_for_run(
        Simulation.objects.get(proxy=True), seed=1
    )
    assert run_source.demand_profile_source_id == template.id
    response = client.get(reverse("demand_profile_list"))
    assert response.status_code == 200
    html = response.content.decode()
    assert "Winter" in html and "cold days" in html
    assert reverse("demand_profile_edit", args=[template.id]) in html
    assert reverse("demand_profile_export", args=[template.id]) in html
    assert reverse("demand_profile_duplicate", args=[template.id]) in html
    assert reverse("demand_profile_delete", args=[template.id]) in html
    assert 'id="sidebar-demand-profiles-link"' in html
    rows = response.context["rows"]
    assert rows[0]["scenario_count"] == 1  # the run is not counted
    assert rows[0]["day_hours"] == "06:00-22:00 (16 h day)"


def test_empty_list(client: Client) -> None:
    html = client.get(reverse("demand_profile_list")).content.decode()
    assert "no demand profiles yet" in html


def test_dashboard_card(client: Client, template: DemandProfileTemplate) -> None:
    response = client.get(reverse("dashboard"))
    assert response.context["num_demand_profiles"] == 1
    html = response.content.decode()
    assert 'id="num-demand-profiles"' in html
    assert reverse("demand_profile_create") in html
    assert 'id="sidebar-demand-profiles-link"' in html


def test_segment_summary_and_row(template: DemandProfileTemplate) -> None:
    assert segment_summary(PROFILE["night"]) == "Uniform on [400, 650] MWh"
    assert (
        segment_summary(PROFILE["day"])
        == "Log-normal (mean 900, std 120) on [600, 1300] MWh"
    )
    assert segment_summary(None) == "-"
    row = profile_row(template)
    assert row["scenario_count"] == 0
    assert row["night"].startswith("Uniform")


# create / edit ------------------------------------------------------------------


def test_create_page_renders_the_shared_editor(client: Client) -> None:
    html = client.get(reverse("demand_profile_create")).content.decode()
    assert 'id="editor-component"' in html and 'data-generated-only="true"' in html
    assert 'id="profile-preview-chart"' in html
    assert "assets/js/demand-profile.js" in html
    assert '<option value="168" selected>' in html


def test_create_and_errors(client: Client) -> None:
    url = reverse("demand_profile_create")
    response = _json_post(client, url, {"name": " Base ", "profile": PROFILE})
    assert response.status_code == 201, response.content
    body = response.json()
    assert body["name"] == "Base" and body["profile"] == PROFILE
    assert body["edit_url"] == reverse("demand_profile_edit", args=[body["id"]])

    bad = copy.deepcopy(PROFILE)
    bad["day"]["std"] = 0
    response = _json_post(client, url, {"name": "base", "profile": bad})
    assert response.status_code == 400
    errors = response.json()["errors"]
    assert "name" in errors and "profile.day.std" in errors
    assert DemandProfileTemplate.objects.count() == 1

    response = client.post(url, "{", content_type="application/json")
    assert response.status_code == 400 and "body" in response.json()["errors"]


def test_edit_page_and_update(client: Client, template: DemandProfileTemplate) -> None:
    url = reverse("demand_profile_edit", args=[template.id])
    html = client.get(url).content.decode()
    assert 'value="Winter"' in html and "cold days" in html
    assert json.dumps(PROFILE["seed"]) in html
    response = _json_post(
        client, url, {"name": "Winter 2", "description": "", "profile": OTHER}
    )
    assert response.status_code == 200, response.content
    template.refresh_from_db()
    assert template.name == "Winter 2" and template.profile == OTHER
    assert client.get(reverse("demand_profile_edit", args=[9999])).status_code == 404


# duplicate / delete ---------------------------------------------------------------


def test_duplicate(client: Client, template: DemandProfileTemplate) -> None:
    url = reverse("demand_profile_duplicate", args=[template.id])
    response = client.post(url)
    copy_template = DemandProfileTemplate.objects.get(name="Winter (copy)")
    assert response.status_code == 302
    assert response.url == reverse("demand_profile_edit", args=[copy_template.id])
    response = client.post(url, HTTP_ACCEPT="application/json")
    assert response.status_code == 201 and response.json()["name"] == "Winter (copy 2)"
    assert client.get(url).status_code == 405


def test_delete_blocked_by_scenario_then_allowed(
    client: Client, template: DemandProfileTemplate, fossil: Resource
) -> None:
    scenario = _create(
        client, _payload([_new_agent(fossil)], demand_profile_source=template.id)
    )
    url = reverse("demand_profile_delete", args=[template.id])
    response = client.post(url, HTTP_ACCEPT="application/json")
    assert response.status_code == 400
    assert "wiz-sim" in response.json()["errors"]["profile"][0]
    assert DemandProfileTemplate.objects.filter(id=template.id).exists()

    scenario.delete()
    response = client.post(url, HTTP_ACCEPT="application/json")
    assert response.status_code == 200 and response.json() == {"status": "success"}
    assert not DemandProfileTemplate.objects.filter(id=template.id).exists()


def test_delete_redirects_for_a_form_post(
    client: Client, template: DemandProfileTemplate
) -> None:
    response = client.post(reverse("demand_profile_delete", args=[template.id]))
    assert response.status_code == 302
    assert response.url == reverse("demand_profile_list")


def test_detail_json(client: Client, template: DemandProfileTemplate) -> None:
    body = client.get(reverse("demand_profile_detail_json", args=[template.id])).json()
    assert body == {
        "id": template.id,
        "name": "Winter",
        "description": "cold days",
        "profile": PROFILE,
    }
    response = client.get(reverse("demand_profile_detail_json", args=[9999]))
    assert response.status_code == 404
    assert "demand_profile_source" in response.json()["errors"]


# YAML ---------------------------------------------------------------------------


def test_export_round_trip_reuses(
    client: Client, template: DemandProfileTemplate
) -> None:
    response = client.get(reverse("demand_profile_export", args=[template.id]))
    assert response.status_code == 200
    assert 'filename="demand_profile_winter.yaml"' in response["Content-Disposition"]
    text = response.content.decode()
    assert text.startswith("# ABMEM demand profile export")
    document = yaml.safe_load(text)
    assert document == {
        "format_version": 1,
        "kind": "demand_profile",
        "name": "Winter",
        "description": "cold days",
        "demand": PROFILE,
    }
    preview = client.post(
        reverse("demand_profile_import_preview"), {"file": _upload(text)}
    )
    assert preview.status_code == 200, preview.content
    assert preview.json()["status"] == "reuse"
    assert preview.json()["existing_id"] == template.id
    confirm = client.post(
        reverse("demand_profile_import_confirm"), {"file": _upload(text)}
    )
    assert confirm.status_code == 200, confirm.content
    assert confirm.json()["id"] == template.id and confirm.json()["status"] == "reuse"
    assert DemandProfileTemplate.objects.count() == 1
    assert client.get(reverse("demand_profile_export", args=[9999])).status_code == 404


def test_import_new_profile(client: Client) -> None:
    text = _yaml(_document())
    preview = client.post(
        reverse("demand_profile_import_preview"), {"file": _upload(text)}
    )
    assert preview.status_code == 200
    assert preview.json()["status"] == "new" and preview.json()["demand"] == PROFILE
    assert DemandProfileTemplate.objects.count() == 0
    confirm = client.post(
        reverse("demand_profile_import_confirm"), {"file": _upload(text)}
    )
    assert confirm.status_code == 200 and confirm.json()["status"] == "new"
    created = DemandProfileTemplate.objects.get()
    assert (created.name, created.description, created.profile) == (
        "Imported",
        "from a file",
        PROFILE,
    )


def test_import_conflict_writes_nothing(
    client: Client, template: DemandProfileTemplate
) -> None:
    text = _yaml(_document(name="WINTER", demand=OTHER))
    preview = client.post(
        reverse("demand_profile_import_preview"), {"file": _upload(text)}
    )
    assert preview.status_code == 400
    assert preview.json()["status"] == "conflict" and "name" in preview.json()["errors"]
    confirm = client.post(
        reverse("demand_profile_import_confirm"), {"file": _upload(text)}
    )
    assert confirm.status_code == 400 and "name" in confirm.json()["errors"]
    template.refresh_from_db()
    assert template.profile == PROFILE
    assert DemandProfileTemplate.objects.count() == 1


@pytest.mark.parametrize(
    "text, key",
    [
        ("format_version: [1\n", "file"),
        ("- 1\n- 2\n", "file"),
        (_yaml(dict(_document(), format_version=2)), "format_version"),
        (_yaml(dict(_document(), kind="scenario")), "kind"),
        (_yaml({k: v for k, v in _document().items() if k != "demand"}), "file"),
        (_yaml(dict(_document(), extra=1)), "file"),
        (_yaml(dict(_document(), name="  ")), "name"),
        (_yaml(dict(_document(), description=5)), "description"),
        (
            _yaml(dict(_document(), demand=dict(PROFILE, autocorrelation=1))),
            "demand.autocorrelation",
        ),
        (_yaml(dict(_document(), demand=[1])), "demand"),
    ],
)
def test_invalid_files(client: Client, text: str, key: str) -> None:
    for url in ("demand_profile_import_preview", "demand_profile_import_confirm"):
        response = client.post(reverse(url), {"file": _upload(text)})
        assert response.status_code == 400, response.content
        assert key in response.json()["errors"], response.json()
    assert DemandProfileTemplate.objects.count() == 0


def test_import_without_file_and_too_large(client: Client) -> None:
    response = client.post(reverse("demand_profile_import_preview"))
    assert response.status_code == 400 and "file" in response.json()["errors"]
    big = "#" * (demand_profile_yaml.MAX_FILE_BYTES + 10)
    response = client.post(
        reverse("demand_profile_import_preview"), {"file": _upload(big)}
    )
    assert response.status_code == 400 and "file" in response.json()["errors"]


# wizard ---------------------------------------------------------------------------


def test_wizard_save_from_library_stores_copy_and_link(
    client: Client, template: DemandProfileTemplate, fossil: Resource
) -> None:
    # The library profile is authoritative, a stale posted profile is ignored.
    scenario = _create(
        client,
        _payload(
            [_new_agent(fossil)],
            demand_profile=OTHER,
            demand_profile_source=template.id,
        ),
    )
    assert scenario.demand_profile == PROFILE
    assert scenario.demand_profile_source_id == template.id

    url = reverse("update_simulation", args=[scenario.id])
    config = wizard_config(client.get(url).content.decode())
    assert config["initial"]["demand_profile_source"] == template.id
    assert {"id": template.id, "name": "Winter"} in config["demand_library"]
    assert config["demand_profile_detail_url"] == reverse(
        "demand_profile_detail_json", args=[0]
    )

    # Editing the library profile keeps the scenario copy, and the wizard then
    # shows the copy as a generated profile.
    demand_library.update_profile(template, "Winter", OTHER)
    scenario.refresh_from_db()
    assert scenario.demand_profile == PROFILE
    config = wizard_config(client.get(url).content.decode())
    assert config["initial"]["demand_profile_source"] is None
    assert config["initial"]["demand_profile"] == PROFILE


def _scenario_agents(simulation: Simulation, fossil: Resource) -> list[dict[str, Any]]:
    return [
        {**_new_agent(fossil, name=a.name), "mode": "scenario", "agent_id": a.id}
        for a in simulation.market.agent_set.all()
    ]


def test_wizard_file_and_generate_clear_the_link(
    client: Client, template: DemandProfileTemplate, fossil: Resource
) -> None:
    scenario = _create(
        client, _payload([_new_agent(fossil)], demand_profile_source=template.id)
    )
    url = reverse("update_simulation", args=[scenario.id])
    agents = _scenario_agents(scenario, fossil)

    # Neither key keeps the stored profile and link.
    assert _post(client, url, _payload(agents)).status_code == 200
    scenario.refresh_from_db()
    assert scenario.demand_profile_source_id == template.id

    # Generate: the posted profile is stored, the link is cleared.
    response = _post(
        client, url, _payload(agents, demand_profile=OTHER, demand_profile_source=None)
    )
    assert response.status_code == 200, response.content
    scenario.refresh_from_db()
    assert scenario.demand_profile == OTHER and scenario.demand_profile_source is None

    # From library again, then File clears both.
    _post(client, url, _payload(agents, demand_profile_source=template.id))
    scenario.refresh_from_db()
    assert scenario.demand_profile_source_id == template.id
    response = _post(
        client, url, _payload(agents, demand_profile=None, demand_profile_source=None)
    )
    assert response.status_code == 200
    scenario.refresh_from_db()
    assert scenario.demand_profile is None and scenario.demand_profile_source is None


def test_wizard_rejects_an_unknown_library_profile(
    client: Client, fossil: Resource
) -> None:
    before = Simulation.objects.count()
    for source in (9999, "abc"):
        response = _post(
            client,
            reverse("create_simulation"),
            _payload([_new_agent(fossil)], demand_profile_source=source),
        )
        assert response.status_code == 400
        assert "demand_profile_source" in response.json()["errors"]
    assert Simulation.objects.count() == before


def test_wizard_page_offers_the_library(
    client: Client, template: DemandProfileTemplate
) -> None:
    html = client.get(reverse("create_simulation")).content.decode()
    assert 'id="demand-source-library"' in html and 'id="demand-library-select"' in html
    assert (
        'id="demand-yaml-import"' in html and 'id="demand-save-library-toggle"' in html
    )
    config = wizard_config(html)
    assert config["initial"]["demand_profile_source"] is None
    assert config["demand_yaml_parse_url"] == reverse("demand_profile_parse_yaml")
    assert config["demand_save_to_library_url"] == reverse("demand_profile_create")


def test_wizard_demand_yaml_parse_writes_nothing(client: Client) -> None:
    url = reverse("demand_profile_parse_yaml")
    raw = copy.deepcopy(PROFILE)
    del raw["version"]
    response = client.post(url, {"file": _upload(_yaml(_document(demand=raw)))})
    assert response.status_code == 200, response.content
    assert response.json() == {
        "name": "Imported",
        "description": "from a file",
        "profile": PROFILE,
    }
    bad = _document(demand=dict(PROFILE, day_end_hour=2))
    response = client.post(url, {"file": _upload(_yaml(bad))})
    assert response.status_code == 400
    assert "demand.day_end_hour" in response.json()["errors"]
    assert DemandProfileTemplate.objects.count() == 0


def test_wizard_save_to_library_endpoint(
    client: Client, template: DemandProfileTemplate
) -> None:
    url = reverse("demand_profile_create")
    response = _json_post(
        client, url, {"name": "winter", "description": "", "profile": OTHER}
    )
    assert response.status_code == 400 and "name" in response.json()["errors"]
    response = _json_post(
        client, url, {"name": "Scenario demand", "description": "x", "profile": OTHER}
    )
    assert response.status_code == 201
    assert response.json()["profile"] == OTHER


# scenario YAML ----------------------------------------------------------------------


def test_scenario_yaml_carries_the_library_name(
    client: Client, template: DemandProfileTemplate, fossil: Resource
) -> None:
    scenario = _create(
        client, _payload([_new_agent(fossil)], demand_profile_source=template.id)
    )
    text = scenario_yaml.export_scenario_yaml(scenario)
    document = yaml.safe_load(text)
    assert document["scenario"]["demand_profile"] == "Winter"
    assert document["scenario"]["demand"] == PROFILE

    upload = SimpleUploadedFile("s.yaml", text.encode("utf-8"))
    preview = client.post(reverse("import_scenario_preview"), {"file": upload})
    assert preview.status_code == 200, preview.content
    assert preview.json()["scenario"]["demand_profile"] == "Winter"
    upload = SimpleUploadedFile("s.yaml", text.encode("utf-8"))
    response = client.post(reverse("import_scenario_confirm"), {"file": upload})
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.demand_profile == PROFILE
    assert imported.demand_profile_source_id == template.id

    # A name with different settings or an unknown name is only informational.
    for name in ("Winter", "Unknown"):
        document["scenario"]["demand_profile"] = name
        document["scenario"]["demand"] = OTHER
        upload = SimpleUploadedFile("s.yaml", _yaml(document).encode("utf-8"))
        response = client.post(reverse("import_scenario_confirm"), {"file": upload})
        assert response.status_code == 200, response.content
        imported = Simulation.objects.get(id=response.json()["simulation_id"])
        assert imported.demand_profile == OTHER
        assert imported.demand_profile_source is None

    document["scenario"]["demand_profile"] = 5
    upload = SimpleUploadedFile("s.yaml", _yaml(document).encode("utf-8"))
    response = client.post(reverse("import_scenario_preview"), {"file": upload})
    assert response.status_code == 400
    assert "scenario.demand_profile" in response.json()["errors"]


# page scripts -----------------------------------------------------------------------

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize(
    "url_name, with_id",
    [
        ("demand_profile_list", False),
        ("demand_profile_create", False),
        ("demand_profile_edit", True),
        ("create_simulation", False),
    ],
)
def test_inline_page_scripts_pass_node_check(
    client: Client,
    template: DemandProfileTemplate,
    tmp_path: pathlib.Path,
    url_name: str,
    with_id: bool,
) -> None:
    args = [template.id] if with_id else []
    html = client.get(reverse(url_name, args=args)).content.decode()
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.S)
    assert scripts
    for index, script in enumerate(scripts):
        path = tmp_path / f"{url_name}_{index}.js"
        path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [NODE, "--check", str(path)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
