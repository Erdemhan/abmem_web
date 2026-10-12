"""Common learning start in the wizard, scenario YAML and analysis (PLAN 2.14).

``learning_starts`` is an optional scenario setting. Empty or null means the
automatic value (stored as NULL), otherwise it is a positive integer. The
wizard, the YAML import and the scenario copies carry it, and the analysis
settings show the value recorded in the parameter snapshot.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from abmem import scenario_yaml
from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent import library_service
from tests.analysis.test_run_analysis import hand_run  # noqa: F401  (fixture)
from tests.factories import make_resource
from tests.views.test_scenario_yaml import scenario  # noqa: F401  (fixture)

pytestmark = pytest.mark.django_db

ERROR_TEXT = "Learning start must be empty (automatic) or a positive integer."


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="ls-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _payload(resource: Resource, **settings: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "simulation_name": "ls-sim",
        "simulationMode": "1",
        "periodNumber": "10",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": [
            {
                "mode": "new",
                "name": "LsAgent",
                "type": AgentType.FOSSIL.value,
                "budget": "250",
                "algorithm_type": "TD3",
                "failstack": {"enabled": False},
                "plants": [{"resource": resource.id, "capacity": "200"}],
            }
        ],
    }
    payload.update(settings)
    return payload


def _post(client: Client, url: str, payload: dict[str, Any]) -> Any:
    return client.post(url, json.dumps(payload), content_type="application/json")


def _created(client: Client, payload: dict[str, Any]) -> Simulation:
    response = _post(client, reverse("create_simulation"), payload)
    assert response.status_code == 200, response.content
    return Simulation.objects.get(id=response.json()["simulation_id"])


def _update(client: Client, simulation: Simulation, **settings: Any) -> Any:
    payload = _payload(Resource.objects.get(name="ls-gas"), **settings)
    payload["agents"] = []
    return _post(client, reverse("update_simulation", args=[simulation.id]), payload)


def _wizard_initial(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="wizard-config" type="application/json">(.*?)</script>',
        html,
        flags=re.S,
    )
    assert match
    return json.loads(match.group(1))["initial"]


# wizard: create and update


@pytest.mark.parametrize(
    ("raw_value", "stored"),
    [("100", 100), (1, 1), ("", None), (None, None), ("  ", None)],
)
def test_create_stores_learning_starts(
    client: Client, fossil: Resource, raw_value: Any, stored: int | None
) -> None:
    simulation = _created(client, _payload(fossil, learning_starts=raw_value))

    assert simulation.learning_starts == stored


def test_create_without_the_key_is_automatic(client: Client, fossil: Resource) -> None:
    assert _created(client, _payload(fossil)).learning_starts is None


@pytest.mark.parametrize("raw_value", ["0", 0, "-3", "abc", 1.5, True, [64]])
def test_create_rejects_invalid_learning_starts(
    client: Client, fossil: Resource, raw_value: Any
) -> None:
    response = _post(
        client,
        reverse("create_simulation"),
        _payload(fossil, learning_starts=raw_value),
    )

    assert response.status_code == 400
    assert response.json()["errors"] == {"learning_starts": [ERROR_TEXT]}
    assert not Simulation.objects.exists()
    assert not Agent.objects.exists()


def test_update_sets_clears_and_keeps_learning_starts(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil, learning_starts="80"))

    assert _update(client, simulation).status_code == 200
    simulation.refresh_from_db()
    assert simulation.learning_starts == 80

    assert _update(client, simulation, learning_starts=200).status_code == 200
    simulation.refresh_from_db()
    assert simulation.learning_starts == 200

    assert _update(client, simulation, learning_starts="").status_code == 200
    simulation.refresh_from_db()
    assert simulation.learning_starts is None


def test_update_rejection_keeps_the_stored_value(
    client: Client, fossil: Resource
) -> None:
    simulation = _created(client, _payload(fossil, learning_starts=80))

    response = _update(client, simulation, learning_starts="0")

    assert response.status_code == 400
    assert list(response.json()["errors"]) == ["learning_starts"]
    simulation.refresh_from_db()
    assert simulation.learning_starts == 80


def test_create_page_renders_an_empty_automatic_field(client: Client) -> None:
    html = client.get(reverse("create_simulation")).content.decode()

    assert re.search(r'id="learning_starts"[^>]*value=""', html, re.S)
    assert "Learning start (transitions)" in html
    assert "64 with default settings" in html
    assert _wizard_initial(html)["learning_starts"] is None


def test_update_page_shows_the_stored_value(client: Client, fossil: Resource) -> None:
    simulation = _created(client, _payload(fossil, learning_starts=150))

    html = client.get(
        reverse("update_simulation", args=[simulation.id])
    ).content.decode()

    assert re.search(r'id="learning_starts"[^>]*value="150"', html, re.S)
    assert _wizard_initial(html)["learning_starts"] == 150


# scenario copies


def test_duplicate_and_clone_copy_learning_starts(
    client: Client, fossil: Resource
) -> None:
    scenario_sim = _created(client, _payload(fossil, learning_starts=120))

    duplicate = library_service.duplicate_scenario(scenario_sim)
    run = library_service.clone_scenario_for_run(scenario_sim)

    assert duplicate.proxy and duplicate.learning_starts == 120
    assert not run.proxy and run.learning_starts == 120


def test_copies_of_an_automatic_scenario_stay_automatic(
    client: Client, fossil: Resource
) -> None:
    scenario_sim = _created(client, _payload(fossil))

    assert library_service.duplicate_scenario(scenario_sim).learning_starts is None
    assert library_service.clone_scenario_for_run(scenario_sim).learning_starts is None


# scenario YAML


def _export(client: Client, simulation: Simulation) -> dict[str, Any]:
    response = client.get(reverse("export_scenario_yaml", args=[simulation.id]))
    assert response.status_code == 200
    return yaml.safe_load(response.content.decode("utf-8"))


def _confirm(client: Client, document: dict[str, Any]) -> Any:
    text = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    upload = SimpleUploadedFile("s.yaml", text, content_type="application/x-yaml")
    return client.post(reverse("import_scenario_confirm"), {"file": upload})


def _preview(client: Client, document: dict[str, Any]) -> Any:
    text = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    upload = SimpleUploadedFile("s.yaml", text, content_type="application/x-yaml")
    return client.post(reverse("import_scenario_preview"), {"file": upload})


def test_yaml_export_omits_an_automatic_learning_start(
    client: Client, scenario: Simulation  # noqa: F811
) -> None:
    assert "learning_starts" not in _export(client, scenario)["scenario"]


def test_yaml_round_trip_with_learning_starts(
    client: Client, scenario: Simulation  # noqa: F811
) -> None:
    scenario.learning_starts = 96
    scenario.save()
    document = _export(client, scenario)
    assert document["scenario"]["learning_starts"] == 96
    keys = list(document["scenario"])
    assert keys.index("learning_starts") == keys.index("evaluation_periods") + 1

    preview = _preview(client, document)
    assert preview.status_code == 200, preview.content
    assert preview.json()["scenario"]["learning_starts"] == 96
    response = _confirm(client, document)

    assert response.status_code == 200, response.content
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.learning_starts == 96
    assert scenario_yaml.scenario_document(imported)["scenario"] == document["scenario"]


@pytest.mark.parametrize("mutate", ["missing", "null"])
def test_yaml_import_without_value_is_automatic(
    client: Client, scenario: Simulation, mutate: str  # noqa: F811
) -> None:
    document = _export(client, scenario)
    if mutate == "null":
        document["scenario"]["learning_starts"] = None

    response = _confirm(client, document)

    assert response.status_code == 200, response.content
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.learning_starts is None
    assert (
        "learning_starts" not in scenario_yaml.scenario_document(imported)["scenario"]
    )


@pytest.mark.parametrize("raw_value", [0, -1, "abc", 2.5, True])
def test_yaml_import_rejects_invalid_learning_starts(
    client: Client, scenario: Simulation, raw_value: Any  # noqa: F811
) -> None:
    document = _export(client, scenario)
    document["scenario"]["learning_starts"] = raw_value
    before = Simulation.objects.count()

    for send in (_preview, _confirm):
        response = send(client, document)
        assert response.status_code == 400, response.content
        assert response.json()["errors"]["scenario.learning_starts"] == [ERROR_TEXT]
    assert Simulation.objects.count() == before


# analysis


def _settings(client: Client, simulation: Simulation) -> dict[str, Any]:
    response = client.get(reverse("simulation_analysis_data", args=[simulation.pk]))
    assert response.status_code == 200
    return response.json()["settings"]


def _make_learning(simulation: Simulation, learning_starts: int) -> None:
    agent = simulation.market.agent_set.get(name="A")
    agent.algorithm_type = "TD3"
    agent.effective_params = {"learning_starts": learning_starts, "batch_size": 64}
    agent.save()


def test_analysis_shows_an_automatic_learning_start(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    _make_learning(hand_run, 64)

    settings = _settings(client, hand_run)

    assert settings["learning_starts"] == 64
    assert settings["learning_starts_automatic"] is True


def test_analysis_shows_a_learning_start_set_on_the_scenario(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    _make_learning(hand_run, 100)
    hand_run.learning_starts = 100
    hand_run.save()

    settings = _settings(client, hand_run)

    assert settings["learning_starts"] == 100
    assert settings["learning_starts_automatic"] is False


def test_analysis_of_a_baseline_run_has_no_learning_start(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    settings = _settings(client, hand_run)

    assert settings["learning_starts"] is None


def test_analysis_page_has_the_learning_start_row(
    client: Client, hand_run: Simulation  # noqa: F811
) -> None:
    html = client.get(
        reverse("simulation_analysis", args=[hand_run.pk])
    ).content.decode()

    assert "['Learning start', learningStartText(s)]" in html
    assert "set on the scenario" in html
