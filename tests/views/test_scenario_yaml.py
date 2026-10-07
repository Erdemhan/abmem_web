"""Scenario YAML export and import (PLAN 2.7)."""

from __future__ import annotations

import copy
import pathlib
import re
import shutil
import subprocess
from decimal import Decimal
from typing import Any

import pytest
import yaml
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.urls import reverse

from abmem import scenario_yaml
from abmem.models import Agent, Simulation
from abmem.models.enums import (
    AgentType,
    AlgorithmType,
    EnergyType,
    MarketStrategy,
    SimulationMode,
)
from abmem.models.market import Market
from abmem.models.models import Plant, Portfolio, Resource
from abmem.services.agent import library_service
from abmem.services.agent.library_service import AgentInput, PlantInput
from tests.factories import make_market, make_resource

pytestmark = pytest.mark.django_db

NODE = shutil.which("node")


def _library(
    name: str,
    agent_type: AgentType,
    plants: list[tuple[Resource, int]],
    **overrides: Any,
) -> Agent:
    values: dict[str, Any] = {
        "budget": Decimal("1500.50"),
        "algorithm_type": AlgorithmType.TD3,
        "failstack_enabled": True,
        "failstack_limit": 4,
        "failstack_delta": Decimal("12.50"),
        "failstack_min_margin": Decimal("0.0250"),
        "algorithm_params": {"lr_actor": 0.0005, "hidden_sizes": [64, 64]},
    }
    values.update(overrides)
    return library_service.create_library_agent(
        AgentInput(
            name=name,
            type=agent_type,
            plants=tuple(PlantInput(resource.id, capacity) for resource, capacity in plants),
            **values,
        )
    )


@pytest.fixture
def scenario() -> Simulation:
    """A two-agent proxy scenario built from library agents."""
    gas = make_resource(
        name="Gas",
        energy_type=EnergyType.FOSSIL,
        fuel_cost=Decimal("45.25"),
        om_cost=Decimal("3.10"),
        investment_cost=Decimal("7.00"),
        co2_cost=Decimal("1.05"),
        emission=Decimal("0.40"),
    )
    wind = make_resource(name="Wind", energy_type=EnergyType.RENEWABLE, om_cost=Decimal("2.50"))
    unused = make_resource(name="Coal", energy_type=EnergyType.FOSSIL)
    assert unused.id
    thermal = _library("Thermal", AgentType.FOSSIL, [(gas, 300), (gas, 150)])
    green = _library(
        "Green",
        AgentType.RENEWABLE,
        [(wind, 200)],
        algorithm_type=AlgorithmType.MARGINAL_COST,
        failstack_enabled=False,
        algorithm_params={},
        budget=Decimal("0"),
    )
    simulation = Simulation.objects.create(
        name="Export test",
        mode=SimulationMode.ONLYRESULT,
        periodType=0,
        periodNumber=168,
        currentPeriod=-1,
        day=0,
        proxy=True,
        seed=12345,
        evaluation_periods=24,
        observation_lags=[1, 24, 168],
    )
    market = Market.objects.create(
        simulation=simulation,
        strategy=MarketStrategy.PAYASBID,
        lowerBidBound=0,
        upperBidBound=250,
        proxy=True,
    )
    library_service.copy_agent(thermal, market, proxy=True, name="Thermal A")
    library_service.copy_agent(green, market, proxy=True)
    return simulation


def _export(client: Client, simulation: Simulation) -> str:
    response = client.get(reverse("export_scenario_yaml", args=[simulation.id]))
    assert response.status_code == 200
    return response.content.decode("utf-8")


def _upload(text: str | bytes, name: str = "scenario.yaml") -> SimpleUploadedFile:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return SimpleUploadedFile(name, data, content_type="application/x-yaml")


def _preview(client: Client, text: str | bytes) -> Any:
    return client.post(reverse("import_scenario_preview"), {"file": _upload(text)})


def _confirm(client: Client, text: str | bytes) -> Any:
    return client.post(reverse("import_scenario_confirm"), {"file": _upload(text)})


def _counts() -> tuple[int, int, int, int, int, int]:
    return (
        Simulation.objects.count(),
        Market.objects.count(),
        Agent.objects.count(),
        Resource.objects.count(),
        Portfolio.objects.count(),
        Plant.objects.count(),
    )


def _document(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)


def _dump(document: dict[str, Any]) -> str:
    return yaml.safe_dump(document, sort_keys=False)


def _normalized(document: dict[str, Any]) -> dict[str, Any]:
    """Return a document with comparable numbers (Decimal) for equality checks."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, float) or isinstance(value, Decimal):
            return Decimal(str(value))
        return value

    return convert(document)


def _assert_rejected(client: Client, text: str, key: str, fragment: str) -> dict[str, list[str]]:
    before = _counts()
    preview = _preview(client, text)
    assert preview.status_code == 400, preview.content
    errors = preview.json()["errors"]
    assert key in errors, errors
    assert any(fragment in message for message in errors[key]), errors
    confirm = _confirm(client, text)
    assert confirm.status_code == 400
    assert confirm.json()["errors"] == errors
    assert _counts() == before
    return errors


# export ----------------------------------------------------------------


def test_export_file_content_and_name(client: Client, scenario: Simulation) -> None:
    response = client.get(reverse("export_scenario_yaml", args=[scenario.id]))
    assert response.status_code == 200
    assert response["Content-Type"].startswith("application/x-yaml")
    assert response["Content-Disposition"] == 'attachment; filename="scenario_export-test.yaml"'
    text = response.content.decode("utf-8")
    assert text.startswith("# ABMEM scenario export\n# Exported: ")
    assert f"# Source scenario id: {scenario.id}\n" in text
    document = _document(text)
    assert list(document) == ["format_version", "scenario", "market", "resources", "agents"]
    assert document["format_version"] == 1
    assert document["scenario"] == {
        "name": "Export test",
        "mode": "ONLYRESULT",
        "period_type": "HOUR",
        "periods": 168,
        "seed": 12345,
        "evaluation_periods": 24,
        "observation_lags": [1, 24, 168],
    }
    assert document["market"] == {"strategy": "PAYASBID", "bid_bounds": [0, 250]}
    assert [r["name"] for r in document["resources"]] == ["Gas", "Wind"]
    gas = document["resources"][0]
    assert gas == {
        "name": "Gas",
        "energy_type": "FOSSIL",
        "fuel_cost": 45.25,
        "om_cost": 3.1,
        "investment_cost": 7.0,
        "co2_cost": 1.05,
        "emission": 0.4,
    }
    thermal, green = document["agents"]
    assert thermal == {
        "name": "Thermal A",
        "type": "FOSSIL",
        "budget": 1500.5,
        "algorithm": "TD3",
        "algorithm_params": {"lr_actor": 0.0005, "hidden_sizes": [64, 64]},
        "failstack": {"enabled": True, "limit": 4, "delta": 12.5, "min_margin": 0.025},
        "plants": [{"resource": "Gas", "capacity": 300}, {"resource": "Gas", "capacity": 150}],
    }
    assert green["name"] == "Green"
    assert green["algorithm"] == "MARGINAL_COST"
    assert green["algorithm_params"] == {}
    assert "bid_bounds: [0, 250]" in text
    assert "fuel_cost: 45.25" in text


def test_export_writes_imprecise_decimals_as_strings(client: Client, scenario: Simulation) -> None:
    agent = scenario.market.agent_set.get(name="Thermal A")
    agent.budget = Decimal("123456789012345678.91")
    agent.save()
    document = _document(_export(client, scenario))
    assert document["agents"][0]["budget"] == "123456789012345678.91"


def test_export_rejects_runs_and_unknown_ids(client: Client) -> None:
    run = make_market().simulation
    assert client.get(reverse("export_scenario_yaml", args=[run.id])).status_code == 404
    assert client.get(reverse("export_scenario_yaml", args=[999999])).status_code == 404


# round trip ------------------------------------------------------------


def test_round_trip_into_an_empty_database(client: Client, scenario: Simulation) -> None:
    text = _export(client, scenario)
    original = _normalized(_document(text))
    Simulation.objects.all().delete()
    Agent.objects.all().delete()
    Resource.objects.all().delete()
    assert _counts() == (0, 0, 0, 0, 0, 0)

    preview = _preview(client, text)
    assert preview.status_code == 200, preview.content
    plan = preview.json()
    assert plan["resources"] == {"reuse": [], "create": ["Gas", "Wind"]}
    assert plan["agents"] == {"reuse": [], "create": ["Thermal A", "Green"]}
    assert plan["scenario"]["name"] == "Export test"
    assert plan["errors"] == {}
    assert _counts() == (0, 0, 0, 0, 0, 0)

    response = _confirm(client, text)
    assert response.status_code == 200, response.content
    body = response.json()
    imported = Simulation.objects.get(id=body["simulation_id"])
    assert body["edit_url"] == reverse("update_simulation", args=[imported.id])
    assert imported.proxy and imported.market.proxy
    assert _normalized(scenario_yaml.scenario_document(imported)) == original

    library = Agent.objects.filter(market__isnull=True)
    assert sorted(library.values_list("name", flat=True)) == ["Green", "Thermal A"]
    assert all(agent.proxy for agent in library)
    for agent in imported.market.agent_set.all():
        assert agent.proxy
        assert agent.source is not None and agent.source.name == agent.name


def test_reimport_into_the_same_database_reuses_everything(
    client: Client, scenario: Simulation
) -> None:
    text = _export(client, scenario)
    # "Thermal A" is a scenario name; the library agent is "Thermal". Make the
    # file refer to library names so both agents are reused.
    document = _document(text)
    document["agents"][0]["name"] = "thermal"
    text = _dump(document)
    resources_before = Resource.objects.count()
    library_before = Agent.objects.filter(market__isnull=True).count()

    plan = _preview(client, text).json()
    assert plan["resources"] == {"reuse": ["Gas", "Wind"], "create": []}
    assert plan["agents"] == {"reuse": ["Thermal", "Green"], "create": []}

    response = _confirm(client, text)
    assert response.status_code == 200, response.content
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.id != scenario.id
    assert Resource.objects.count() == resources_before
    assert Agent.objects.filter(market__isnull=True).count() == library_before
    copies = {agent.name: agent for agent in imported.market.agent_set.all()}
    assert set(copies) == {"thermal", "Green"}
    assert copies["thermal"].source.name == "Thermal"


def test_exported_scenario_names_create_new_library_agents(
    client: Client, scenario: Simulation
) -> None:
    text = _export(client, scenario)
    plan = _preview(client, text).json()
    assert plan["agents"] == {"reuse": ["Green"], "create": ["Thermal A"]}
    response = _confirm(client, text)
    assert response.status_code == 200
    assert Agent.objects.filter(market__isnull=True, name="Thermal A").exists()


def test_resource_name_match_is_case_insensitive(client: Client, scenario: Simulation) -> None:
    document = _document(_export(client, scenario))
    document["resources"][0]["name"] = "GAS"
    for agent in document["agents"]:
        for plant in agent["plants"]:
            if plant["resource"] == "Gas":
                plant["resource"] = "gas"
    plan = _preview(client, _dump(document)).json()
    assert plan["resources"]["reuse"] == ["Gas", "Wind"]


def test_preview_and_confirm_agree(client: Client, scenario: Simulation) -> None:
    text = _export(client, scenario)
    preview = _preview(client, text).json()
    response = _confirm(client, text)
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    created = Agent.objects.filter(market__isnull=True, name__in=preview["agents"]["create"])
    assert created.count() == len(preview["agents"]["create"])
    assert [a.name for a in imported.market.agent_set.order_by("id")] == preview["scenario"]["agents"]


# conflicts -------------------------------------------------------------


def test_resource_conflict_lists_the_differences(client: Client, scenario: Simulation) -> None:
    document = _document(_export(client, scenario))
    document["resources"][0]["fuel_cost"] = 46
    document["resources"][0]["energy_type"] = "NUCLEAR"
    errors = _assert_rejected(client, _dump(document), "resources[0]", "fuel_cost: file 46 vs existing 45.25")
    assert "energy_type: file NUCLEAR vs existing FOSSIL" in errors["resources[0]"][0]


def test_library_agent_conflict_lists_the_agents(client: Client, scenario: Simulation) -> None:
    document = _document(_export(client, scenario))
    document["agents"][1]["budget"] = 10
    document["agents"][1]["plants"][0]["capacity"] = 201
    errors = _assert_rejected(client, _dump(document), "agents[1]", "budget: file 10 vs existing 0")
    assert "plants: file [Wind 201 MW] vs existing [Wind 200 MW]" in errors["agents[1]"][0]
    assert errors["agents"] == [
        "Conflicting library agents: Green. Rename them in the file or align them with the library."
    ]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("algorithm", "DDPG", "algorithm: file DDPG vs existing MARGINAL_COST"),
        ("type", "HYBRID", "type: file HYBRID vs existing RENEWABLE"),
        ("algorithm_params", {"markup": 0.2}, "algorithm_params"),
    ],
)
def test_library_agent_conflict_per_field(
    client: Client, scenario: Simulation, field: str, value: Any, fragment: str
) -> None:
    document = _document(_export(client, scenario))
    document["agents"][1][field] = value
    if field == "algorithm_params":
        document["agents"][1]["algorithm"] = "FIXED_MARKUP"
    _assert_rejected(client, _dump(document), "agents[1]", fragment)


def test_failstack_conflict(client: Client, scenario: Simulation) -> None:
    document = _document(_export(client, scenario))
    document["agents"][0]["name"] = "Thermal"
    document["agents"][0]["failstack"]["delta"] = 13
    _assert_rejected(client, _dump(document), "agents[0]", "failstack.delta: file 13 vs existing 12.5")


# validation ------------------------------------------------------------


def _base_document(client: Client, scenario: Simulation) -> dict[str, Any]:
    return _document(_export(client, scenario))


@pytest.mark.parametrize(
    ("mutate", "key", "fragment"),
    [
        (lambda d: d.pop("format_version"), "format_version", "missing"),
        (lambda d: d.update(format_version=2), "format_version", "Unsupported format_version 2"),
        (lambda d: d.update(extra=1), "file", "Unknown key(s) in file: extra"),
        (lambda d: d["scenario"].update(colour="red"), "scenario", "Unknown key(s) in scenario: colour"),
        (lambda d: d["agents"][0].update(colour="red"), "agents[0]", "Unknown key(s)"),
        (lambda d: d["agents"][0]["failstack"].update(x=1), "agents[0].failstack", "Unknown key(s)"),
        (lambda d: d["resources"][0].pop("emission"), "resources[0]", "Missing key(s)"),
        (lambda d: d["scenario"].update(name="x" * 21), "scenario.name", "at most 20"),
        (lambda d: d["scenario"].update(mode="FAST"), "scenario.mode", "must be one of"),
        (lambda d: d["scenario"].update(seed=2**31), "scenario.seed", "Seed must be an integer"),
        (lambda d: d["scenario"].update(evaluation_periods=168), "scenario.evaluation_periods", "less than"),
        (lambda d: d["scenario"].update(observation_lags=[1, 2]), "scenario.observation_lags", "must be one of"),
        (lambda d: d["market"].update(bid_bounds=[10, 10]), "market.bid_bounds", "greater than"),
        (lambda d: d["market"].update(bid_bounds=[0.5, 10]), "market.bid_bounds", "must be an integer"),
        (lambda d: d["market"].update(bid_bounds=5), "market.bid_bounds", "list [low, high]"),
        (lambda d: d["agents"][0].update(name="y" * 21), "agents[0].name", "at most 20"),
        (lambda d: d["agents"][0].update(plants=[]), "agents[0].plants", "At least one plant"),
        (
            lambda d: d["agents"][0]["plants"][0].update(capacity=0),
            "agents[0].plants",
            "capacity must be a positive integer",
        ),
        (
            lambda d: d["agents"][0]["plants"][0].update(resource="Nope"),
            "agents[0].plants",
            "is not listed under resources",
        ),
        (lambda d: d["agents"][1].update(type="FOSSIL"), "agents[1].plants", "may only own FOSSIL"),
        (lambda d: d["agents"][0].update(algorithm="QLEARN"), "agents[0].algorithm", "must be one of"),
        (
            lambda d: d["agents"][1]["failstack"].update(enabled=True),
            "agents[1].failstack.enabled",
            "cannot be enabled for the baseline",
        ),
        (lambda d: d["agents"][0]["failstack"].update(limit=0), "agents[0].failstack.limit", ">= 1"),
        (
            lambda d: d["agents"][0].update(algorithm_params={"bogus": 1}),
            "agents[0].algorithm_params",
            "unknown parameter",
        ),
        (lambda d: d["agents"][0].update(budget="abc"), "agents[0].budget", ""),
        (
            lambda d: d["agents"].append(copy.deepcopy(d["agents"][1]) | {"name": "GREEN"}),
            "agents[2].name",
            "compared case-insensitively",
        ),
        (
            lambda d: d["resources"].append(copy.deepcopy(d["resources"][0]) | {"name": "gas"}),
            "resources[2].name",
            "listed twice",
        ),
        (lambda d: d["resources"][0].update(fuel_cost=12345.5), "resources[0].fuel_cost", "digits"),
    ],
)
def test_invalid_documents_write_nothing(
    client: Client, scenario: Simulation, mutate: Any, key: str, fragment: str
) -> None:
    document = _base_document(client, scenario)
    mutate(document)
    _assert_rejected(client, _dump(document), key, fragment)


def test_python_tags_are_rejected(client: Client, scenario: Simulation) -> None:
    text = "!!python/object/apply:os.system ['echo hacked']\n"
    _assert_rejected(client, text, "file", "not valid YAML")
    text = "format_version: 1\nscenario: !!python/object:abmem.models.Simulation {}\n"
    _assert_rejected(client, text, "file", "not valid YAML")


def test_size_limit(client: Client, scenario: Simulation) -> None:
    text = "# " + "x" * scenario_yaml.MAX_FILE_BYTES + "\nformat_version: 1\n"
    _assert_rejected(client, text, "file", "larger than 1024 KB")
    with pytest.raises(scenario_yaml.ScenarioImportError):
        scenario_yaml.load_yaml_bytes(b"x" * (scenario_yaml.MAX_FILE_BYTES + 1))


def test_missing_file_and_non_mapping(client: Client) -> None:
    response = client.post(reverse("import_scenario_preview"))
    assert response.status_code == 400
    assert "file" in response.json()["errors"]
    assert _preview(client, "- 1\n- 2\n").json()["errors"] == {
        "file": ["The file must contain a YAML mapping."]
    }
    assert client.get(reverse("import_scenario_preview")).status_code == 405
    assert client.get(reverse("import_scenario_confirm")).status_code == 405


def test_import_requires_csrf(scenario: Simulation) -> None:
    client = Client(enforce_csrf_checks=True)
    text = _export(client, scenario)
    before = _counts()
    assert _confirm(client, text).status_code == 403
    assert _counts() == before


def test_quoted_decimal_strings_round_trip(client: Client, scenario: Simulation) -> None:
    agent = scenario.market.agent_set.get(name="Thermal A")
    agent.budget = Decimal("123456789012345678.91")
    agent.save()
    text = _export(client, scenario)
    Simulation.objects.all().delete()
    Agent.objects.all().delete()
    response = _confirm(client, text)
    assert response.status_code == 200, response.content
    imported = Simulation.objects.get(id=response.json()["simulation_id"])
    assert imported.market.agent_set.get(name="Thermal A").budget == Decimal("123456789012345678.91")


# pages -----------------------------------------------------------------


def test_templates_list_renders_yaml_buttons(client: Client, scenario: Simulation) -> None:
    html = client.get(reverse("simulation_list_proxy")).content.decode()
    assert f'href="{reverse("export_scenario_yaml", args=[scenario.id])}"' in html
    assert 'aria-label="Export Export test as YAML"' in html
    assert 'id="import-yaml-button"' in html
    assert 'id="import-modal"' in html
    assert f'data-preview-url="{reverse("import_scenario_preview")}"' in html
    assert f'data-confirm-url="{reverse("import_scenario_confirm")}"' in html
    assert 'accept=".yaml,.yml' in html


def test_wizard_library_cards_are_limited_to_one_row(client: Client) -> None:
    html = client.get(reverse("create_simulation")).content.decode()
    assert 'id="library-show-all"' in html
    assert "function applyLibraryLimit()" in html
    assert "'Show all (' +" in html
    assert "window.addEventListener('resize'" in html


def _inline_scripts(html: str) -> list[str]:
    return [
        body
        for attrs, body in re.findall(r"<script([^>]*)>(.*?)</script>", html, flags=re.S)
        if "src=" not in attrs and "application/json" not in attrs and body.strip()
    ]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_inline_scripts_pass_node_check(
    client: Client, scenario: Simulation, tmp_path: pathlib.Path
) -> None:
    pages = {
        "list_proxy": reverse("simulation_list_proxy"),
        "create": reverse("create_simulation"),
        "update": reverse("update_simulation", args=[scenario.id]),
    }
    for name, url in pages.items():
        scripts = _inline_scripts(client.get(url).content.decode())
        assert scripts, name
        for index, body in enumerate(scripts):
            path = tmp_path / f"{name}_{index}.js"
            path.write_text(body, encoding="utf-8")
            result = subprocess.run(
                [NODE, "--check", str(path)], capture_output=True, text=True, check=False
            )
            assert result.returncode == 0, f"{name} script {index}: {result.stderr}"
