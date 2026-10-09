"""Starter data, resource list YAML and the root redirect (Docker setup)."""

from __future__ import annotations

from decimal import Decimal
from io import StringIO

import pytest
from django.core.management import call_command
from django.test import Client

from abmem.management.commands.load_starter_data import (
    STARTER_DIR,
    database_is_empty,
    load_starter_data,
)
from abmem.models import Agent, Resource, Simulation
from abmem.models.enums import EnergyType
from abmem.scenario_yaml import (
    ScenarioImportError,
    export_resources_yaml,
    import_resources,
    load_yaml_bytes,
)

pytestmark = pytest.mark.django_db


def _resource(name: str, fuel: str = "5.00") -> Resource:
    return Resource.objects.create(
        name=name,
        energyType=EnergyType.FOSSIL,
        fuelCost=Decimal(fuel),
        omCost=Decimal("1.00"),
        investmentCost=Decimal("2.00"),
        co2Cost=Decimal("0.00"),
        emission=Decimal("0.00"),
    )


def test_starter_files_exist() -> None:
    assert (STARTER_DIR / "resources.yaml").is_file()
    assert len(list((STARTER_DIR / "scenarios").glob("*.yaml"))) >= 1


def test_load_into_empty_database() -> None:
    assert database_is_empty()
    counts = load_starter_data()
    assert counts["resources"] == Resource.objects.count() > 0
    assert counts["scenarios"] == Simulation.objects.filter(proxy=True).count() > 0
    assert Agent.objects.filter(market__isnull=True).exists()
    assert not database_is_empty()


def test_command_skips_a_database_with_data() -> None:
    _resource("Existing")
    out = StringIO()
    call_command("load_starter_data", stdout=out)
    assert "skipped" in out.getvalue()
    assert Resource.objects.count() == 1


def test_command_loads_once() -> None:
    call_command("load_starter_data", stdout=StringIO())
    scenarios = Simulation.objects.count()
    out = StringIO()
    call_command("load_starter_data", stdout=out)
    assert "skipped" in out.getvalue()
    assert Simulation.objects.count() == scenarios


def test_resource_round_trip_reuses_existing() -> None:
    first = _resource("Gas")
    document = load_yaml_bytes(export_resources_yaml([first]).encode())
    assert import_resources(document) == [first]
    assert Resource.objects.count() == 1


def test_resource_import_creates_new() -> None:
    document = load_yaml_bytes(export_resources_yaml([_resource("Gas")]).encode())
    Resource.objects.all().delete()
    created = import_resources(document)
    assert [r.name for r in created] == ["Gas"]
    assert created[0].fuelCost == Decimal("5.00")


def test_resource_import_conflict_writes_nothing() -> None:
    document = load_yaml_bytes(
        export_resources_yaml([_resource("Gas"), _resource("Oil")]).encode()
    )
    Resource.objects.all().delete()
    _resource("gas", fuel="9.00")
    with pytest.raises(ScenarioImportError) as info:
        import_resources(document)
    assert "resources[0]" in info.value.errors
    assert list(Resource.objects.values_list("name", flat=True)) == ["gas"]


@pytest.mark.parametrize(
    ("document", "key"),
    [
        ({"format_version": 2, "kind": "resources", "resources": []}, "format_version"),
        ({"format_version": 1, "kind": "scenario", "resources": []}, "kind"),
        ({"format_version": 1, "kind": "resources", "resources": "x"}, "resources"),
        ([], "document"),
    ],
)
def test_resource_import_rejects_invalid_documents(document: object, key: str) -> None:
    with pytest.raises(ScenarioImportError) as info:
        import_resources(document)
    assert key in info.value.errors


def test_root_redirects_to_the_app() -> None:
    response = Client().get("/")
    assert response.status_code == 302
    assert response["Location"] == "/dev/"


def test_secret_key_file_reader(tmp_path) -> None:
    from web_project.settings import _read_secret_file

    key = tmp_path / "secret_key"
    key.write_text("  abc123\n", encoding="utf-8")
    assert _read_secret_file(str(key)) == "abc123"
    assert _read_secret_file(str(tmp_path / "missing")) == ""
    assert _read_secret_file("") == ""
