"""Demand profile library service (PLAN 2.9, spec demand_profile_library)."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from django.db import IntegrityError, connection, transaction
from django.db.migrations.executor import MigrationExecutor

from abmem.models import DemandProfileTemplate, Simulation
from abmem.models.demand import DEMAND_PROFILE_NAME_MAX_LENGTH
from abmem.services.agent import library_service
from abmem.services.market import demand_library
from abmem.services.market.demand_library import DemandLibraryError
from tests.factories import make_market, make_simulation

pytestmark = pytest.mark.django_db

PROFILE: dict[str, Any] = {
    "version": 1,
    "seed": 11,
    "day_start_hour": 7,
    "day_end_hour": 21,
    "autocorrelation": 0.8,
    "morning_ramp_hours": 0,
    "evening_ramp_hours": 0,
    "seasons": None,
    "day": {
        "distribution": "NORMAL",
        "mean": 900.0,
        "std": 100.0,
        "min": 600.0,
        "max": 1200.0,
    },
    "night": {"distribution": "UNIFORM", "min": 400.0, "max": 650.0},
}


def _scenario(
    template: DemandProfileTemplate | None, *, proxy: bool = True, name: str = "scn"
) -> Simulation:
    simulation = make_simulation(name=name, save=False)
    simulation.proxy = proxy
    simulation.currentPeriod = -1
    simulation.demand_profile = copy.deepcopy(template.profile) if template else None
    simulation.demand_profile_source = template
    simulation.save()
    make_market(simulation)
    return simulation


def _errors(exc_info: pytest.ExceptionInfo[DemandLibraryError]) -> dict[str, list[str]]:
    return exc_info.value.message_dict


# create ---------------------------------------------------------------------


def test_create_stores_the_normalised_profile() -> None:
    raw = copy.deepcopy(PROFILE)
    del raw["version"]
    raw["seed"] = "11"
    template = demand_library.create_profile("  Winter  ", raw, " cold days ")
    template.refresh_from_db()
    assert template.name == "Winter"
    assert template.description == "cold days"
    assert template.profile == PROFILE


def test_names_are_unique_case_insensitively() -> None:
    demand_library.create_profile("Winter", PROFILE)
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.create_profile("WINTER", PROFILE)
    assert "already exists" in _errors(exc_info)["name"][0]
    assert DemandProfileTemplate.objects.count() == 1


def test_database_constraint_rejects_case_variants() -> None:
    DemandProfileTemplate.objects.create(name="Base", profile=PROFILE)
    with pytest.raises(IntegrityError), transaction.atomic():
        DemandProfileTemplate.objects.create(name="base", profile=PROFILE)


@pytest.mark.parametrize(
    "name, fragment",
    [
        ("", "required"),
        ("   ", "required"),
        ("x" * (DEMAND_PROFILE_NAME_MAX_LENGTH + 1), "at most"),
        (None, "must be a string"),
    ],
)
def test_invalid_names(name: Any, fragment: str) -> None:
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.create_profile(name, PROFILE)
    assert fragment in _errors(exc_info)["name"][0]


def test_invalid_profile_errors_are_keyed_per_field() -> None:
    bad = copy.deepcopy(PROFILE)
    bad["day"]["std"] = 0
    bad["day_end_hour"] = 3
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.create_profile("", bad, description=5)  # type: ignore[arg-type]
    errors = _errors(exc_info)
    assert {"name", "profile.day.std", "profile.day_end_hour", "description"} <= set(
        errors
    )
    assert DemandProfileTemplate.objects.count() == 0


def test_profile_that_is_not_a_mapping() -> None:
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.create_profile("P", [1, 2])  # type: ignore[arg-type]
    assert "profile" in _errors(exc_info)


def test_validate_profile_uses_the_given_key() -> None:
    bad = dict(PROFILE, autocorrelation=1)
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.validate_profile(bad, key="demand")
    assert set(_errors(exc_info)) == {"demand.autocorrelation"}
    assert demand_library.validate_profile(PROFILE) == PROFILE


# update ---------------------------------------------------------------------


def test_update_keeps_its_own_name_and_rejects_taken_names() -> None:
    first = demand_library.create_profile("First", PROFILE)
    demand_library.create_profile("Second", PROFILE)
    changed = dict(PROFILE, seed=99)
    demand_library.update_profile(first, "FIRST", changed, "new")
    first.refresh_from_db()
    assert (first.name, first.profile["seed"], first.description) == (
        "FIRST",
        99,
        "new",
    )
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.update_profile(first, "second", PROFILE)
    assert "name" in _errors(exc_info)
    first.refresh_from_db()
    assert first.name == "FIRST"


def test_scenario_copy_does_not_change_when_the_library_profile_is_updated() -> None:
    template = demand_library.create_profile("Lib", PROFILE)
    scenario = _scenario(template)
    demand_library.update_profile(template, "Lib", dict(PROFILE, seed=12345))
    scenario.refresh_from_db()
    assert scenario.demand_profile == PROFILE
    assert scenario.demand_profile_source_id == template.id


# duplicate --------------------------------------------------------------------


def test_duplicate_picks_the_first_free_copy_name() -> None:
    template = demand_library.create_profile("Summer", PROFILE, "hot")
    first = demand_library.duplicate_profile(template)
    second = demand_library.duplicate_profile(template)
    assert first.name == "Summer (copy)"
    assert second.name == "Summer (copy 2)"
    assert first.profile == PROFILE and first.description == "hot"
    assert first.id != template.id


def test_copy_name_fits_the_column() -> None:
    name = "y" * DEMAND_PROFILE_NAME_MAX_LENGTH
    copy_name = demand_library.copy_profile_name(name)
    assert len(copy_name) <= DEMAND_PROFILE_NAME_MAX_LENGTH
    assert copy_name.endswith(" (copy)")


# delete -----------------------------------------------------------------------


def test_delete_is_blocked_by_a_scenario() -> None:
    template = demand_library.create_profile("Used", PROFILE)
    scenario = _scenario(template, name="uses-it")
    assert demand_library.scenarios_using(template) == [scenario]
    with pytest.raises(DemandLibraryError) as exc_info:
        demand_library.delete_profile(template)
    message = _errors(exc_info)["profile"][0]
    assert "'uses-it'" in message and f"#{scenario.id}" in message
    assert exc_info.value.error_dict["profile"][0].code == "in_use"
    assert DemandProfileTemplate.objects.filter(id=template.id).exists()


def test_delete_is_allowed_when_only_runs_use_it() -> None:
    template = demand_library.create_profile("RunOnly", PROFILE)
    run = _scenario(template, proxy=False)
    assert demand_library.scenarios_using(template) == []
    demand_library.delete_profile(template)
    run.refresh_from_db()
    assert run.demand_profile_source is None
    assert run.demand_profile == PROFILE


def test_scenarios_using_is_distinct_and_ordered() -> None:
    template = demand_library.create_profile("Many", PROFILE)
    other = demand_library.create_profile("Other", PROFILE)
    b = _scenario(template, name="b")
    a = _scenario(template, name="a")
    _scenario(other, name="c")
    assert demand_library.scenarios_using(template) == [b, a]


# lookup -----------------------------------------------------------------------


def test_find_by_name() -> None:
    template = demand_library.create_profile("Spring", PROFILE)
    assert demand_library.find_by_name(" spring ") == template
    assert demand_library.find_by_name("autumn") is None
    assert demand_library.find_by_name("") is None
    assert demand_library.find_by_name(None) is None  # type: ignore[arg-type]


# scenario copies keep the link ------------------------------------------------


def test_duplicate_and_clone_copy_the_link() -> None:
    template = demand_library.create_profile("Linked", PROFILE)
    scenario = _scenario(template)
    duplicate = library_service.duplicate_scenario(scenario)
    run = library_service.clone_scenario_for_run(scenario, seed=3)
    assert duplicate.demand_profile_source_id == template.id
    assert run.demand_profile_source_id == template.id
    assert run.demand_profile == PROFILE


# migration ----------------------------------------------------------------------


def test_migration_0016_is_applied() -> None:
    executor = MigrationExecutor(connection)
    applied = executor.loader.applied_migrations
    assert ("abmem", "0016_demand_profile_template") in applied
    assert not executor.migration_plan(executor.loader.graph.leaf_nodes())
