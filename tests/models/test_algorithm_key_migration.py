"""Forward and reverse tests of migration 0017 (algorithm keys, clearing algorithm).

Uses the ``MigrationExecutor`` pattern of ``test_library_import_migration``:
migrate back to 0016, build rows with the historical models, migrate forward
to 0017, check, migrate back, check, and always restore the latest schema.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

APP = "abmem"
BEFORE = [(APP, "0016_demand_profile_template")]
AFTER = [(APP, "0017_algorithm_keys_and_clearing_algorithm")]
LEGACY = {
    0: "DDPG",
    1: "RDPG",
    2: "TD3",
    3: "SAC",
    4: "PPO",
    5: "RANDOM",
    6: "MARGINAL_COST",
    7: "FIXED_MARKUP",
}

pytestmark = pytest.mark.django_db(transaction=True)


def _migrate(targets: Sequence[tuple[str, str]]) -> Any:
    """Migrate the test database to ``targets`` and return the historical apps."""
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    executor.migrate(list(targets))
    executor.loader.build_graph()
    return executor.loader.project_state(list(targets)).apps


def _restore_latest() -> None:
    """Migrate the test database back to the latest migrations of every app."""
    executor = MigrationExecutor(connection)
    executor.loader.build_graph()
    executor.migrate(executor.loader.graph.leaf_nodes())


@pytest.fixture
def at_0016() -> Iterator[Any]:
    """Yield the historical apps at 0016 and restore the latest schema afterwards."""
    try:
        yield _migrate(BEFORE)
    finally:
        _restore_latest()


def _rows(apps: Any) -> dict[str, list[Any]]:
    """Return ``algorithm_type`` of every row of the four models, ordered by id."""
    return {
        name: list(
            apps.get_model(APP, name)
            .objects.order_by("id")
            .values_list("algorithm_type", flat=True)
        )
        for name in ("Agent", "HPOStudy", "AlgorithmDefault", "PolicyCheckpoint")
    }


def _build(apps: Any) -> Any:
    """Create one row per legacy value in each of the four models at 0016."""
    simulation_model = apps.get_model(APP, "Simulation")
    market_model = apps.get_model(APP, "Market")
    agent_model = apps.get_model(APP, "Agent")
    simulation = simulation_model.objects.create(
        name="s",
        mode=0,
        state=0,
        periodType=0,
        periodNumber=10,
        currentPeriod=0,
        day=0,
        proxy=True,
    )
    market = market_model.objects.create(
        strategy=1,
        lowerBidBound=0,
        upperBidBound=100,
        simulation=simulation,
        proxy=True,
    )
    agents = [
        agent_model.objects.create(
            name=f"a{value}", market=market, proxy=True, algorithm_type=value
        )
        for value in LEGACY
    ]
    for value in LEGACY:
        apps.get_model(APP, "HPOStudy").objects.create(
            name=f"h{value}",
            algorithm_type=value,
            scenario=simulation,
            param_bounds={},
            colony_size=2,
            max_iter=1,
            limit=1,
            search_seeds=[1],
            evaluation_seeds=[2],
            objective={},
        )
        apps.get_model(APP, "AlgorithmDefault").objects.create(
            algorithm_type=value, params={"gamma": 0.9}
        )
        apps.get_model(APP, "PolicyCheckpoint").objects.create(
            simulation=simulation,
            agent=agents[value],
            algorithm_type=value,
            path=f"p{value}.pt",
            file_size=1,
            sha256="0" * 64,
            format_version=1,
        )
    return market


def test_forward_maps_values_to_keys_and_reverse_restores_them(at_0016: Any) -> None:
    market = _build(at_0016)
    keys = [LEGACY[v] for v in sorted(LEGACY)]

    apps = _migrate(AFTER)
    assert _rows(apps) == {
        "Agent": keys,
        "HPOStudy": keys,
        "AlgorithmDefault": keys,
        "PolicyCheckpoint": keys,
    }
    market_after = apps.get_model(APP, "Market").objects.get(pk=market.pk)
    assert market_after.clearing_algorithm == "MERIT_ORDER"
    assert (
        apps.get_model(APP, "Agent")._meta.get_field("algorithm_type").default == "DDPG"
    )

    apps = _migrate(BEFORE)
    values = sorted(LEGACY)
    assert _rows(apps) == {
        "Agent": values,
        "HPOStudy": values,
        "AlgorithmDefault": values,
        "PolicyCheckpoint": values,
    }


def test_new_default_and_unique_key_after_forward(at_0016: Any) -> None:
    apps = _migrate(AFTER)
    simulation = apps.get_model(APP, "Simulation").objects.create(
        name="s",
        mode=0,
        state=0,
        periodType=0,
        periodNumber=1,
        currentPeriod=0,
        day=0,
    )
    market = apps.get_model(APP, "Market").objects.create(
        strategy=0, lowerBidBound=0, upperBidBound=10, simulation=simulation
    )
    agent = apps.get_model(APP, "Agent").objects.create(name="n", market=market)
    assert agent.algorithm_type == "DDPG"
    assert market.clearing_algorithm == "MERIT_ORDER"
    default_model = apps.get_model(APP, "AlgorithmDefault")
    default_model.objects.create(algorithm_type="TD3")
    from django.db import IntegrityError, transaction

    with pytest.raises(IntegrityError), transaction.atomic():
        default_model.objects.create(algorithm_type="TD3")


def test_reverse_refuses_keys_without_legacy_value(at_0016: Any) -> None:
    apps = _migrate(AFTER)
    simulation = apps.get_model(APP, "Simulation").objects.create(
        name="s",
        mode=0,
        state=0,
        periodType=0,
        periodNumber=1,
        currentPeriod=0,
        day=0,
        proxy=True,
    )
    apps.get_model(APP, "HPOStudy").objects.create(
        name="plugin",
        algorithm_type="MY_PLUGIN",
        scenario=simulation,
        param_bounds={},
        colony_size=2,
        max_iter=1,
        limit=1,
        search_seeds=[1],
        evaluation_seeds=[2],
        objective={},
    )
    with pytest.raises(ValueError, match="MY_PLUGIN"):
        _migrate(BEFORE)
