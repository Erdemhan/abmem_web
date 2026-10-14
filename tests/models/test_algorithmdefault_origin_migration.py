"""Forward and reverse tests of migration 0021 (origin of algorithm defaults).

Uses the ``MigrationExecutor`` pattern of ``test_algorithm_key_migration``:
migrate back to 0020, build a row with the historical models, migrate forward
to 0021, check, migrate back, check, and always restore the latest schema.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor

APP = "abmem"
BEFORE = [(APP, "0020_hpostudy_target")]
AFTER = [(APP, "0021_algorithmdefault_origin")]

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
def at_0020() -> Iterator[Any]:
    """Yield the historical apps at 0020 and restore the latest schema afterwards."""
    try:
        yield _migrate(BEFORE)
    finally:
        _restore_latest()


def test_existing_defaults_become_custom_without_study(at_0020: Any) -> None:
    at_0020.get_model(APP, "AlgorithmDefault").objects.create(
        algorithm_type="TD3", params={"gamma": 0.9}
    )
    apps = _migrate(AFTER)
    row = apps.get_model(APP, "AlgorithmDefault").objects.get(algorithm_type="TD3")
    assert row.origin == "CUSTOM"
    assert row.source_study_id is None
    assert row.params == {"gamma": 0.9}

    apps = _migrate(BEFORE)
    row = apps.get_model(APP, "AlgorithmDefault").objects.get(algorithm_type="TD3")
    assert row.params == {"gamma": 0.9}
