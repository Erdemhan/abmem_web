"""Forward and reverse tests of the K5 library import (migration 0008, OQ-15).

Uses the manual ``MigrationExecutor`` pattern on the test database: migrate
back to 0007, build data with the historical models, migrate forward to 0009,
check, migrate back to 0007, check, and always restore the latest schema.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Any

import pytest
from django.db import connection
from django.db.migrations.executor import MigrationExecutor


APP = "abmem"
BEFORE = [(APP, "0007_agent_source")]
AFTER = [(APP, "0009_agent_name_constraints")]

pytestmark = pytest.mark.django_db(transaction=True)


def _increment_suffix(name: str) -> str:
    """Load ``increment_suffix`` from the migration module and apply it."""
    import importlib

    module = importlib.import_module("abmem.migrations.0008_agent_library_import")
    return module.increment_suffix(name)


def _migrate(targets: Sequence[tuple[str, str]]) -> Any:
    """Migrate the test database to ``targets`` and return the historical apps.

    Args:
        targets: Migration targets.

    Returns:
        The app registry of the project state at ``targets``.
    """
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
def at_0007() -> Iterator[Any]:
    """Yield the historical apps at 0007 and restore the latest schema afterwards."""
    try:
        yield _migrate(BEFORE)
    finally:
        _restore_latest()


class _Builder:
    """Creates historical rows at migration state 0007."""

    def __init__(self, apps: Any) -> None:
        self.apps = apps
        self.simulation = apps.get_model(APP, "Simulation")
        self.market = apps.get_model(APP, "Market")
        self.agent = apps.get_model(APP, "Agent")
        self.portfolio = apps.get_model(APP, "Portfolio")
        self.plant = apps.get_model(APP, "Plant")
        self.resource = apps.get_model(APP, "Resource")
        self._resources: dict[str, Any] = {}

    def resource_named(self, name: str) -> Any:
        if name not in self._resources:
            self._resources[name] = self.resource.objects.create(
                name=name, energyType=1
            )
        return self._resources[name]

    def make_market(self, name: str, proxy: bool = True) -> Any:
        simulation = self.simulation.objects.create(
            name=name,
            mode=0,
            state=0,
            periodType=0,
            periodNumber=10,
            currentPeriod=0,
            day=0,
            proxy=proxy,
        )
        return self.market.objects.create(
            strategy=0,
            lowerBidBound=0,
            upperBidBound=100,
            simulation=simulation,
            proxy=proxy,
        )

    def make_agent(
        self,
        market: Any,
        name: str,
        plants: Sequence[tuple[str, int]],
        proxy: bool = True,
        **fields: Any,
    ) -> Any:
        agent = self.agent.objects.create(
            name=name, market=market, proxy=proxy, **fields
        )
        portfolio = self.portfolio.objects.create(agent=agent, proxy=proxy)
        for resource_name, capacity in plants:
            self.plant.objects.create(
                portfolio=portfolio,
                resource=self.resource_named(resource_name),
                capacity=capacity,
                proxy=proxy,
            )
        return agent


def _library(apps: Any) -> dict[str, Any]:
    agent_model = apps.get_model(APP, "Agent")
    return {a.name: a for a in agent_model.objects.filter(market__isnull=True)}


def _plants(apps: Any, agent_id: int) -> list[tuple[str, int, bool]]:
    plant_model = apps.get_model(APP, "Plant")
    return sorted(
        (p.resource.name, p.capacity, p.proxy)
        for p in plant_model.objects.filter(portfolio__agent_id=agent_id)
    )


def test_increment_suffix_semantics() -> None:
    assert _increment_suffix("CCGT") == "CCGT_2"
    assert _increment_suffix("CCGT_2") == "CCGT_3"
    assert _increment_suffix("A" * 20) == "A" * 18 + "_2"
    assert _increment_suffix("A" * 18 + "_9") == "A" * 17 + "_10"


def test_empty_database_forward_and_reverse_are_noops(at_0007: Any) -> None:
    apps = _migrate(AFTER)
    assert apps.get_model(APP, "Agent").objects.count() == 0
    apps = _migrate(BEFORE)
    assert apps.get_model(APP, "Agent").objects.count() == 0


def test_import_deduplicates_suffixes_and_reverses(at_0007: Any) -> None:
    b = _Builder(at_0007)
    old = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)

    # Pre-existing library agents: a reusable one and a case collision pair.
    nuclear_lib = b.make_agent(None, "NUCLEAR", [("NUC", 10)])
    lib_a = b.make_agent(None, "Lib", [("NUC", 1)])
    lib_b = b.make_agent(None, "LIB", [("NUC", 2)])
    b.agent.objects.filter(id__in=[nuclear_lib.id, lib_a.id, lib_b.id]).update(
        created_at=old
    )

    m1, m2, m3 = b.make_market("s1"), b.make_market("s2"), b.make_market("s3")
    run_market = b.make_market("run", proxy=False)

    ccgt1 = b.make_agent(m1, "CCGT", [("CCGT", 100)])
    ccgt1_twin = b.make_agent(m1, "CCGT2", [("CCGT", 100)])
    ccgt2 = b.make_agent(m2, "CCGT", [("CCGT", 200)])
    ccgt3 = b.make_agent(m3, "CCGT", [("CCGT", 300)])
    coal_lower = b.make_agent(m2, "coal", [("COAL", 50)])
    coal_upper = b.make_agent(m3, "COAL", [("COAL", 50)])
    coal_diff = b.make_agent(m1, "Coal", [("COAL", 70)])
    long1 = b.make_agent(m1, "ABCDEFGHIJKLMNOPQRST", [("COAL", 1)])
    long2 = b.make_agent(m2, "ABCDEFGHIJKLMNOPQRST", [("COAL", 2)])
    x1 = b.make_agent(m1, "X_2", [("XR", 1)])
    x2 = b.make_agent(m2, "X_2", [("XR", 2)])
    nuc_same = b.make_agent(m1, "nuclear", [("NUC", 10)])
    nuc_diff = b.make_agent(m2, "NUCLEAR", [("NUC", 20)])
    mix1 = b.make_agent(m1, "MIX", [("CCGT", 10), ("COAL", 20)])
    mix2 = b.make_agent(m2, "MIX", [("COAL", 20), ("CCGT", 10)])
    hydro1 = b.make_agent(m1, "HYDRO", [("HYD", 5)], budget=Decimal("0"))
    hydro2 = b.make_agent(m2, "HYDRO", [("HYD", 5)], budget=Decimal("5"))
    wind1 = b.make_agent(m1, "WIND", [("WIN", 5)])
    wind2 = b.make_agent(m2, "WIND", [("WIN", 5)], failstack_enabled=True)
    run_agent = b.make_agent(run_market, "CCGT", [("CCGT", 100)], proxy=False)

    apps = _migrate(AFTER)
    agent_model = apps.get_model(APP, "Agent")
    library = _library(apps)

    def source_name(agent: Any) -> str | None:
        source = agent_model.objects.get(id=agent.id).source
        return None if source is None else source.name

    assert sorted(library) == sorted(
        [
            "NUCLEAR",
            "NUCLEAR_2",
            "Lib",
            "LIB_2",
            "CCGT",
            "CCGT_2",
            "CCGT_3",
            "coal",
            "Coal_2",
            "ABCDEFGHIJKLMNOPQRST",
            "ABCDEFGHIJKLMNOPQR_2",
            "X_2",
            "X_3",
            "MIX",
            "HYDRO",
            "HYDRO_2",
            "WIND",
            "WIND_2",
        ]
    )
    assert source_name(ccgt1) == source_name(ccgt1_twin) == "CCGT"
    assert source_name(ccgt2) == "CCGT_2"
    assert source_name(ccgt3) == "CCGT_3"
    assert source_name(coal_lower) == source_name(coal_upper) == "coal"
    assert source_name(coal_diff) == "Coal_2"
    assert source_name(long1) == "ABCDEFGHIJKLMNOPQRST"
    assert source_name(long2) == "ABCDEFGHIJKLMNOPQR_2"
    assert source_name(x1) == "X_2"
    assert source_name(x2) == "X_3"
    assert agent_model.objects.get(id=nuc_same.id).source_id == nuclear_lib.id
    assert source_name(nuc_diff) == "NUCLEAR_2"
    assert source_name(mix1) == source_name(mix2) == "MIX"
    assert source_name(hydro2) == "HYDRO_2" and source_name(hydro1) == "HYDRO"
    assert source_name(wind2) == "WIND_2" and source_name(wind1) == "WIND"
    assert source_name(run_agent) is None

    created = library["CCGT_2"]
    assert created.proxy is True and created.market_id is None and created.state == 0
    assert _plants(apps, created.id) == [("CCGT", 200, True)]
    assert _plants(apps, library["MIX"].id) == [("CCGT", 10, True), ("COAL", 20, True)]
    assert library["WIND_2"].failstack_enabled is True
    assert library["HYDRO_2"].budget == Decimal("5")
    portfolio_model = apps.get_model(APP, "Portfolio")
    assert portfolio_model.objects.get(agent_id=created.id).proxy is True

    # Invariant: every scenario agent points to a library agent with equal plants.
    for agent in agent_model.objects.filter(proxy=True, market__isnull=False):
        assert agent.source_id is not None
        assert agent.source.type == agent.type and agent.source.budget == agent.budget
        assert [p[:2] for p in _plants(apps, agent.source_id)] == [
            p[:2] for p in _plants(apps, agent.id)
        ]

    apps = _migrate(BEFORE)
    agent_model = apps.get_model(APP, "Agent")
    assert sorted(_library(apps)) == ["LIB_2", "Lib", "NUCLEAR"]
    assert not agent_model.objects.filter(source__isnull=False).exists()
    assert agent_model.objects.filter(market__isnull=False).count() == 20
    plant_model = apps.get_model(APP, "Plant")
    assert (
        not plant_model.objects.filter(portfolio__agent__market__isnull=True)
        .exclude(portfolio__agent_id__in=[nuclear_lib.id, lib_a.id, lib_b.id])
        .exists()
    )


def test_duplicate_names_in_one_market_abort_the_import(at_0007: Any) -> None:
    b = _Builder(at_0007)
    market = b.make_market("dup")
    b.make_agent(market, "CCGT", [("CCGT", 1)])
    duplicate = b.make_agent(market, "CCGT", [("CCGT", 2)])

    with pytest.raises(RuntimeError, match="CCGT"):
        _migrate(AFTER)

    # The failed migration left the data untouched; fixing it lets the chain run.
    apps = _migrate(BEFORE)
    apps.get_model(APP, "Agent").objects.filter(id=duplicate.id).delete()
    apps = _migrate(AFTER)
    assert sorted(_library(apps)) == ["CCGT"]
