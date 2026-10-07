"""K5 data migration: import scenario agents into the agent library (PLAN 2.3, OQ-15).

Scenario agents (proxy True, market not NULL) with an identical configuration
(type, budget, algorithm_type, fail-stack fields and the multiset of
(resource, capacity) plants) map to ONE library agent (proxy True, market
NULL) named after the lowest-id member of the group. A name suffix ``_2``,
``_3``, ... is used only when the name is already taken (case-insensitively,
as the database ``LOWER`` computes it) by a library agent with a different
configuration. A pre-existing library agent with the same name and the same
configuration is reused. Every member gets ``source`` = its library agent.

Reverse: deletes the library agents this migration created and clears every
``source``. A created library agent is identified by being the source of a
scenario agent and by ``created_at`` lying in the window between the time
0007 and the time 0008 were recorded as applied in ``django_migrations``.
Renames of pre-existing colliding library agents are not undone.
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.recorder import MigrationRecorder
from django.db.models import Count

logger = logging.getLogger(__name__)

NAME_MAX_LENGTH: int = 20
AGENT_STATE_CREATED: int = 0
PREVIOUS_MIGRATION: str = "0007_agent_source"
THIS_MIGRATION: str = "0008_agent_library_import"
_SUFFIX_PATTERN = re.compile(r"_(\d+)$")

ConfigKey = tuple[Any, ...]


def increment_suffix(name: str, max_length: int = NAME_MAX_LENGTH) -> str:
    """Return the next suffixed candidate of ``name``.

    Same semantics as ``abmem.views.increment_suffix`` ('X' -> 'X_2',
    'X_2' -> 'X_3'), with the stem truncated so that the result has at most
    ``max_length`` characters.

    Args:
        name: Current candidate name.
        max_length: Maximum length of the returned name.

    Returns:
        The next candidate name.
    """
    match = _SUFFIX_PATTERN.search(name)
    if match:
        stem = name[: match.start()]
        number = int(match.group(1)) + 1
    else:
        stem = name
        number = 2
    suffix = f"_{number}"
    return stem[: max_length - len(suffix)] + suffix


def _db_lower(connection: BaseDatabaseWrapper, name: str) -> str:
    """Lower-case ``name`` with the database ``LOWER`` used by the constraint.

    Args:
        connection: Database connection of the running migration.
        name: Name to lower-case.

    Returns:
        ``LOWER(name)`` as computed by the database.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT LOWER(%s)", [name])
        return str(cursor.fetchone()[0])


def _config_key(agent: Any, plant_model: Any) -> ConfigKey:
    """Return the OQ-15 configuration key of an agent.

    Args:
        agent: Historical Agent instance.
        plant_model: Historical Plant model.

    Returns:
        Tuple of type, budget, algorithm_type, fail-stack fields and the
        sorted tuple of (resource_id, capacity) pairs of its plants.
    """
    plants = tuple(
        sorted(
            plant_model.objects.filter(portfolio__agent_id=agent.id).values_list(
                "resource_id", "capacity"
            )
        )
    )
    return (
        agent.type,
        Decimal(agent.budget),
        agent.algorithm_type,
        agent.failstack_enabled,
        agent.failstack_limit,
        Decimal(agent.failstack_delta),
        Decimal(agent.failstack_min_margin),
        plants,
    )


def _check_market_names(agent_model: Any) -> None:
    """Raise when a market holds two agents with the same (case-sensitive) name.

    Args:
        agent_model: Historical Agent model.

    Raises:
        RuntimeError: Listing every (market_id, name, count) duplicate.
    """
    duplicates = list(
        agent_model.objects.filter(market__isnull=False)
        .values("market_id", "name")
        .annotate(n=Count("id"))
        .filter(n__gt=1)
        .order_by("market_id", "name")
    )
    if duplicates:
        listing = ", ".join(
            f"market {d['market_id']}: {d['name']!r} x{d['n']}" for d in duplicates
        )
        raise RuntimeError(
            f"Duplicate agent names inside a market, fix them first: {listing}"
        )


def _free_name(
    connection: BaseDatabaseWrapper,
    base: str,
    names: dict[str, Any],
    accept: Any = None,
) -> tuple[str, Any]:
    """Find the first candidate of ``base`` that is free or held by ``accept``.

    Args:
        connection: Database connection of the running migration.
        base: Start name.
        names: Lower-cased library names mapped to their holder.
        accept: Callable(holder) -> bool, True when the holder may be reused.

    Returns:
        (candidate, holder) where holder is None for a free name, otherwise
        the reusable holder.
    """
    candidate = base[:NAME_MAX_LENGTH]
    while True:
        holder = names.get(_db_lower(connection, candidate))
        if holder is None:
            return candidate, None
        if accept is not None and accept(holder):
            return candidate, holder
        candidate = increment_suffix(candidate)


def import_library(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    """Forward K5 import (see module docstring).

    Args:
        apps: Historical app registry.
        schema_editor: Schema editor of the running migration.

    Raises:
        RuntimeError: If a market holds duplicate agent names.
    """
    agent_model = apps.get_model("abmem", "Agent")
    portfolio_model = apps.get_model("abmem", "Portfolio")
    plant_model = apps.get_model("abmem", "Plant")
    connection = schema_editor.connection

    _check_market_names(agent_model)

    # Existing library agents: resolve case-insensitive collisions among them.
    names: dict[str, Any] = {}
    for library_agent in agent_model.objects.filter(market__isnull=True).order_by("id"):
        candidate, _ = _free_name(connection, library_agent.name, names)
        if candidate != library_agent.name:
            logger.warning(
                "Library agent %s renamed from %r to %r (name collision).",
                library_agent.id,
                library_agent.name,
                candidate,
            )
            library_agent.name = candidate
            library_agent.save(update_fields=["name"])
        names[_db_lower(connection, candidate)] = library_agent

    # Group scenario agents by configuration, groups ordered by lowest member id.
    groups: dict[ConfigKey, list[Any]] = {}
    scenario_agents = agent_model.objects.filter(
        proxy=True, market__isnull=False
    ).order_by("id")
    for agent in scenario_agents:
        groups.setdefault(_config_key(agent, plant_model), []).append(agent)

    key_cache: dict[int, ConfigKey] = {}

    def holder_key(holder: Any) -> ConfigKey:
        if holder.id not in key_cache:
            key_cache[holder.id] = _config_key(holder, plant_model)
        return key_cache[holder.id]

    for key, members in groups.items():
        template = members[0]
        candidate, library_agent = _free_name(
            connection, template.name, names, accept=lambda h, k=key: holder_key(h) == k
        )
        if library_agent is None:
            library_agent = agent_model.objects.create(
                name=candidate,
                market=None,
                proxy=True,
                state=AGENT_STATE_CREATED,
                budget=template.budget,
                type=template.type,
                algorithm_type=template.algorithm_type,
                failstack_enabled=template.failstack_enabled,
                failstack_limit=template.failstack_limit,
                failstack_delta=template.failstack_delta,
                failstack_min_margin=template.failstack_min_margin,
            )
            portfolio = portfolio_model.objects.create(agent=library_agent, proxy=True)
            plant_model.objects.bulk_create(
                [
                    plant_model(
                        portfolio=portfolio,
                        resource_id=plant.resource_id,
                        capacity=plant.capacity,
                        proxy=True,
                    )
                    for plant in plant_model.objects.filter(
                        portfolio__agent_id=template.id
                    ).order_by("id")
                ]
            )
            key_cache[library_agent.id] = key
            names[_db_lower(connection, candidate)] = library_agent
        agent_model.objects.filter(id__in=[m.id for m in members]).update(
            source=library_agent
        )


def remove_library(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    """Reverse K5 import: delete the created library agents, clear ``source``.

    Args:
        apps: Historical app registry.
        schema_editor: Schema editor of the running migration.
    """
    agent_model = apps.get_model("abmem", "Agent")
    applied = dict(
        MigrationRecorder(schema_editor.connection)
        .migration_qs.filter(app="abmem", name__in=[PREVIOUS_MIGRATION, THIS_MIGRATION])
        .values_list("name", "applied")
    )
    created = agent_model.objects.filter(
        market__isnull=True,
        proxy=True,
        copies__market__isnull=False,
        copies__proxy=True,
    )
    if PREVIOUS_MIGRATION in applied:
        created = created.filter(created_at__gt=applied[PREVIOUS_MIGRATION])
    if THIS_MIGRATION in applied:
        created = created.filter(created_at__lte=applied[THIS_MIGRATION])
    created_ids = list(created.values_list("id", flat=True).distinct())
    agent_model.objects.filter(source__isnull=False).update(source=None)
    agent_model.objects.filter(id__in=created_ids).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", PREVIOUS_MIGRATION),
    ]

    operations = [
        migrations.RunPython(import_library, remove_library),
    ]
