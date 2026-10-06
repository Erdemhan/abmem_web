"""Data migration for decision K6: the CCGT resource is FOSSIL, not RENEWABLE."""

from __future__ import annotations

from django.apps.registry import Apps
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor

_RENEWABLE = 0
_FOSSIL = 1


def ccgt_to_fossil(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    """Set the energy type of resource 'CCGT' from RENEWABLE to FOSSIL.

    Args:
        apps: Historical app registry.
        schema_editor: Schema editor of the running migration (unused).
    """
    resource_model = apps.get_model("abmem", "Resource")
    resource_model.objects.filter(name__iexact="CCGT", energyType=_RENEWABLE).update(
        energyType=_FOSSIL
    )


def ccgt_to_renewable(apps: Apps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    """Reverse of ``ccgt_to_fossil``: set 'CCGT' back to RENEWABLE.

    Args:
        apps: Historical app registry.
        schema_editor: Schema editor of the running migration (unused).
    """
    resource_model = apps.get_model("abmem", "Resource")
    resource_model.objects.filter(name__iexact="CCGT", energyType=_FOSSIL).update(
        energyType=_RENEWABLE
    )


class Migration(migrations.Migration):
    dependencies = [
        ("abmem", "0003_remove_agent_seed"),
    ]

    operations = [
        migrations.RunPython(ccgt_to_fossil, ccgt_to_renewable),
    ]
