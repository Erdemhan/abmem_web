"""Store agent algorithms as registry keys and add Market.clearing_algorithm (PLAN 2.10).

The four ``algorithm_type`` columns change from the integer value of the
legacy ``AlgorithmType`` enum to the text key of a registered algorithm class.
The column type change casts the integers to digit strings, then the data
step maps each digit string to the enum member name, which is the algorithm
key. The reverse step maps the names back to digit strings before the column
returns to an integer type. ``Market.clearing_algorithm`` is added with the
default ``MERIT_ORDER``, the clearing every existing market used.
"""

from __future__ import annotations

from typing import Any, Final

from django.db import migrations, models

# Values of the legacy AlgorithmType enum (abmem/models/enums.py). Written out
# here so the migration does not depend on live code.
LEGACY_ALGORITHM_KEYS: Final[dict[str, str]] = {
    "0": "DDPG",
    "1": "RDPG",
    "2": "TD3",
    "3": "SAC",
    "4": "PPO",
    "5": "RANDOM",
    "6": "MARGINAL_COST",
    "7": "FIXED_MARKUP",
}
LEGACY_ALGORITHM_VALUES: Final[dict[str, str]] = {
    key: value for value, key in LEGACY_ALGORITHM_KEYS.items()
}
MODELS: Final[tuple[str, ...]] = (
    "Agent",
    "HPOStudy",
    "AlgorithmDefault",
    "PolicyCheckpoint",
)


def _remap(apps: Any, mapping: dict[str, str]) -> None:
    """Rewrite ``algorithm_type`` of every row of :data:`MODELS` with ``mapping``.

    Values without an entry in ``mapping`` are left unchanged.

    Args:
        apps: Historical app registry.
        mapping: Old stored text to new stored text.
    """
    for model_name in MODELS:
        model = apps.get_model("abmem", model_name)
        for old, new in mapping.items():
            model.objects.filter(algorithm_type=old).update(algorithm_type=new)


def values_to_keys(apps: Any, schema_editor: Any) -> None:
    """Forward data step: legacy integer values (as text) to algorithm keys."""
    _remap(apps, LEGACY_ALGORITHM_KEYS)


def keys_to_values(apps: Any, schema_editor: Any) -> None:
    """Reverse data step: algorithm keys to legacy integer values (as text).

    Raises:
        ValueError: If a row holds a key without a legacy integer value, so
            the column cannot return to the integer type.
    """
    _remap(apps, LEGACY_ALGORITHM_VALUES)
    for model_name in MODELS:
        model = apps.get_model("abmem", model_name)
        unknown = (
            model.objects.exclude(algorithm_type__in=list(LEGACY_ALGORITHM_KEYS))
            .values_list("algorithm_type", flat=True)
            .distinct()
        )
        if unknown:
            raise ValueError(
                f"{model_name}.algorithm_type holds keys without a legacy value: "
                f"{sorted(unknown)}"
            )


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0016_demand_profile_template"),
    ]

    operations = [
        migrations.AlterField(
            model_name="agent",
            name="algorithm_type",
            field=models.CharField(default="DDPG", max_length=64),
        ),
        migrations.AlterField(
            model_name="hpostudy",
            name="algorithm_type",
            field=models.CharField(max_length=64),
        ),
        migrations.AlterField(
            model_name="algorithmdefault",
            name="algorithm_type",
            field=models.CharField(max_length=64, unique=True),
        ),
        migrations.AlterField(
            model_name="policycheckpoint",
            name="algorithm_type",
            field=models.CharField(max_length=64),
        ),
        migrations.RunPython(values_to_keys, keys_to_values),
        migrations.AddField(
            model_name="market",
            name="clearing_algorithm",
            field=models.CharField(default="MERIT_ORDER", max_length=64),
        ),
    ]
