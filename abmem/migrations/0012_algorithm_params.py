"""Agent.algorithm_params and AlgorithmDefault (OQ-9)."""

import django_enumfield.db.fields
from django.db import migrations, models

import abmem.models.enums


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0011_hpo"),
    ]

    operations = [
        migrations.AddField(
            model_name="agent",
            name="algorithm_params",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.CreateModel(
            name="AlgorithmDefault",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "algorithm_type",
                    django_enumfield.db.fields.EnumField(
                        enum=abmem.models.enums.AlgorithmType, unique=True
                    ),
                ),
                ("params", models.JSONField(default=dict)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
        ),
    ]
