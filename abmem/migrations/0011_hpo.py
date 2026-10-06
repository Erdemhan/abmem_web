"""HPOStudy (with heartbeat_at) and HPOTrial (PLAN 2.4, ADR-009, OQ-17)."""

import django.db.models.deletion
import django_enumfield.db.fields
from django.db import migrations, models

import abmem.models.enums


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0010_simulation_fields_agentsteplog"),
    ]

    operations = [
        migrations.CreateModel(
            name="HPOStudy",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("proxy", models.BooleanField(default=False)),
                ("name", models.CharField(max_length=100)),
                (
                    "algorithm_type",
                    django_enumfield.db.fields.EnumField(
                        enum=abmem.models.enums.AlgorithmType
                    ),
                ),
                ("param_bounds", models.JSONField()),
                ("fixed_params", models.JSONField(default=dict)),
                ("colony_size", models.PositiveIntegerField()),
                ("max_iter", models.PositiveIntegerField()),
                ("limit", models.PositiveIntegerField()),
                ("search_seeds", models.JSONField()),
                ("evaluation_seeds", models.JSONField()),
                ("objective", models.JSONField()),
                (
                    "status",
                    django_enumfield.db.fields.EnumField(
                        default=0, enum=abmem.models.enums.HPOStatus
                    ),
                ),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("heartbeat_at", models.DateTimeField(blank=True, null=True)),
                ("best_params", models.JSONField(blank=True, null=True)),
                ("best_score", models.FloatField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
                ("cancel_requested", models.BooleanField(default=False)),
                (
                    "scenario",
                    models.ForeignKey(
                        limit_choices_to={"proxy": True},
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="hpo_studies",
                        to="abmem.simulation",
                    ),
                ),
            ],
            options={
                "constraints": [
                    models.CheckConstraint(
                        check=models.Q(("colony_size__gte", 2)),
                        name="hpostudy_colony_size_gte_2",
                    ),
                    models.CheckConstraint(
                        check=models.Q(("max_iter__gte", 1)),
                        name="hpostudy_max_iter_gte_1",
                    ),
                    models.CheckConstraint(
                        check=models.Q(("limit__gte", 1)), name="hpostudy_limit_gte_1"
                    ),
                ],
            },
        ),
        migrations.CreateModel(
            name="HPOTrial",
            fields=[
                ("id", models.AutoField(primary_key=True, serialize=False)),
                ("created_at", models.DateTimeField(auto_now_add=True, db_index=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("proxy", models.BooleanField(default=False)),
                ("iteration", models.PositiveIntegerField()),
                (
                    "phase",
                    django_enumfield.db.fields.EnumField(
                        enum=abmem.models.enums.HPOPhase
                    ),
                ),
                ("source_index", models.PositiveIntegerField()),
                ("params", models.JSONField()),
                ("seed", models.IntegerField()),
                ("agent_score", models.FloatField(blank=True, null=True)),
                ("market_score", models.FloatField(blank=True, null=True)),
                (
                    "status",
                    django_enumfield.db.fields.EnumField(
                        default=0, enum=abmem.models.enums.HPOTrialStatus
                    ),
                ),
                ("duration_s", models.FloatField(blank=True, null=True)),
                ("error", models.TextField(blank=True, default="")),
                (
                    "simulation",
                    models.OneToOneField(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="hpo_trial",
                        to="abmem.simulation",
                    ),
                ),
                (
                    "study",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="trials",
                        to="abmem.hpostudy",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["study", "iteration", "phase"],
                        name="abmem_hpotr_study_i_83cc56_idx",
                    )
                ],
            },
        ),
    ]
