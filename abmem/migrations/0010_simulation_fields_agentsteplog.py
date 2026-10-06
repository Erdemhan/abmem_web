"""Simulation run settings, AgentStepLog and decimal demand.

PLAN 1.2, 1.10, C9, OQ-1, OQ-5, OQ-8, B12.
"""

import django.db.models.deletion
from django.db import migrations, models

import abmem.models.simulation


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0009_agent_name_constraints"),
    ]

    operations = [
        migrations.AddField(
            model_name="simulation",
            name="seed",
            field=models.IntegerField(default=17081999),
        ),
        migrations.AddField(
            model_name="simulation",
            name="observation_lags",
            field=models.JSONField(
                default=abmem.models.simulation.default_observation_lags
            ),
        ),
        migrations.AddField(
            model_name="simulation",
            name="evaluation_periods",
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="simulation",
            name="error",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AddConstraint(
            model_name="simulation",
            constraint=models.CheckConstraint(
                check=models.Q(("seed__gte", 0), ("seed__lt", 4294967296)),
                name="simulation_seed_range",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulation",
            constraint=models.CheckConstraint(
                check=models.Q(
                    ("observation_lags", [1, 24]),
                    ("observation_lags", [1, 24, 168]),
                    _connector="OR",
                ),
                name="simulation_observation_lags_allowed",
            ),
        ),
        migrations.AddConstraint(
            model_name="simulation",
            constraint=models.CheckConstraint(
                check=models.Q(
                    ("evaluation_periods", 0),
                    ("evaluation_periods__lt", models.F("periodNumber")),
                    _connector="OR",
                ),
                name="simulation_evaluation_periods_lt_period_number",
            ),
        ),
        migrations.AlterField(
            model_name="period",
            name="demand",
            field=models.DecimalField(decimal_places=3, max_digits=12),
        ),
        migrations.CreateModel(
            name="AgentStepLog",
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
                ("reward", models.FloatField(null=True)),
                ("critic_loss", models.FloatField(null=True)),
                ("actor_loss", models.FloatField(null=True)),
                ("mean_q", models.FloatField(null=True)),
                ("exploration_scale", models.FloatField(null=True)),
                ("intervened_plants", models.JSONField(default=list)),
                ("learning_active", models.BooleanField(default=False)),
                ("evaluation_phase", models.BooleanField(default=False)),
                ("extra", models.JSONField(default=dict)),
                (
                    "agent",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="step_logs",
                        to="abmem.agent",
                    ),
                ),
                (
                    "period",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="step_logs",
                        to="abmem.period",
                    ),
                ),
            ],
            options={
                "indexes": [
                    models.Index(
                        fields=["agent", "period"],
                        name="abmem_agent_agent_i_22a576_idx",
                    )
                ],
                "constraints": [
                    models.UniqueConstraint(
                        fields=("agent", "period"), name="uniq_steplog_agent_period"
                    )
                ],
            },
        ),
    ]
