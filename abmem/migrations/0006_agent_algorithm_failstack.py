"""Agent algorithm selection and fail-stack settings (PLAN 1.5, 2.1, K3)."""

from decimal import Decimal

import django.db.models.deletion
import django_enumfield.db.fields
from django.db import migrations, models

import abmem.models.enums


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0005_merge_wave1"),
    ]

    operations = [
        migrations.AddField(
            model_name="agent",
            name="algorithm_type",
            field=django_enumfield.db.fields.EnumField(
                default=0, enum=abmem.models.enums.AlgorithmType
            ),
        ),
        migrations.AddField(
            model_name="agent",
            name="failstack_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="agent",
            name="failstack_limit",
            field=models.PositiveIntegerField(default=6),
        ),
        migrations.AddField(
            model_name="agent",
            name="failstack_delta",
            field=models.DecimalField(
                decimal_places=2, default=Decimal("15"), max_digits=8
            ),
        ),
        migrations.AddField(
            model_name="agent",
            name="failstack_min_margin",
            field=models.DecimalField(
                decimal_places=4, default=Decimal("0"), max_digits=6
            ),
        ),
        migrations.AlterField(
            model_name="agent",
            name="market",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.CASCADE,
                to="abmem.market",
            ),
        ),
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.CheckConstraint(
                check=models.Q(failstack_limit__gte=1), name="failstack_limit_gte_1"
            ),
        ),
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.CheckConstraint(
                check=models.Q(failstack_delta__gte=0), name="failstack_delta_gte_0"
            ),
        ),
    ]
