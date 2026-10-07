"""Agent.source: the library agent a copy was made from (PLAN 2.3, K4)."""

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0006_agent_algorithm_failstack"),
    ]

    operations = [
        migrations.AddField(
            model_name="agent",
            name="source",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="copies",
                to="abmem.agent",
            ),
        ),
    ]
