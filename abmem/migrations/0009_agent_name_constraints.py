"""Agent name uniqueness and library role constraints (PLAN 2.3, K4, K5)."""

import django.db.models.functions.text
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0008_agent_library_import"),
    ]

    operations = [
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.UniqueConstraint(
                django.db.models.functions.text.Lower("name"),
                condition=models.Q(market__isnull=True),
                name="uniq_library_agent_name_ci",
            ),
        ),
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.UniqueConstraint(
                fields=("market", "name"), name="uniq_agent_name_per_market"
            ),
        ),
        migrations.AddConstraint(
            model_name="agent",
            constraint=models.CheckConstraint(
                check=models.Q(market__isnull=False) | models.Q(proxy=True),
                name="library_agent_is_proxy",
            ),
        ),
    ]
