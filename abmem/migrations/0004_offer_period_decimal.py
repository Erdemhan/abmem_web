"""Store fractional accepted quantities and met demand as decimals (PLAN 1.7, B8)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0003_remove_agent_seed"),
    ]

    operations = [
        migrations.AlterField(
            model_name="offer",
            name="acceptanceAmount",
            field=models.DecimalField(decimal_places=3, max_digits=12, null=True),
        ),
        migrations.AlterField(
            model_name="period",
            name="metDemand",
            field=models.DecimalField(decimal_places=3, max_digits=12),
        ),
    ]
