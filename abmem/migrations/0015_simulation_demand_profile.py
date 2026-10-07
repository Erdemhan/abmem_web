from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("abmem", "0014_policy_checkpoint"),
    ]

    operations = [
        migrations.AddField(
            model_name="simulation",
            name="demand_profile",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
