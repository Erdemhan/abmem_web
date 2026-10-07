"""Demand profile library (PLAN 2.9, spec demand_profile_library)."""

from django.db import models
from django.db.models import UniqueConstraint
from django.db.models.functions import Lower

DEMAND_PROFILE_NAME_MAX_LENGTH: int = 100


class DemandProfileTemplate(models.Model):
    """A named synthetic demand profile of the library.

    ``profile`` is the normalised profile
    (``demand_generator.DemandProfile.to_dict``), settings only. A scenario
    that uses a library profile stores its own copy in
    ``Simulation.demand_profile`` and links back through
    ``Simulation.demand_profile_source``, so editing a library profile never
    changes a scenario. Every write goes through
    ``abmem.services.market.demand_library``.

    Attributes:
        name: Library name, unique case-insensitively (database ``LOWER``).
        description: Optional free text.
        profile: Version-1 profile mapping.
        created_at: Creation time.
        updated_at: Time of the last change.
    """

    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=DEMAND_PROFILE_NAME_MAX_LENGTH)
    description = models.TextField(blank=True, default="")
    profile = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "abmem"
        ordering = ["name"]
        constraints = [
            UniqueConstraint(Lower("name"), name="uniq_demand_profile_name_ci"),
        ]

    def __str__(self) -> str:
        """Return the library name.

        Returns:
            ``name``.
        """
        return self.name
