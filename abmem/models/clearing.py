"""Global default parameters of market clearing algorithms (PLAN 2.12)."""

from __future__ import annotations

from django.db import models


class ClearingDefault(models.Model):
    """Default parameters of one clearing algorithm key.

    Effective parameters of a run are the class defaults of the algorithm's
    ``config_class``, overridden by these defaults, overridden by
    ``Market.clearing_params`` of the scenario. Edited on the Algorithms
    page.
    """

    algorithm_key = models.CharField(max_length=64, unique=True)
    params = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "abmem"

    def __str__(self) -> str:
        """Return the algorithm key."""
        return self.algorithm_key
