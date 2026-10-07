"""Simulation model (run or proxy scenario)."""

from typing import Any

from django.db import models
from django.db.models import CheckConstraint, F, Q
from django_enumfield import enum

from .base import Base
from .enums import PeriodType, SimulationMode, SimulationState

DEFAULT_SEED: int = 17081999
DEFAULT_OBSERVATION_LAGS: list[int] = [1, 24]
ALLOWED_OBSERVATION_LAGS: tuple[list[int], ...] = ([1, 24], [1, 24, 168])


def default_observation_lags() -> list[int]:
    """Return a fresh copy of the default observation lag set (OQ-1).

    Returns:
        The list ``[1, 24]``.
    """
    return list(DEFAULT_OBSERVATION_LAGS)


# SIMULATION
class Simulation(Base):
    """A simulation run, or a scenario template when ``proxy`` is True.

    ``seed`` is stored with every run (PLAN 1.10, C9). ``observation_lags`` is
    the lag set of the observation (OQ-1). The last ``evaluation_periods``
    periods run without exploration and learning (OQ-5). ``error`` holds the
    failure message of a FAILED run (OQ-8).

    Policy persistence (finding A9): ``scenario`` is the proxy scenario a run
    was cloned from (NULL for scenarios and older runs). ``warm_start_from``
    is the finished run whose saved policies initialised this run, and
    ``freeze_policies`` runs those loaded policies without learning.
    """

    name = models.CharField(max_length=20, null=False)
    mode = enum.EnumField(SimulationMode, null=False)
    state = enum.EnumField(SimulationState, null=False, default=SimulationState.CREATED)
    periodType = enum.EnumField(PeriodType, null=False)
    periodNumber = models.IntegerField(null=False)
    currentPeriod = models.IntegerField(null=False)
    day = models.IntegerField(null=False)
    seed = models.IntegerField(default=DEFAULT_SEED)
    observation_lags = models.JSONField(default=default_observation_lags)
    evaluation_periods = models.PositiveIntegerField(default=0)
    error = models.TextField(blank=True, default="")
    scenario = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="runs",
    )
    warm_start_from = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="warm_started_runs",
    )
    freeze_policies = models.BooleanField(default=False)

    class Meta:
        constraints = [
            CheckConstraint(
                check=Q(seed__gte=0) & Q(seed__lt=2**32),
                name="simulation_seed_range",
            ),
            CheckConstraint(
                check=Q(observation_lags=ALLOWED_OBSERVATION_LAGS[0])
                | Q(observation_lags=ALLOWED_OBSERVATION_LAGS[1]),
                name="simulation_observation_lags_allowed",
            ),
            CheckConstraint(
                check=Q(evaluation_periods=0)
                | Q(evaluation_periods__lt=F("periodNumber")),
                name="simulation_evaluation_periods_lt_period_number",
            ),
        ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.data = None
