"""Hyperparameter optimisation records and per-algorithm defaults (PLAN 2.4, OQ-9, OQ-17)."""

from django.db import models
from django.db.models import CheckConstraint, Q
from django_enumfield import enum

from .base import Base
from .enums import AlgorithmType, HPOPhase, HPOStatus, HPOTrialStatus
from .simulation import Simulation


class HPOStudy(Base):
    """One ABC search for one algorithm on one proxy scenario (ADR-009).

    JSON shapes (``param_bounds``, seed lists, ``objective``) are validated by
    hpo_service. ``heartbeat_at`` is refreshed by the running study process so
    that stale RUNNING studies can be marked FAILED (OQ-17).
    """

    name = models.CharField(max_length=100)
    algorithm_type = enum.EnumField(AlgorithmType)
    scenario = models.ForeignKey(
        Simulation,
        on_delete=models.PROTECT,
        related_name="hpo_studies",
        limit_choices_to={"proxy": True},
    )
    param_bounds = models.JSONField()
    fixed_params = models.JSONField(default=dict)
    colony_size = models.PositiveIntegerField()
    max_iter = models.PositiveIntegerField()
    limit = models.PositiveIntegerField()
    search_seeds = models.JSONField()
    evaluation_seeds = models.JSONField()
    objective = models.JSONField()
    status = enum.EnumField(HPOStatus, default=HPOStatus.CREATED)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    best_params = models.JSONField(null=True, blank=True)
    best_score = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True, default="")
    cancel_requested = models.BooleanField(default=False)

    class Meta:
        constraints = [
            CheckConstraint(
                check=Q(colony_size__gte=2), name="hpostudy_colony_size_gte_2"
            ),
            CheckConstraint(check=Q(max_iter__gte=1), name="hpostudy_max_iter_gte_1"),
            CheckConstraint(check=Q(limit__gte=1), name="hpostudy_limit_gte_1"),
        ]


class HPOTrial(Base):
    """One (candidate, seed) evaluation of a study.

    A candidate is identified by ``(study, iteration, phase, source_index)``.
    ``simulation`` is the evaluation run, so user runs are
    ``Simulation.objects.filter(proxy=False, hpo_trial__isnull=True)``.
    """

    study = models.ForeignKey(HPOStudy, on_delete=models.CASCADE, related_name="trials")
    iteration = models.PositiveIntegerField()
    phase = enum.EnumField(HPOPhase)
    source_index = models.PositiveIntegerField()
    params = models.JSONField()
    seed = models.IntegerField()
    agent_score = models.FloatField(null=True, blank=True)
    market_score = models.FloatField(null=True, blank=True)
    simulation = models.OneToOneField(
        Simulation,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="hpo_trial",
    )
    status = enum.EnumField(HPOTrialStatus, default=HPOTrialStatus.PENDING)
    duration_s = models.FloatField(null=True, blank=True)
    error = models.TextField(blank=True, default="")

    class Meta:
        indexes = [models.Index(fields=["study", "iteration", "phase"])]


class AlgorithmDefault(models.Model):
    """Default hyperparameters of one algorithm type (OQ-9).

    Effective parameters are these defaults, overridden by
    ``Agent.algorithm_params``, overridden by HPO overrides.
    """

    algorithm_type = enum.EnumField(AlgorithmType, unique=True)
    params = models.JSONField(default=dict)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        app_label = "abmem"
