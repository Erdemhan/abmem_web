"""Saved policy of one run agent (finding A9)."""

from django.db import models
from django_enumfield import enum

from .agent import Agent
from .base import Base
from .enums import AlgorithmType
from .simulation import Simulation


class PolicyCheckpoint(Base):
    """Lookup row of one checkpoint file (``services/algorithms/persistence.py``).

    The learned state lives in a file under ``settings.ABMEM_CHECKPOINT_DIR``,
    never in the database. ``path`` is relative to that directory. The file is
    deleted when the row is deleted (``services/simulation/checkpoints.py``),
    which happens when its run is deleted.

    Attributes:
        simulation: Run that produced the checkpoint.
        agent: Run agent whose policy was saved.
        algorithm_type: Algorithm of the saved policy.
        path: File path relative to ``ABMEM_CHECKPOINT_DIR`` (POSIX separators).
        file_size: Size of the file in bytes.
        sha256: Hex SHA-256 digest of the file, checked before loading.
        format_version: ``persistence.FORMAT_VERSION`` of the file.
        periods_trained: Periods with learning of this policy, including the
            periods of every warm-start ancestor.
        parent: Checkpoint the policy was warm-started from, if any.
    """

    simulation = models.ForeignKey(
        Simulation, on_delete=models.CASCADE, related_name="checkpoints"
    )
    agent = models.ForeignKey(
        Agent,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="checkpoints",
    )
    algorithm_type = enum.EnumField(AlgorithmType)
    path = models.CharField(max_length=500)
    file_size = models.BigIntegerField()
    sha256 = models.CharField(max_length=64)
    format_version = models.PositiveIntegerField()
    periods_trained = models.PositiveIntegerField(default=0)
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["simulation", "agent"], name="uniq_checkpoint_per_run_agent"
            )
        ]
