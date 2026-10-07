"""Per-step training diagnostics of learning agents (PLAN 1.2, A9)."""

from django.db import models
from django.db.models import UniqueConstraint

from .agent import Agent
from .models import Period


class AgentStepLog(models.Model):
    """One row per agent step.

    ``period`` is the decision period t, ``reward`` refers to the outcome of
    period t-1. A plain ``models.Model`` (no ``Base``) keeps rows small. Float
    fields must be finite, which the service layer (step_logger) enforces.
    """

    agent = models.ForeignKey(Agent, on_delete=models.CASCADE, related_name="step_logs")
    period = models.ForeignKey(
        Period, on_delete=models.CASCADE, related_name="step_logs"
    )
    reward = models.FloatField(null=True)
    critic_loss = models.FloatField(null=True)
    actor_loss = models.FloatField(null=True)
    mean_q = models.FloatField(null=True)
    exploration_scale = models.FloatField(null=True)
    intervened_plants = models.JSONField(default=list)
    learning_active = models.BooleanField(default=False)
    evaluation_phase = models.BooleanField(default=False)
    extra = models.JSONField(default=dict)

    class Meta:
        app_label = "abmem"
        constraints = [
            UniqueConstraint(
                fields=["agent", "period"], name="uniq_steplog_agent_period"
            )
        ]
        indexes = [models.Index(fields=["agent", "period"])]
