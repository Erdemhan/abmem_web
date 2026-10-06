"""Agent model: library agents, scenario agents and run agents (K4, ARCHITECTURE 4.2)."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from django.db import models
from django.db.models import CheckConstraint, Q, UniqueConstraint
from django.db.models.functions import Lower
from django_enumfield import enum

from .base import Base
from .enums import AgentState, AgentType, AlgorithmType
from .market import Market

if TYPE_CHECKING:
    from abmem.services.algorithms.base import BaseAgentAlgorithm


# AGENT
class Agent(Base):
    """Market participant.

    A library agent has ``market`` NULL and ``proxy`` True. A scenario agent
    belongs to the market of a proxy simulation, a run agent to the market of
    a run. ``source`` points to the library agent a copy was made from.

    ``runtime_policy`` holds the in-memory policy object during a run. It is a
    plain class-level default, never a model field, and is never saved.
    """

    name = models.CharField(max_length=20, null=False)
    market = models.ForeignKey(Market, on_delete=models.CASCADE, null=True, blank=True)
    state = enum.EnumField(AgentState, null=True, default=AgentState.CREATED)
    budget = models.DecimalField(decimal_places=2, max_digits=12, null=False, default=0)
    type = enum.EnumField(AgentType, null=True, default=AgentType.FOSSIL)
    algorithm_type = enum.EnumField(AlgorithmType, default=AlgorithmType.DDPG)
    algorithm_params = models.JSONField(default=dict, blank=True)
    failstack_enabled = models.BooleanField(default=False)
    failstack_limit = models.PositiveIntegerField(default=6)
    failstack_delta = models.DecimalField(
        max_digits=8, decimal_places=2, default=Decimal("15")
    )
    failstack_min_margin = models.DecimalField(
        max_digits=6, decimal_places=4, default=Decimal("0")
    )
    source = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="copies",
    )

    runtime_policy: BaseAgentAlgorithm | None = None

    class Meta:
        constraints = [
            UniqueConstraint(
                Lower("name"),
                condition=Q(market__isnull=True),
                name="uniq_library_agent_name_ci",
            ),
            UniqueConstraint(
                fields=["market", "name"], name="uniq_agent_name_per_market"
            ),
            CheckConstraint(
                check=Q(market__isnull=False) | Q(proxy=True),
                name="library_agent_is_proxy",
            ),
            CheckConstraint(
                check=Q(failstack_limit__gte=1), name="failstack_limit_gte_1"
            ),
            CheckConstraint(
                check=Q(failstack_delta__gte=0), name="failstack_delta_gte_0"
            ),
        ]

    @property
    def is_library(self) -> bool:
        """Return True for a library agent (proxy and without market)."""
        return bool(self.proxy) and self.market_id is None

    @property
    def is_baseline(self) -> bool:
        """Return True when ``algorithm_type`` is a non-learning baseline."""
        from abmem.services.algorithms import registry

        return self.algorithm_type in registry.BASELINE_TYPES

    @property
    def algorithm(self) -> Any:
        """Backwards-compatible alias of ``runtime_policy``.

        Kept until the runtime services (agent_service) are migrated to
        ``runtime_policy`` by the runtime_core package.
        """
        return self.runtime_policy

    @algorithm.setter
    def algorithm(self, value: Any) -> None:
        self.runtime_policy = value
