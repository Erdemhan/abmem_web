"""Evaluation of one (candidate, seed) pair in an HPO pool worker (K7, K8).

``evaluate_task`` is a top-level, picklable function. It clones the study
scenario into an evaluation run, forces the study algorithm on every agent
(K7, target ALL) or tunes only the agents that already use it (target
MATCHING, PLAN 2.15), runs the simulation in-process, scores it from the
database and updates the ``HPOTrial`` row. A failed evaluation marks the trial FAILED and returns
objectives ``(-inf, -inf)``, so the candidate is never selected.

ORM-dependent modules are imported inside the functions so that spawned
processes can import this module before Django is configured.
"""

from __future__ import annotations

import logging
import math
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any, Final

logger = logging.getLogger(__name__)

FAILSTACK_SEARCH_KEYS: Final[frozenset[str]] = frozenset(
    {"failstack_limit", "failstack_delta"}
)
"""Searchable parameters that are agent fields, not algorithm parameters."""
FAILSTACK_FIXED_KEYS: Final[frozenset[str]] = frozenset(
    {"failstack_enabled", "failstack_limit", "failstack_delta"}
)
"""Keys allowed in ``HPOStudy.fixed_params``."""
ERROR_TAIL: Final[int] = 4000
FAILSTACK_DELTA_QUANTUM: Final[Decimal] = Decimal("0.01")


@dataclass(frozen=True)
class EvaluationTask:
    """One (candidate, seed) evaluation.

    Attributes:
        study_id: Primary key of the study.
        trial_id: Primary key of the PENDING ``HPOTrial`` row.
        seed: Run seed.
        params: Decoded candidate parameters (searched values only).
    """

    study_id: int
    trial_id: int
    seed: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialOutcome:
    """Result of :func:`evaluate_task`.

    Attributes:
        trial_id: Primary key of the trial.
        ok: True if the run finished and was scored.
        agent_score: Agent score, ``-inf`` on failure.
        market_score: Market score, ``-inf`` on failure.
        error: Error text on failure.
    """

    trial_id: int
    ok: bool
    agent_score: float = -math.inf
    market_score: float = -math.inf
    error: str = ""

    @property
    def objectives(self) -> tuple[float, float]:
        """Return ``(agent_score, market_score)``, ``(-inf, -inf)`` on failure."""
        if not self.ok:
            return (-math.inf, -math.inf)
        return (self.agent_score, self.market_score)


def split_params(
    params: Mapping[str, Any], fixed_params: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split candidate parameters into algorithm overrides and agent fields.

    Args:
        params: Decoded candidate (searched parameters).
        fixed_params: ``HPOStudy.fixed_params``.

    Returns:
        ``(algorithm_overrides, fixed_agent_fields)``. ``failstack_enabled``
        is always set (False when absent). Fixed fail-stack values apply only
        when the rule is enabled and are overridden by searched values.
    """
    overrides = {k: v for k, v in params.items() if k not in FAILSTACK_SEARCH_KEYS}
    enabled = bool(fixed_params.get("failstack_enabled", False))
    agent_fields: dict[str, Any] = {"failstack_enabled": enabled}
    if enabled:
        for key in ("failstack_limit", "failstack_delta"):
            if key in params:
                agent_fields[key] = params[key]
            elif key in fixed_params:
                agent_fields[key] = fixed_params[key]
    if "failstack_limit" in agent_fields:
        agent_fields["failstack_limit"] = int(agent_fields["failstack_limit"])
    if "failstack_delta" in agent_fields:
        agent_fields["failstack_delta"] = Decimal(
            str(agent_fields["failstack_delta"])
        ).quantize(FAILSTACK_DELTA_QUANTUM, rounding=ROUND_HALF_EVEN)
    return overrides, agent_fields


def target_algorithms(target: str, algorithm: str) -> tuple[str | None, str | None]:
    """Return the algorithm arguments of an evaluation run for a study target.

    Args:
        target: ``HPOStudy.target`` (``"ALL"`` or ``"MATCHING"``).
        algorithm: Study algorithm key.

    Returns:
        ``(forced_algorithm, override_algorithm)``. ALL forces the study
        algorithm on every agent and applies the overrides to all of them
        (``(algorithm, None)``). MATCHING forces nothing and restricts the
        overrides to the agents of the study algorithm (``(None, algorithm)``).

    Raises:
        ValueError: If ``target`` is not a known target.
    """
    if str(target) == "ALL":
        return str(algorithm), None
    if str(target) == "MATCHING":
        return None, str(algorithm)
    raise ValueError(f"unknown HPO target {target!r}, expected 'ALL' or 'MATCHING'")


def _touch_heartbeat(study_id: int) -> None:
    """Set ``heartbeat_at`` of the study to now (OQ-17)."""
    from django.utils import timezone

    from abmem.models import HPOStudy

    HPOStudy.objects.filter(pk=study_id).update(heartbeat_at=timezone.now())


def evaluate_task(task: EvaluationTask) -> TrialOutcome:
    """Run and score one evaluation and update its ``HPOTrial`` row.

    Target ALL: the run agents get the study algorithm (K7), the candidate
    as algorithm overrides and the fail-stack settings from ``split_params``.
    A scenario agent's own ``algorithm_params`` belong to its own algorithm,
    so they are dropped when that algorithm differs from the study algorithm
    (OQ-9: agent parameters apply only to the agent's own algorithm type).

    Target MATCHING (PLAN 2.15): no algorithm is forced. Only the agents
    whose algorithm is the study algorithm get the overrides and the
    fail-stack settings, and the agent score is the mean over these agents.
    The other agents run unchanged.

    Args:
        task: The evaluation.

    Returns:
        The outcome. Never raises for errors of the run or the scoring.
    """
    from abmem.models import Agent, HPOStudy, HPOTrial
    from abmem.models.enums import HPOTrialStatus
    from abmem.services.agent import library_service
    from abmem.services.hpo.objective import score_simulation
    from abmem.services.simulation import simulation_service

    started = time.perf_counter()
    HPOTrial.objects.filter(pk=task.trial_id).update(status=HPOTrialStatus.RUNNING)
    try:
        study = HPOStudy.objects.select_related("scenario").get(pk=task.study_id)
        overrides, agent_fields = split_params(task.params, study.fixed_params or {})
        forced, only = target_algorithms(study.target, study.algorithm_type)
        simulation = library_service.clone_scenario_for_run(
            study.scenario,
            seed=task.seed,
            forced_algorithm=forced,
            algorithm_overrides=overrides,
            fixed_agent_fields=agent_fields,
            override_algorithm=only,
        )
        HPOTrial.objects.filter(pk=task.trial_id).update(simulation=simulation)
        foreign = [
            name
            for name, algorithm in study.scenario.market.agent_set.values_list(
                "name", "algorithm_type"
            )
            if only is None and str(algorithm) != str(study.algorithm_type)
        ]
        for agent in Agent.objects.filter(market=simulation.market, name__in=foreign):
            agent.algorithm_params = dict(overrides)
            agent.save(update_fields=["algorithm_params", "updated_at"])
        simulation_service.run(
            simulation,
            algorithm_overrides=overrides,
            forced_algorithm=forced,
            override_algorithm=only,
            export_json=False,
        )
        if only is None:
            scores = score_simulation(simulation)
        else:
            tuned = Agent.objects.filter(
                market=simulation.market, algorithm_type=only
            ).values_list("pk", flat=True)
            scores = score_simulation(simulation, agent_ids=list(tuned))
        if not (
            math.isfinite(scores.agent_score) and math.isfinite(scores.market_score)
        ):
            raise ValueError(f"non-finite scores {scores}")
    except Exception:
        error = traceback.format_exc()[-ERROR_TAIL:]
        logger.warning("HPO trial %s failed:\n%s", task.trial_id, error)
        HPOTrial.objects.filter(pk=task.trial_id).update(
            status=HPOTrialStatus.FAILED,
            error=error,
            duration_s=time.perf_counter() - started,
        )
        _touch_heartbeat(task.study_id)
        return TrialOutcome(trial_id=task.trial_id, ok=False, error=error)

    HPOTrial.objects.filter(pk=task.trial_id).update(
        status=HPOTrialStatus.FINISHED,
        agent_score=scores.agent_score,
        market_score=scores.market_score,
        duration_s=time.perf_counter() - started,
    )
    _touch_heartbeat(task.study_id)
    return TrialOutcome(
        trial_id=task.trial_id,
        ok=True,
        agent_score=scores.agent_score,
        market_score=scores.market_score,
    )
