"""Saving, matching and loading the policies of runs (finding A9).

* :func:`save_run_checkpoints` writes one checkpoint per learning agent when a
  run has FINISHED (not for HPO evaluation runs).
* :func:`resolve_warm_start` matches the agents of a scenario or run to the
  checkpoints of a finished source run: by ``Agent.source`` (the library
  agent), by name when ``source`` is empty.
* :func:`check_warm_start` validates a warm start before a run is created
  (view, HTTP 400 on error). :func:`apply_warm_start` loads the matched
  checkpoints into the policies of a built :class:`SimulationContext`.
* Deleting a :class:`PolicyCheckpoint` row (also through the cascade when its
  run is deleted) deletes its file after the transaction commits.

Loading for a warm start keeps the generators of the new run
(``restore_runtime=False``), so the new run stays reproducible from its own
seed. A run without ``warm_start_from`` is not touched.
"""

from __future__ import annotations

import logging
import pathlib
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import numpy as np
from django.conf import settings
from django.db import transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver

from abmem.models import PolicyCheckpoint, Simulation
from abmem.models.enums import AlgorithmType, SimulationState

if TYPE_CHECKING:
    from abmem.models import Agent
    from abmem.services.algorithms.base import BaseAgentAlgorithm

    from .context import SimulationContext

logger = logging.getLogger(__name__)


class WarmStartError(ValueError):
    """A warm start cannot be used (invalid source, broken file, mismatch)."""


def checkpoint_dir() -> pathlib.Path:
    """Return ``settings.ABMEM_CHECKPOINT_DIR`` as a path.

    Returns:
        The directory of all checkpoint files.
    """
    configured = getattr(settings, "ABMEM_CHECKPOINT_DIR", None)
    if configured:
        return pathlib.Path(configured)
    return pathlib.Path(settings.BASE_DIR) / "checkpoints"


def checkpoint_path(checkpoint: PolicyCheckpoint) -> pathlib.Path:
    """Return the absolute file path of a checkpoint row.

    Args:
        checkpoint: The row.

    Returns:
        ``checkpoint_dir() / checkpoint.path``.
    """
    return checkpoint_dir() / pathlib.PurePosixPath(checkpoint.path)


def _is_baseline(agent: Agent) -> bool:
    """Return True if the agent's algorithm is a non-learning baseline."""
    from abmem.services.algorithms.registry import BASELINE_TYPES

    return AlgorithmType.get(int(agent.algorithm_type)) in BASELINE_TYPES


def _match_key(agent: Agent) -> tuple[str, Any]:
    """Return the key that identifies an agent across runs of one scenario.

    Args:
        agent: Scenario or run agent.

    Returns:
        ``("source", library agent id)`` or, without a source,
        ``("name", name)``.
    """
    if agent.source_id is not None:
        return ("source", agent.source_id)
    return ("name", agent.name)


# Saving ----------------------------------------------------------------------


def save_run_checkpoints(ctx: SimulationContext) -> list[PolicyCheckpoint]:
    """Save the policy of every learning agent of a finished run.

    Nothing is saved for HPO evaluation runs (``HPOTrial.simulation``). The
    files are written first, the rows are created in one transaction. On an
    error every file written by this call is removed again.

    Args:
        ctx: Context of the run that has just FINISHED.

    Returns:
        The created rows in agent order (empty for HPO runs and runs without
        learning agents).

    Raises:
        OSError: If a file cannot be written.
    """
    from abmem.models import HPOTrial
    from abmem.services.algorithms import persistence

    simulation = ctx.simulation
    if HPOTrial.objects.filter(simulation=simulation).exists():
        return []
    learning = [
        rt for rt in ctx.agents if rt.policy.is_learning and not _is_baseline(rt.agent)
    ]
    if not learning:
        return []
    base = checkpoint_dir()
    written: list[pathlib.Path] = []
    rows: list[PolicyCheckpoint] = []
    trained_in_run = max(0, int(ctx.evaluation_start) - 1)
    try:
        for rt in learning:
            agent = rt.agent
            relative = (
                f"run_{simulation.pk}/agent_{agent.pk}_"
                f"{rt.policy.algorithm_type.name.lower()}.pt"
            )
            parent = ctx.warm_start_parents.get(agent.pk)
            periods_trained = (0 if rt.evaluate_only else trained_in_run) + (
                parent.periods_trained if parent is not None else 0
            )
            meta = {
                "simulation_id": simulation.pk,
                "agent_id": agent.pk,
                "agent_name": agent.name,
                "agent_source_id": agent.source_id,
                "seed": int(ctx.seed),
                "periods_trained": periods_trained,
                "parent_checkpoint_id": parent.pk if parent is not None else None,
                "observation_lags": list(ctx.lags),
                "normalizer_mean": [float(v) for v in ctx.normalizer.mean],
                "normalizer_scale": [float(v) for v in ctx.normalizer.scale],
            }
            path = base / relative
            size, digest = persistence.save_checkpoint(path, rt.policy, meta)
            written.append(path)
            rows.append(
                PolicyCheckpoint(
                    simulation=simulation,
                    agent=agent,
                    algorithm_type=rt.policy.algorithm_type,
                    path=relative,
                    file_size=size,
                    sha256=digest,
                    format_version=persistence.FORMAT_VERSION,
                    periods_trained=periods_trained,
                    parent=parent,
                )
            )
        with transaction.atomic():
            for row in rows:
                row.save()
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return rows


# Matching --------------------------------------------------------------------


@dataclass(frozen=True)
class WarmStartPlan:
    """Agents of a target matched to checkpoints of a source run.

    Attributes:
        source: The finished source run.
        matches: Checkpoint per target ``Agent.id``.
        unmatched: Names of learning target agents without a checkpoint.
    """

    source: Simulation
    matches: dict[int, PolicyCheckpoint]
    unmatched: list[str]


def resolve_warm_start(target: Simulation, source: Simulation) -> WarmStartPlan:
    """Match the learning agents of ``target`` to the checkpoints of ``source``.

    Args:
        target: The proxy scenario that is about to be run, or a run cloned
            from it.
        source: Run whose checkpoints initialise the policies.

    Returns:
        The plan. Baseline agents are neither matched nor listed as unmatched.

    Raises:
        WarmStartError: If ``source`` is a scenario, not FINISHED, belongs to
            another scenario, or has no checkpoints.
    """
    scenario_id = target.pk if target.proxy else target.scenario_id
    if source.proxy:
        raise WarmStartError(f"Simulation {source.pk} is a scenario, not a run.")
    if scenario_id is None or source.scenario_id != scenario_id:
        raise WarmStartError(
            f"Run {source.pk} was not started from this scenario."
        )
    if source.state != SimulationState.FINISHED:
        raise WarmStartError(f"Run {source.pk} has not FINISHED.")
    by_key: dict[tuple[str, Any], PolicyCheckpoint] = {}
    for checkpoint in source.checkpoints.select_related("agent").order_by("id"):
        if checkpoint.agent is not None:
            by_key.setdefault(_match_key(checkpoint.agent), checkpoint)
    if not by_key:
        raise WarmStartError(f"Run {source.pk} has no saved policies.")
    matches: dict[int, PolicyCheckpoint] = {}
    unmatched: list[str] = []
    for agent in target.market.agent_set.order_by("id"):
        if _is_baseline(agent):
            continue
        checkpoint = by_key.get(_match_key(agent))
        if checkpoint is None:
            unmatched.append(agent.name)
        else:
            matches[agent.pk] = checkpoint
    return WarmStartPlan(source=source, matches=matches, unmatched=unmatched)


# Loading ---------------------------------------------------------------------


def _load_matches(
    plan: WarmStartPlan,
    policies: Iterable[tuple[Agent, BaseAgentAlgorithm]],
    *,
    freeze: bool,
) -> tuple[dict[int, PolicyCheckpoint], list[str]]:
    """Load the matched checkpoints into ``policies``.

    Args:
        plan: Output of :func:`resolve_warm_start`.
        policies: ``(agent, policy)`` pairs of the target.
        freeze: The loaded policies will be frozen. A policy that does not fit
            its checkpoint is then an error instead of a fresh start.

    Returns:
        Checkpoint per loaded ``Agent.id`` and the warnings (agents that
        start fresh).

    Raises:
        WarmStartError: If a file is missing or corrupted, or (with
            ``freeze``) a checkpoint does not fit its policy.
    """
    from abmem.services.algorithms import persistence

    loaded: dict[int, PolicyCheckpoint] = {}
    warnings = [
        f"Agent '{name}' has no saved policy in run {plan.source.pk} and starts fresh."
        for name in plan.unmatched
    ]
    for agent, policy in policies:
        checkpoint = plan.matches.get(agent.pk)
        if checkpoint is None:
            continue
        try:
            data = persistence.load_checkpoint(
                checkpoint_path(checkpoint),
                expected_sha256=checkpoint.sha256,
                device=policy.device,
            )
        except persistence.CheckpointError as error:
            raise WarmStartError(f"Agent '{agent.name}': {error}") from error
        try:
            persistence.load_state_into(policy, data, restore_runtime=False)
        except persistence.CheckpointError as error:
            if freeze:
                raise WarmStartError(
                    f"Agent '{agent.name}' cannot be evaluated with the policy of "
                    f"run {plan.source.pk}: {error}"
                ) from error
            warnings.append(
                f"Agent '{agent.name}' does not fit its saved policy ({error}) "
                "and starts fresh."
            )
            continue
        spec = data.get("spec") or {}
        if (spec.get("price_low"), spec.get("price_high")) != (
            policy.spec.price_low,
            policy.spec.price_high,
        ):
            warnings.append(
                f"Agent '{agent.name}': the bid bounds differ from run "
                f"{plan.source.pk}, so observations and prices are scaled "
                "differently for the loaded policy."
            )
        loaded[agent.pk] = checkpoint
    return loaded, warnings


def check_warm_start(
    scenario: Simulation, source: Simulation, *, freeze: bool
) -> list[str]:
    """Validate a warm start of ``scenario`` from ``source`` before a run exists.

    Every matched policy is built as the run would build it (same algorithm,
    parameters and plants) and its checkpoint is loaded into it, so shape and
    file errors are found here.

    Args:
        scenario: Proxy scenario to run.
        source: Finished run of that scenario.
        freeze: ``Simulation.freeze_policies`` of the new run.

    Returns:
        Warnings: agents that will start fresh.

    Raises:
        WarmStartError: If the warm start cannot be used.
    """
    from abmem.models.models import Plant
    from abmem.services.algorithms import registry

    from .context import agent_params, algorithm_defaults, policy_spec_for

    plan = resolve_warm_start(scenario, source)
    market = scenario.market
    lags = tuple(int(k) for k in (scenario.observation_lags or ()))
    low, high = Decimal(market.lowerBidBound), Decimal(market.upperBidBound)
    plants_by_agent: dict[int, list[Plant]] = defaultdict(list)
    for plant in (
        Plant.objects.filter(portfolio__agent__market=market)
        .select_related("resource", "portfolio")
        .order_by("id")
    ):
        plants_by_agent[plant.portfolio.agent_id].append(plant)
    defaults = algorithm_defaults()
    policies: list[tuple[Agent, BaseAgentAlgorithm]] = []
    warnings: list[str] = []
    for agent in market.agent_set.filter(pk__in=plan.matches).order_by("id"):
        algorithm_type = AlgorithmType.get(int(agent.algorithm_type))
        try:
            policy = registry.build_policy(
                algorithm_type,
                policy_spec_for(
                    tuple(plants_by_agent.get(agent.pk, ())), lags, low, high
                ),
                agent_params(agent, algorithm_type, defaults),
                np.random.default_rng(0),
            )
        except (KeyError, TypeError, ValueError) as error:
            if freeze:
                raise WarmStartError(f"Agent '{agent.name}': {error}") from error
            warnings.append(f"Agent '{agent.name}' cannot be built: {error}")
            continue
        policies.append((agent, policy))
    _, load_warnings = _load_matches(plan, policies, freeze=freeze)
    return load_warnings + warnings


def apply_warm_start(ctx: SimulationContext) -> None:
    """Load the warm-start policies into a built context.

    Does nothing when ``ctx.simulation.warm_start_from`` is NULL. With
    ``freeze_policies`` every loaded policy is frozen and its agent runs in the
    evaluation phase in every period.

    Args:
        ctx: Context built by ``SimulationContext.build``.

    Raises:
        WarmStartError: If the warm start cannot be used.
    """
    simulation = ctx.simulation
    if simulation.warm_start_from_id is None:
        return
    freeze = bool(simulation.freeze_policies)
    plan = resolve_warm_start(simulation, simulation.warm_start_from)
    loaded, warnings = _load_matches(
        plan, [(rt.agent, rt.policy) for rt in ctx.agents], freeze=freeze
    )
    for rt in ctx.agents:
        if rt.agent.pk in loaded and freeze:
            rt.policy.set_frozen(True)
            rt.evaluate_only = True
    ctx.warm_start_parents = loaded
    ctx.warm_start_warnings = warnings
    for warning in warnings:
        logger.warning("Run %s: %s", simulation.pk, warning)


# Deletion --------------------------------------------------------------------


def _remove_file(path: pathlib.Path) -> None:
    """Delete a checkpoint file and its run directory once it is empty."""
    try:
        path.unlink(missing_ok=True)
        if path.parent != checkpoint_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    except OSError:
        logger.exception("Checkpoint file %s could not be deleted", path)


@receiver(post_delete, sender=PolicyCheckpoint)
def _delete_checkpoint_file(
    sender: type[PolicyCheckpoint], instance: PolicyCheckpoint, **kwargs: Any
) -> None:
    """Delete the file of a deleted row after the transaction commits."""
    path = checkpoint_path(instance)
    transaction.on_commit(lambda: _remove_file(path))
