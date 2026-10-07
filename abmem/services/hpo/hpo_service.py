"""DB-backed hyperparameter search service (PLAN 1.11, 2.4, K7, K8, K9).

Process model (ARCHITECTURE.md 7.2): ``start`` spawns a separate process
that calls ``run_study``. ``run_study`` runs the batch-synchronous ABC of
:mod:`abmem.services.hpo.abc_optimizer` with an evaluation callback that
expands every candidate into one ``HPOTrial`` per search seed, evaluates the
trials in a spawn ``Pool`` (``maxtasksperchild=1``) and returns the mean
objectives over the search seeds. After the search the best candidate is
re-evaluated on the held-out evaluation seeds (phase ``FINAL_EVAL``) and that
value is reported (1.11, C9). Status, heartbeat, cancel flag and trial rows
live in the database. Retention follows K9.

The objective is the weighted single objective
``w1 * agent_score + w2 * market_score`` with per-study weights (OQ-11).
The ABC random generator is seeded from the study's search seeds
(``numpy.random.SeedSequence(search_seeds)``), so a study is reproducible
from its stored settings when the evaluations are deterministic.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import multiprocessing
import os
import traceback
import typing
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connections
from django.utils import timezone

from abmem import bootstrap
from abmem.models import HPOStudy, HPOTrial, Simulation
from abmem.models.enums import AlgorithmType, HPOPhase, HPOStatus, HPOTrialStatus
from abmem.services.algorithms.algorithm_config import CONFIG_CLASSES, AlgorithmConfig
from abmem.services.hpo.abc_optimizer import (
    ABCSettings,
    ArtificialBeeColony,
    CandidateRequest,
    CandidateScore,
    weighted_score,
)
from abmem.services.hpo.param_space import ParamSpace, ParamSpec
from abmem.services.hpo.retention import apply_retention
from abmem.services.hpo.worker import (
    ERROR_TAIL,
    FAILSTACK_FIXED_KEYS,
    FAILSTACK_SEARCH_KEYS,
    EvaluationTask,
    TrialOutcome,
    evaluate_task,
)

if TYPE_CHECKING:
    from multiprocessing.pool import Pool

logger = logging.getLogger(__name__)

SEED_UPPER: Final[int] = 2**31
"""Exclusive upper seed bound. ``HPOTrial.seed`` and ``Simulation.seed`` are
4-byte signed integer columns, so seeds must lie in ``[0, 2**31)``."""
DEFAULT_HEARTBEAT_TIMEOUT: Final[timedelta] = timedelta(minutes=30)
POOL_POLL_S: Final[float] = 2.0
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (1.0, 0.0)
CANCELED_MESSAGE: Final[str] = "canceled"

TaskRunner = Callable[
    [Sequence[EvaluationTask], Callable[[], bool]], list[TrialOutcome]
]
"""Evaluates a batch of tasks in order. The second argument returns True when
the study was canceled, then the runner raises :class:`StudyCanceled`."""


class StudyCanceled(Exception):
    """Raised inside ``run_study`` when ``cancel_requested`` is set."""


class StudyFailed(RuntimeError):
    """Raised inside ``run_study`` when the study cannot continue."""


# ---------------------------------------------------------------- validation
def _int_field_names(config_cls: type[AlgorithmConfig]) -> set[str]:
    """Names of the ``int`` fields of a config class."""
    hints = typing.get_type_hints(config_cls)
    return {f.name for f in dataclasses.fields(config_cls) if hints[f.name] is int}


def _searchable_field_names(config_cls: type[AlgorithmConfig]) -> set[str]:
    """Names of the scalar ``int`` or ``float`` fields of a config class."""
    hints = typing.get_type_hints(config_cls)
    return {
        f.name for f in dataclasses.fields(config_cls) if hints[f.name] in (int, float)
    }


def _parse_algorithm(value: Any) -> AlgorithmType:
    """Parse an algorithm given as enum, value or member name.

    Raises:
        ValueError: If the value is not an algorithm.
    """
    if isinstance(value, AlgorithmType):
        return value
    if isinstance(value, str) and not value.strip().lstrip("-").isdigit():
        try:
            return AlgorithmType[value.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown algorithm {value!r}") from None
    try:
        return AlgorithmType(int(value))
    except (TypeError, ValueError):
        raise ValueError(f"unknown algorithm {value!r}") from None


def _parse_positive_int(value: Any, name: str, minimum: int) -> int:
    """Parse an integer ``>= minimum``.

    Raises:
        ValueError: If the value is not such an integer.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if isinstance(value, float) and value != number:
        raise ValueError(f"{name} must be an integer")
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _parse_seeds(value: Any, name: str) -> list[int]:
    """Parse a non-empty list of distinct seeds in ``[0, 2**31)``.

    Accepts a sequence or a comma-separated string.

    Raises:
        ValueError: If the list is empty, malformed, has duplicates or a
            seed is out of range.
    """
    if isinstance(value, str):
        items: list[Any] = [s.strip() for s in value.split(",") if s.strip()]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError(f"{name} must be a list of integers")
    if not items:
        raise ValueError(f"{name} must not be empty")
    seeds: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{name}: {item!r} is not an integer")
        try:
            seed = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: {item!r} is not an integer") from None
        if isinstance(item, float) and item != seed:
            raise ValueError(f"{name}: {item!r} is not an integer")
        if not 0 <= seed < SEED_UPPER:
            raise ValueError(f"{name}: seed {seed} outside [0, {SEED_UPPER})")
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must be distinct")
    return seeds


def _parse_bound(name: str, entry: Any) -> ParamSpec:
    """Parse one bound entry ``{low, high, scale}`` or ``(low, high, scale)``.

    Raises:
        ValueError: If the entry is malformed or invalid.
    """
    space = ParamSpace.from_bounds({name: entry})
    return space.specs[0]


def _validate_bounds(
    raw: Any, config_cls: type[AlgorithmConfig], failstack_enabled: bool
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Validate ``param_bounds`` against the config class.

    Every searched config field is checked at both bounds with the defaults of
    the other fields, and all lows and all highs are checked together.

    Returns:
        ``(normalised bounds, errors by parameter name)``.
    """
    errors: dict[str, list[str]] = {}
    if not isinstance(raw, Mapping) or not raw:
        return {}, {"param_bounds": ["At least one parameter must be searched."]}
    allowed = _searchable_field_names(config_cls)
    int_fields = _int_field_names(config_cls) | {"failstack_limit"}
    if failstack_enabled:
        allowed |= FAILSTACK_SEARCH_KEYS
    bounds: dict[str, dict[str, Any]] = {}
    for name, entry in raw.items():
        name = str(name)
        if name not in allowed:
            errors.setdefault(name, []).append(
                f"'{name}' is not a searchable parameter of {config_cls.__name__}"
                + (
                    ""
                    if failstack_enabled
                    else " (fail-stack parameters need fail-stack enabled)"
                )
            )
            continue
        try:
            spec = _parse_bound(name, entry)
        except ValueError as error:
            errors.setdefault(name, []).append(str(error))
            continue
        if name in int_fields and spec.scale != "int":
            errors.setdefault(name, []).append(
                f"'{name}' is an integer, scale must be 'int'"
            )
            continue
        if name not in int_fields and spec.scale == "int":
            errors.setdefault(name, []).append(f"'{name}' is not an integer parameter")
            continue
        if name == "failstack_limit" and spec.low < 1:
            errors.setdefault(name, []).append("failstack_limit low must be >= 1")
            continue
        if name == "failstack_delta" and spec.low < 0:
            errors.setdefault(name, []).append("failstack_delta low must be >= 0")
            continue
        if name not in FAILSTACK_SEARCH_KEYS:
            for value in (spec.from_unit(0.0), spec.from_unit(1.0)):
                try:
                    config_cls.from_dict({name: value})
                except (TypeError, ValueError) as error:
                    errors.setdefault(name, []).append(
                        f"bound {value} is outside the valid range: {error}"
                    )
                    break
        bounds[name] = {"low": spec.low, "high": spec.high, "scale": spec.scale}
    if not errors:
        algo = {k: v for k, v in bounds.items() if k not in FAILSTACK_SEARCH_KEYS}
        if algo:
            space = ParamSpace.from_bounds(algo)
            for corner in (0.0, 1.0):
                try:
                    config_cls.from_dict(space.decode(np.full(space.dim, corner)))
                except (TypeError, ValueError) as error:
                    errors.setdefault("param_bounds", []).append(
                        f"the {'lower' if corner == 0.0 else 'upper'} bounds are "
                        f"invalid together: {error}"
                    )
    return bounds, errors


def _validate_fixed_params(raw: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Validate ``fixed_params`` (fail-stack settings only).

    Returns:
        ``(normalised fixed params, errors by key)``.
    """
    if raw is None:
        return {"failstack_enabled": False}, {}
    if not isinstance(raw, Mapping):
        return {}, {"fixed_params": ["fixed_params must be an object"]}
    errors: dict[str, list[str]] = {}
    unknown = sorted(str(k) for k in raw if k not in FAILSTACK_FIXED_KEYS)
    if unknown:
        errors["fixed_params"] = [f"unknown keys {unknown}"]
    enabled = raw.get("failstack_enabled", False)
    if not isinstance(enabled, bool):
        errors.setdefault("failstack_enabled", []).append("must be true or false")
    fixed: dict[str, Any] = {"failstack_enabled": bool(enabled)}
    if "failstack_limit" in raw:
        try:
            fixed["failstack_limit"] = _parse_positive_int(
                raw["failstack_limit"], "failstack_limit", 1
            )
        except ValueError as error:
            errors.setdefault("failstack_limit", []).append(str(error))
    if "failstack_delta" in raw:
        try:
            delta = float(raw["failstack_delta"])
            if (
                isinstance(raw["failstack_delta"], bool)
                or not math.isfinite(delta)
                or delta < 0
            ):
                raise ValueError
            fixed["failstack_delta"] = delta
        except (TypeError, ValueError):
            errors.setdefault("failstack_delta", []).append(
                "failstack_delta must be a finite number >= 0"
            )
    return fixed, errors


def _validate_objective(raw: Any) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Validate the objective settings (weighted mode only, OQ-11)."""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        return {}, {"objective": ["objective must be an object"]}
    mode = raw.get("mode", "weighted")
    weights = raw.get("weights", DEFAULT_WEIGHTS)
    try:
        abc_settings = ABCSettings(
            colony_size=2, max_iter=1, limit=1, mode=mode, weights=tuple(weights)
        )
    except (TypeError, ValueError) as error:
        return {}, {"objective": [str(error)]}
    return {"mode": "weighted", "weights": list(abc_settings.weights)}, {}


def validate_study_input(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise the input of a new study.

    Args:
        data: Keys ``name``, ``scenario`` (proxy ``Simulation`` or its id,
            ``scenario_id`` is accepted too), ``algorithm_type`` (enum, value
            or name), ``param_bounds`` (``{name: {low, high, scale}}``),
            optional ``fixed_params`` (``failstack_enabled``,
            ``failstack_limit``, ``failstack_delta``), ``colony_size``,
            ``max_iter``, ``limit``, ``search_seeds``, ``evaluation_seeds``
            (lists or comma-separated strings) and optional ``objective``
            (``{"mode": "weighted", "weights": [w_agent, w_market]}``).

    Returns:
        The keyword arguments of ``HPOStudy`` (``scenario`` as instance,
        ``algorithm_type`` as enum).

    Raises:
        ValidationError: With ``message_dict`` keyed by field name.
    """
    errors: dict[str, list[str]] = {}
    out: dict[str, Any] = {}

    name = str(data.get("name") or "").strip()
    max_name = HPOStudy._meta.get_field("name").max_length or 100
    if not name:
        errors["name"] = ["Name is required."]
    elif len(name) > max_name:
        errors["name"] = [f"Name must have at most {max_name} characters."]
    out["name"] = name

    scenario_value = data.get("scenario", data.get("scenario_id"))
    scenario: Simulation | None = None
    if isinstance(scenario_value, Simulation):
        scenario = scenario_value
    elif scenario_value in (None, ""):
        errors["scenario"] = ["Scenario is required."]
    else:
        try:
            scenario = Simulation.objects.filter(pk=int(scenario_value)).first()
        except (TypeError, ValueError):
            scenario = None
        if scenario is None:
            errors["scenario"] = ["Scenario does not exist."]
    if scenario is not None and not scenario.proxy:
        errors["scenario"] = [
            "Scenario must be a proxy simulation (scenario template)."
        ]
        scenario = None
    out["scenario"] = scenario

    algorithm: AlgorithmType | None = None
    try:
        algorithm = _parse_algorithm(data.get("algorithm_type"))
    except ValueError as error:
        errors["algorithm_type"] = [str(error)]
    out["algorithm_type"] = algorithm

    fixed, fixed_errors = _validate_fixed_params(data.get("fixed_params"))
    errors.update(fixed_errors)
    out["fixed_params"] = fixed

    if algorithm is not None:
        bounds, bound_errors = _validate_bounds(
            data.get("param_bounds"),
            CONFIG_CLASSES[algorithm],
            bool(fixed.get("failstack_enabled")),
        )
        if bound_errors:
            errors["param_bounds"] = [
                f"{key}: {msg}" if key != "param_bounds" else msg
                for key, msgs in bound_errors.items()
                for msg in msgs
            ]
        out["param_bounds"] = bounds

    for key, minimum in (("colony_size", 2), ("max_iter", 1), ("limit", 1)):
        try:
            out[key] = _parse_positive_int(data.get(key), key, minimum)
        except ValueError as error:
            errors[key] = [str(error)]

    for key in ("search_seeds", "evaluation_seeds"):
        try:
            out[key] = _parse_seeds(data.get(key), key)
        except ValueError as error:
            errors[key] = [str(error)]
    if "search_seeds" in out and "evaluation_seeds" in out:
        shared = sorted(set(out["search_seeds"]) & set(out["evaluation_seeds"]))
        if shared:
            errors["evaluation_seeds"] = [
                f"search and evaluation seeds must be disjoint, shared: {shared}"
            ]

    objective, objective_errors = _validate_objective(data.get("objective"))
    errors.update(objective_errors)
    out["objective"] = objective

    if errors:
        raise ValidationError(errors)
    return out


def create_study(data: Mapping[str, Any]) -> HPOStudy:
    """Validate the input and create a study in state CREATED.

    Args:
        data: See :func:`validate_study_input`.

    Returns:
        The saved study.

    Raises:
        ValidationError: If the input is invalid.
    """
    return HPOStudy.objects.create(**validate_study_input(data))


def param_space_payload(algorithm_type: AlgorithmType) -> dict[str, Any]:
    """Return defaults and default search bounds for the create form.

    Args:
        algorithm_type: The algorithm.

    Returns:
        ``{"algorithm", "defaults", "bounds", "failstack_defaults"}`` with
        ``bounds`` as ``{name: {low, high, scale}}`` from
        ``AlgorithmConfig.default_search_bounds`` and ``failstack_defaults``
        from the ``Agent`` field defaults.
    """
    from abmem.models import Agent

    algorithm = _parse_algorithm(algorithm_type)
    config_cls = CONFIG_CLASSES[algorithm]
    bounds = {
        name: {"low": low, "high": high, "scale": scale}
        for name, (low, high, scale) in config_cls.default_search_bounds().items()
    }
    return {
        "algorithm": algorithm.name,
        "defaults": config_cls().to_dict(),
        "bounds": bounds,
        "failstack_defaults": {
            "failstack_limit": Agent._meta.get_field("failstack_limit").default,
            "failstack_delta": float(Agent._meta.get_field("failstack_delta").default),
        },
    }


# --------------------------------------------------------------- lifecycle
def start(study: HPOStudy) -> None:
    """Spawn the process that runs ``study`` (K8).

    Args:
        study: Study in state CREATED.

    Raises:
        ValueError: If the study is not in state CREATED.
    """
    if study.status != HPOStatus.CREATED:
        raise ValueError(f"study {study.pk} is {study.status.name}, expected CREATED")
    connections.close_all()
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=bootstrap.run_hpo_process, args=(study.id,))
    process.start()


def request_cancel(study: HPOStudy) -> None:
    """Request cancellation of a study.

    A CREATED study becomes CANCELED at once. A RUNNING study is canceled by
    its process before the next batch or while polling the running batch.

    Args:
        study: The study.
    """
    now = timezone.now()
    HPOStudy.objects.filter(pk=study.pk).update(cancel_requested=True)
    HPOStudy.objects.filter(pk=study.pk, status=HPOStatus.CREATED).update(
        status=HPOStatus.CANCELED, finished_at=now
    )
    study.refresh_from_db()


def heartbeat_timeout() -> timedelta:
    """Return ``settings.ABMEM_HPO_HEARTBEAT_TIMEOUT`` (timedelta or seconds).

    Returns:
        The timeout, 30 minutes when the setting is absent (OQ-17).
    """
    value = getattr(settings, "ABMEM_HPO_HEARTBEAT_TIMEOUT", DEFAULT_HEARTBEAT_TIMEOUT)
    if isinstance(value, timedelta):
        return value
    return timedelta(seconds=float(value))


def fail_stale_studies(now: Any = None) -> int:
    """Set RUNNING studies whose heartbeat is too old to FAILED (OQ-17).

    A study without heartbeat uses ``started_at``, then ``created_at``.

    Args:
        now: Current time, ``timezone.now()`` when None.

    Returns:
        Number of studies set to FAILED.
    """
    now = now or timezone.now()
    timeout = heartbeat_timeout()
    cutoff = now - timeout
    count = 0
    for study in HPOStudy.objects.filter(status=HPOStatus.RUNNING):
        last = study.heartbeat_at or study.started_at or study.created_at
        if last is not None and last < cutoff:
            count += HPOStudy.objects.filter(
                pk=study.pk, status=HPOStatus.RUNNING
            ).update(
                status=HPOStatus.FAILED,
                finished_at=now,
                error=(
                    f"No heartbeat since {last.isoformat()} (timeout {timeout}). "
                    "The study process probably stopped (for example after a "
                    "server restart)."
                ),
            )
    return count


def _touch(study_id: int) -> None:
    """Update ``heartbeat_at``."""
    HPOStudy.objects.filter(pk=study_id).update(heartbeat_at=timezone.now())


def _cancel_requested(study_id: int) -> bool:
    """Read the cancel flag and update the heartbeat."""
    _touch(study_id)
    return bool(
        HPOStudy.objects.filter(pk=study_id)
        .values_list("cancel_requested", flat=True)
        .first()
    )


def pool_runner(
    tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]
) -> list[TrialOutcome]:
    """Evaluate tasks in a spawn ``Pool`` with one task per child (K8, B10).

    The batch is polled every ``POOL_POLL_S`` seconds. On cancel the pool is
    terminated and :class:`StudyCanceled` is raised.

    Args:
        tasks: Tasks of one batch.
        is_canceled: Returns True when the study was canceled.

    Returns:
        Outcomes in task order.

    Raises:
        StudyCanceled: If the study was canceled during the batch.
    """
    if not tasks:
        return []
    connections.close_all()
    context = multiprocessing.get_context("spawn")
    processes = max(1, min(os.cpu_count() or 1, len(tasks)))
    pool: Pool = context.Pool(
        processes=processes,
        initializer=bootstrap.init_hpo_worker,
        maxtasksperchild=1,
    )
    try:
        async_result = pool.map_async(evaluate_task, list(tasks), chunksize=1)
        while True:
            async_result.wait(POOL_POLL_S)
            if async_result.ready():
                break
            if is_canceled():
                pool.terminate()
                raise StudyCanceled()
        outcomes = list(async_result.get())
        pool.close()
        return outcomes
    finally:
        pool.terminate()
        pool.join()


def serial_runner(
    tasks: Sequence[EvaluationTask], is_canceled: Callable[[], bool]
) -> list[TrialOutcome]:
    """Evaluate tasks one after another in the current process.

    Args:
        tasks: Tasks of one batch.
        is_canceled: Checked before every task.

    Returns:
        Outcomes in task order.

    Raises:
        StudyCanceled: If the study was canceled between two tasks.
    """
    outcomes: list[TrialOutcome] = []
    for task in tasks:
        if outcomes and is_canceled():
            raise StudyCanceled()
        outcomes.append(evaluate_task(task))
    return outcomes


class _StudyEvaluator:
    """Evaluation callback of the ABC: one trial per (candidate, seed)."""

    def __init__(self, study: HPOStudy, space: ParamSpace, runner: TaskRunner) -> None:
        """Create the callback.

        Args:
            study: The running study.
            space: Its parameter space.
            runner: Batch task runner.
        """
        self.study = study
        self.space = space
        self.runner = runner
        self.weights = tuple(
            float(w) for w in study.objective.get("weights", DEFAULT_WEIGHTS)
        )
        self.best_score = -math.inf
        self.best_trial_ids: list[int] = []
        self.best_params: dict[str, Any] | None = None
        self.best_source_index = 0

    def run_candidates(
        self,
        candidates: Sequence[tuple[int, HPOPhase, int, dict[str, Any]]],
        seeds: Sequence[int],
    ) -> list[tuple[tuple[float, float], list[int]]]:
        """Create trials, evaluate them and average over seeds.

        Args:
            candidates: ``(iteration, phase, source_index, params)`` each.
            seeds: Seeds every candidate is evaluated on.

        Returns:
            Per candidate the mean objectives (``-inf`` if any seed failed)
            and its trial ids.

        Raises:
            StudyFailed: If every evaluation of the batch failed.
            StudyCanceled: From the runner.
        """
        study_id = self.study.pk
        tasks: list[EvaluationTask] = []
        trial_ids: list[list[int]] = []
        for iteration, phase, source_index, params in candidates:
            ids: list[int] = []
            for seed in seeds:
                trial = HPOTrial.objects.create(
                    study_id=study_id,
                    iteration=iteration,
                    phase=phase,
                    source_index=source_index,
                    params=params,
                    seed=seed,
                    status=HPOTrialStatus.PENDING,
                )
                ids.append(trial.pk)
                tasks.append(
                    EvaluationTask(
                        study_id=study_id, trial_id=trial.pk, seed=seed, params=params
                    )
                )
            trial_ids.append(ids)
        try:
            outcomes = self.runner(tasks, lambda: _cancel_requested(study_id))
        except StudyCanceled:
            HPOTrial.objects.filter(
                study_id=study_id,
                status__in=[HPOTrialStatus.PENDING, HPOTrialStatus.RUNNING],
            ).update(status=HPOTrialStatus.FAILED, error=CANCELED_MESSAGE)
            raise
        _touch(study_id)
        by_trial = {o.trial_id: o for o in outcomes}
        if len(by_trial) != len(tasks) or set(by_trial) != {t.trial_id for t in tasks}:
            raise StudyFailed(
                "task runner returned outcomes that do not match the tasks"
            )
        if not any(o.ok for o in outcomes):
            raise StudyFailed(
                f"all {len(outcomes)} evaluations of the batch failed. Last error:\n"
                + outcomes[-1].error
            )
        results: list[tuple[tuple[float, float], list[int]]] = []
        for ids in trial_ids:
            objs = [by_trial[i].objectives for i in ids]
            if any(not all(math.isfinite(v) for v in o) for o in objs):
                mean: tuple[float, float] = (-math.inf, -math.inf)
            else:
                mean = (
                    float(np.mean([o[0] for o in objs])),
                    float(np.mean([o[1] for o in objs])),
                )
            results.append((mean, ids))
        return results

    def __call__(self, requests: Sequence[CandidateRequest]) -> list[CandidateScore]:
        """Evaluate one ABC batch on all search seeds and apply retention.

        Args:
            requests: Candidates of one phase.

        Returns:
            One score per request, in order.
        """
        candidates = [
            (r.iteration, HPOPhase[r.phase], r.source_index, self.space.decode(r.x))
            for r in requests
        ]
        results = self.run_candidates(candidates, self.study.search_seeds)
        scores: list[CandidateScore] = []
        for request, candidate, (objectives, ids) in zip(requests, candidates, results):
            scalar = weighted_score(objectives, self.weights)
            if scalar > self.best_score:
                self.best_score = scalar
                self.best_trial_ids = ids
                self.best_params = candidate[3]
                self.best_source_index = request.source_index
            scores.append(CandidateScore(request=request, objectives=objectives))
        apply_retention(self.study, None, keep_trial_ids=self.best_trial_ids)
        return scores


def _finish(study_id: int, status: HPOStatus, **fields: Any) -> None:
    """Set a final status with ``finished_at`` and heartbeat."""
    now = timezone.now()
    HPOStudy.objects.filter(pk=study_id).update(
        status=status, finished_at=now, heartbeat_at=now, **fields
    )


def run_study(study_id: int, *, task_runner: TaskRunner | None = None) -> None:
    """Run a study to completion in the current process.

    Transitions CREATED -> RUNNING -> FINISHED, FAILED or CANCELED. A study
    that is not CREATED (already running, canceled before the start) is left
    unchanged.

    Args:
        study_id: Primary key of the study.
        task_runner: Batch runner, :func:`pool_runner` when None. Tests pass
            :func:`serial_runner`.
    """
    runner = task_runner or pool_runner
    now = timezone.now()
    claimed = HPOStudy.objects.filter(pk=study_id, status=HPOStatus.CREATED).update(
        status=HPOStatus.RUNNING, started_at=now, heartbeat_at=now, error=""
    )
    if not claimed:
        logger.warning("HPO study %s is not CREATED, not started", study_id)
        return
    try:
        study = HPOStudy.objects.get(pk=study_id)
        space = ParamSpace.from_bounds(study.param_bounds)
        weights = tuple(
            float(w) for w in study.objective.get("weights", DEFAULT_WEIGHTS)
        )
        abc_settings = ABCSettings(
            colony_size=study.colony_size,
            max_iter=study.max_iter,
            limit=study.limit,
            mode="weighted",
            weights=weights,
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([int(s) for s in study.search_seeds])
        )
        evaluator = _StudyEvaluator(study, space, runner)
        abc = ArtificialBeeColony(
            space,
            abc_settings,
            evaluator,
            rng,
            should_stop=lambda: _cancel_requested(study_id),
        )
        result = abc.run()
        if result.stopped:
            raise StudyCanceled()
        if evaluator.best_params is None or not math.isfinite(evaluator.best_score):
            raise StudyFailed("no candidate was evaluated successfully")
        if _cancel_requested(study_id):
            raise StudyCanceled()

        final_iteration = study.max_iter
        ((final_objectives, final_ids),) = evaluator.run_candidates(
            [
                (
                    final_iteration,
                    HPOPhase.FINAL_EVAL,
                    evaluator.best_source_index,
                    evaluator.best_params,
                )
            ],
            study.evaluation_seeds,
        )
        best_score = weighted_score(final_objectives, weights)
        if not math.isfinite(best_score):
            raise StudyFailed("the final evaluation of the best candidate failed")
        apply_retention(
            study, None, keep_trial_ids=[*evaluator.best_trial_ids, *final_ids]
        )
        _finish(
            study_id,
            HPOStatus.FINISHED,
            best_params=evaluator.best_params,
            best_score=best_score,
        )
    except StudyCanceled:
        HPOTrial.objects.filter(
            study_id=study_id,
            status__in=[HPOTrialStatus.PENDING, HPOTrialStatus.RUNNING],
        ).update(status=HPOTrialStatus.FAILED, error=CANCELED_MESSAGE)
        _finish(study_id, HPOStatus.CANCELED)
    except Exception:
        logger.exception("HPO study %s failed", study_id)
        _finish(study_id, HPOStatus.FAILED, error=traceback.format_exc()[-ERROR_TAIL:])
