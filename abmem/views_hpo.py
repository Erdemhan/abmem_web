"""Views of the hyperparameter search UI (PLAN 2.4, spec hpo.ui, ARCHITECTURE.md 8).

Routes (prefix ``/dev/hpo/``):

* ``""`` list of studies,
* ``create/`` create form, creates and starts a study,
* ``param-space/?algorithm=`` defaults and default search bounds,
* ``<id>/`` watch page, ``<id>/status/`` polled status JSON,
* ``<id>/cancel/`` and ``<id>/apply/`` (POST, CSRF protected).

State lives in ``HPOStudy`` and ``HPOTrial``. The list, watch and status views
first mark stale RUNNING studies as FAILED (OQ-17). Validation errors return
HTTP 400 with ``{"errors": {field: [messages]}}``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any, Final

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Count, Q
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .forms_hpo import (
    HPOApplyForm,
    HPOStudyForm,
    searchable_algorithms,
    supports_failstack,
)
from .models import Agent, AlgorithmDefault, HPOStudy, HPOTarget, HPOTrial, Simulation
from .models.enums import HPOPhase, HPOStatus, HPOTrialStatus
from .services.algorithms import registry as algorithm_registry
from .services.hpo import hpo_service
from .services.hpo.abc_optimizer import weighted_score
from .services.hpo.pareto import pareto_front
from .services.hpo.worker import FAILSTACK_SEARCH_KEYS, split_params

logger = logging.getLogger(__name__)

TRIAL_TABLE_LIMIT: Final[int] = 500
"""Most recent trials sent with every status poll (the scatter uses all)."""
ACTIVE_STATUSES: Final[tuple[HPOStatus, ...]] = (HPOStatus.CREATED, HPOStatus.RUNNING)
DEFAULT_FORM_VALUES: Final[dict[str, Any]] = {
    "colony_size": 5,
    "max_iter": 10,
    "limit": 10,
    "search_seeds": "1, 2, 3",
    "evaluation_seeds": "101, 102",
    "weight_agent": 1.0,
    "weight_market": 0.0,
}
"""Initial values of the create form. Weights follow the service default
``(1, 0)``."""


# ------------------------------------------------------------------ helpers
def _errors_response(
    errors: Mapping[str, Iterable[str]], status: int = 400
) -> JsonResponse:
    """Return ``{"errors": {field: [messages]}}`` with the given status."""
    return JsonResponse(
        {"errors": {key: [str(m) for m in msgs] for key, msgs in errors.items()}},
        status=status,
    )


def _service_errors(exc: ValidationError) -> dict[str, list[str]]:
    """Convert a service ``ValidationError`` to ``{field: [messages]}``."""
    if hasattr(exc, "error_dict"):
        return {key: list(msgs) for key, msgs in exc.message_dict.items()}
    return {"__all__": list(exc.messages)}


def _request_data(request: HttpRequest) -> dict[str, Any]:
    """Read a JSON body or form fields.

    Args:
        request: A POST request.

    Returns:
        The submitted values. Form fields are returned as single values.

    Raises:
        ValueError: If a JSON body cannot be parsed or is not an object.
    """
    if request.content_type == "application/json":
        try:
            data = json.loads(request.body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON body: {error}") from None
        if not isinstance(data, dict):
            raise ValueError("the JSON body must be an object")
        return data
    return {key: request.POST.get(key) for key in request.POST}


def _form_data(data: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare submitted values for :class:`HPOStudyForm`.

    Seed lists sent as JSON arrays become comma-separated strings, the
    objective weights may come as ``objective.weights``.
    """
    out = dict(data)
    for key in ("search_seeds", "evaluation_seeds"):
        if isinstance(out.get(key), (list, tuple)):
            out[key] = ", ".join(str(v) for v in out[key])
    objective = out.get("objective")
    if isinstance(objective, Mapping) and isinstance(objective.get("weights"), list):
        weights = objective["weights"]
        if len(weights) == 2:
            out.setdefault("weight_agent", weights[0])
            out.setdefault("weight_market", weights[1])
    if isinstance(out.get("failstack_enabled"), str):
        out["failstack_enabled"] = out["failstack_enabled"].lower() in (
            "1",
            "true",
            "on",
        )
    return out


def _finite_or_none(value: float | None) -> float | None:
    """Return ``value`` if it is a finite number, else None (JSON safe)."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _iso(value: Any) -> str | None:
    """Return an ISO timestamp or None."""
    return value.isoformat() if value is not None else None


def _status_name(value: Any, enum_cls: type) -> str:
    """Return the member name of an enum field value."""
    member = value if isinstance(value, enum_cls) else enum_cls(int(value))
    return member.name


def _weights(study: HPOStudy) -> tuple[float, float]:
    """Return the objective weights of a study."""
    raw = (study.objective or {}).get("weights", hpo_service.DEFAULT_WEIGHTS)
    return (float(raw[0]), float(raw[1]))


def _total_trials(study: HPOStudy, n_scout_candidates: int) -> int:
    """Return the expected number of trials of a study.

    ``(colony_size + max_iter * 2 * colony_size + scouts) * len(search_seeds)
    + len(evaluation_seeds)``. Scout candidates are counted as they occur.
    """
    candidates = study.colony_size * (1 + 2 * study.max_iter) + n_scout_candidates
    return candidates * len(study.search_seeds) + len(study.evaluation_seeds)


def _candidate_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    """Identify a candidate by iteration, phase, source and parameters.

    Onlookers may pick the same source several times in one iteration, so
    the parameters are part of the key.
    """
    return (
        int(row["iteration"]),
        int(row["phase"]),
        int(row["source_index"]),
        json.dumps(row["params"], sort_keys=True),
    )


def _status_payload(study: HPOStudy) -> dict[str, Any]:
    """Build the JSON of ``study_status`` (spec hpo, public_interfaces).

    The search history and the search best are recomputed from the trial
    rows: the score of a candidate is the weighted mean over its search seeds
    and is defined only when all its trials finished. A candidate with a
    failed trial scores ``-inf`` and is never the best (same rule as the
    service).

    Args:
        study: The study.

    Returns:
        The status payload.
    """
    weights = _weights(study)
    rows = list(
        HPOTrial.objects.filter(study=study)
        .order_by("id")
        .values(
            "id",
            "iteration",
            "phase",
            "source_index",
            "seed",
            "params",
            "agent_score",
            "market_score",
            "status",
            "duration_s",
            "simulation_id",
            "error",
        )
    )
    finished = int(HPOTrialStatus.FINISHED)
    failed = int(HPOTrialStatus.FAILED)
    final_eval = int(HPOPhase.FINAL_EVAL)
    scout = int(HPOPhase.SCOUT)

    candidates: dict[tuple[int, int, int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if int(row["phase"]) != final_eval:
            candidates.setdefault(_candidate_key(row), []).append(row)

    n_seeds = len(study.search_seeds)
    best_by_iteration: dict[int, float] = {}
    search_best: tuple[float, dict[str, Any]] | None = None
    for key, trials in candidates.items():
        if len(trials) < n_seeds or any(int(t["status"]) != finished for t in trials):
            continue
        objectives = [(t["agent_score"], t["market_score"]) for t in trials]
        if any(a is None or m is None for a, m in objectives):
            continue
        mean = (
            sum(a for a, _ in objectives) / len(objectives),
            sum(m for _, m in objectives) / len(objectives),
        )
        score = weighted_score(mean, weights)
        if not math.isfinite(score):
            continue
        iteration = key[0]
        best_by_iteration[iteration] = max(
            best_by_iteration.get(iteration, -math.inf), score
        )
        if search_best is None or score > search_best[0]:
            search_best = (score, trials[0]["params"])
    history: list[list[float]] = []
    running_best = -math.inf
    for iteration in sorted(best_by_iteration):
        running_best = max(running_best, best_by_iteration[iteration])
        history.append([iteration, running_best])

    scored = [
        r
        for r in rows
        if int(r["status"]) == finished
        and _finite_or_none(r["agent_score"]) is not None
        and _finite_or_none(r["market_score"]) is not None
    ]
    search_points = [r for r in scored if int(r["phase"]) != final_eval]
    front = set(
        pareto_front([(r["agent_score"], r["market_score"]) for r in search_points])
        if search_points
        else []
    )
    front_ids = {search_points[i]["id"] for i in front}
    scatter = [
        [
            float(r["agent_score"]),
            float(r["market_score"]),
            _status_name(r["phase"], HPOPhase),
            r["id"] in front_ids,
        ]
        for r in scored
    ]

    n_scouts = len({k for k in candidates if k[1] == scout})
    done = sum(1 for r in rows if int(r["status"]) in (finished, failed))
    total = max(_total_trials(study, n_scouts), done)

    table_rows = rows[-TRIAL_TABLE_LIMIT:][::-1]
    trials = [
        {
            "id": r["id"],
            "iteration": r["iteration"],
            "phase": _status_name(r["phase"], HPOPhase),
            "source_index": r["source_index"],
            "seed": r["seed"],
            "params": r["params"],
            "agent_score": _finite_or_none(r["agent_score"]),
            "market_score": _finite_or_none(r["market_score"]),
            "status": _status_name(r["status"], HPOTrialStatus),
            "duration_s": _finite_or_none(r["duration_s"]),
            "simulation_id": r["simulation_id"],
            "error": r["error"],
        }
        for r in table_rows
    ]
    return {
        "id": study.pk,
        "name": study.name,
        "algorithm": str(study.algorithm_type),
        "target": str(study.target),
        "status": _status_name(study.status, HPOStatus),
        "cancel_requested": bool(study.cancel_requested),
        "created_at": _iso(study.created_at),
        "started_at": _iso(study.started_at),
        "finished_at": _iso(study.finished_at),
        "heartbeat_at": _iso(study.heartbeat_at),
        "progress": {"done": done, "total": total},
        "best_score": _finite_or_none(study.best_score),
        "best_params": study.best_params,
        "search_best": (
            {"score": search_best[0], "params": search_best[1]}
            if search_best is not None
            else None
        ),
        "weights": list(weights),
        "history": history,
        "scatter": scatter,
        "trials": trials,
        "trials_total": len(rows),
        "trials_truncated": len(rows) > TRIAL_TABLE_LIMIT,
        "error": study.error,
    }


def scenario_algorithms(
    scenarios: Iterable[Simulation], searchable: Iterable[str]
) -> dict[str, dict[str, int]]:
    """Return the searchable learning algorithms of each scenario with agent counts.

    Embedded in the create page for target MATCHING (PLAN 2.15).

    Args:
        scenarios: Proxy simulations (scenario templates).
        searchable: Algorithm keys offered by the create form.

    Returns:
        ``{str(scenario id): {algorithm key: number of agents}}`` with the
        keys of ``hpo_service.scenario_algorithm_counts`` that are in
        ``searchable``.
    """
    allowed = set(searchable)
    return {
        str(scenario.pk): {
            key: count
            for key, count in hpo_service.scenario_algorithm_counts(scenario).items()
            if key in allowed
        }
        for scenario in scenarios
    }


def tuned_agent_names(study: HPOStudy) -> list[str]:
    """Return the names of the scenario agents that the study tunes.

    Args:
        study: The study.

    Returns:
        Every scenario agent for target ALL, the agents whose algorithm is
        the study algorithm for target MATCHING, ordered by id.
    """
    agents = Agent.objects.filter(market__simulation=study.scenario_id)
    if study.target == HPOTarget.MATCHING:
        agents = agents.filter(algorithm_type=study.algorithm_type)
    return list(agents.order_by("id").values_list("name", flat=True))


def _mark_start_failed(study: HPOStudy, error: Exception) -> None:
    """Set a study whose process could not be started to FAILED."""
    now = timezone.now()
    HPOStudy.objects.filter(pk=study.pk, status=HPOStatus.CREATED).update(
        status=HPOStatus.FAILED,
        finished_at=now,
        error=f"The study process could not be started: {error!r}"[-4000:],
    )


# -------------------------------------------------------------------- views
@require_GET
def study_list(request: HttpRequest) -> HttpResponse:
    """List all studies, newest first.

    Args:
        request: GET request.

    Returns:
        The list page.
    """
    hpo_service.fail_stale_studies()
    studies = (
        HPOStudy.objects.select_related("scenario")
        .annotate(
            n_trials=Count("trials"),
            n_done=Count(
                "trials",
                filter=Q(
                    trials__status__in=[HPOTrialStatus.FINISHED, HPOTrialStatus.FAILED]
                ),
            ),
        )
        .order_by("-created_at", "-id")
    )
    rows = [
        {
            "study": study,
            "status": _status_name(study.status, HPOStatus),
            "algorithm": str(study.algorithm_type),
            "n_done": study.n_done,
        }
        for study in studies
    ]
    return render(request, "pages/hpo-list.html", {"rows": rows})


@require_http_methods(["GET", "POST"])
def study_create(request: HttpRequest) -> HttpResponse:
    """Show the create form or create and start a study.

    A POST accepts a JSON body (the create page) or form fields with
    ``param_bounds`` as a JSON string. On success the study is created in
    state CREATED and its process is started (K8).

    Args:
        request: GET or POST request.

    Returns:
        GET: the form page. POST: JSON ``{"id", "watch_url"}`` with status
        201, or ``{"errors": ...}`` with status 400, or 500 if the process
        could not be started.
    """
    if request.method == "GET":
        scenarios = list(Simulation.objects.filter(proxy=True).order_by("name", "id"))
        searchable = searchable_algorithms()
        algorithms = [
            {
                "name": key,
                "label": algorithm_registry.label_for(key),
                "failstack": supports_failstack(key),
            }
            for key in searchable
        ]
        selected_scenario = request.GET.get("scenario", "")
        selected_target = request.GET.get("target", "").strip().upper()
        context = {
            "scenarios": scenarios,
            "algorithms": algorithms,
            "algorithm_options": [
                {"name": a["name"], "label": a["label"]} for a in algorithms
            ],
            "scenario_algorithms": scenario_algorithms(scenarios, searchable),
            "selected_target": (
                selected_target
                if selected_target in HPOTarget.values
                else str(HPOTarget.ALL)
            ),
            "initial": DEFAULT_FORM_VALUES,
            "selected_scenario": selected_scenario,
            "selected_algorithm": request.GET.get(
                "algorithm", algorithm_registry.DEFAULT_ALGORITHM
            ),
        }
        return render(request, "pages/hpo-create.html", context)

    try:
        data = _request_data(request)
    except ValueError as error:
        return _errors_response({"__all__": [str(error)]})
    form = HPOStudyForm(_form_data(data))
    if not form.is_valid():
        return _errors_response(form.errors)
    try:
        study = hpo_service.create_study(form.to_service_input())
    except ValidationError as error:
        return _errors_response(_service_errors(error))
    try:
        hpo_service.start(study)
    except Exception as error:  # noqa: BLE001 - reported to the user
        logger.exception("could not start HPO study %s", study.pk)
        _mark_start_failed(study, error)
        return JsonResponse(
            {
                "id": study.pk,
                "watch_url": reverse("hpo_watch", args=[study.pk]),
                "errors": {
                    "__all__": [f"The study process could not be started: {error}"]
                },
            },
            status=500,
        )
    return JsonResponse(
        {"id": study.pk, "watch_url": reverse("hpo_watch", args=[study.pk])},
        status=201,
    )


@require_GET
def param_space(request: HttpRequest) -> JsonResponse:
    """Return defaults and default search bounds of an algorithm.

    Args:
        request: GET request with ``algorithm`` (algorithm key).

    Returns:
        JSON of ``hpo_service.param_space_payload`` plus ``failstack``
        (whether fail-stack applies) and ``integer`` (integer parameters),
        or 400 for an unknown or non-searchable algorithm.
    """
    raw = request.GET.get("algorithm", "")
    names = set(searchable_algorithms())
    algorithm = raw.strip().upper()
    if algorithm not in names:
        return _errors_response(
            {
                "algorithm": [
                    f"Unknown or non-searchable algorithm {raw!r}. Choose one of {sorted(names)}."
                ]
            }
        )
    payload = hpo_service.param_space_payload(algorithm)
    payload["failstack"] = supports_failstack(algorithm)
    payload["integer"] = sorted(
        name for name, bound in payload["bounds"].items() if bound["scale"] == "int"
    )
    return JsonResponse(payload)


@require_GET
def study_watch(request: HttpRequest, study_id: int) -> HttpResponse:
    """Show the watch page of a study.

    Args:
        request: GET request.
        study_id: Primary key of the study.

    Returns:
        The watch page.
    """
    hpo_service.fail_stale_studies()
    study = get_object_or_404(HPOStudy.objects.select_related("scenario"), pk=study_id)
    algorithm = str(study.algorithm_type)
    context = {
        "study": study,
        "status": _status_name(study.status, HPOStatus),
        "algorithm": algorithm,
        "matching": study.target == HPOTarget.MATCHING,
        "tuned_agents": tuned_agent_names(study),
        "weights": _weights(study),
        "bounds": sorted(study.param_bounds.items()),
        "fixed_params": study.fixed_params or {},
        "search_seeds": ", ".join(str(s) for s in study.search_seeds),
        "evaluation_seeds": ", ".join(str(s) for s in study.evaluation_seeds),
        "status_url": reverse("hpo_status", args=[study.pk]),
        "cancel_url": reverse("hpo_cancel", args=[study.pk]),
        "apply_url": reverse("hpo_apply", args=[study.pk]),
    }
    return render(request, "pages/hpo-watch.html", context)


@require_GET
def study_status(request: HttpRequest, study_id: int) -> JsonResponse:
    """Return the status JSON polled by the watch page.

    Args:
        request: GET request.
        study_id: Primary key of the study.

    Returns:
        JSON described in :func:`_status_payload`.
    """
    hpo_service.fail_stale_studies()
    study = get_object_or_404(HPOStudy, pk=study_id)
    return JsonResponse(_status_payload(study))


@require_POST
def study_cancel(request: HttpRequest, study_id: int) -> JsonResponse:
    """Request cancellation of a CREATED or RUNNING study.

    Args:
        request: POST request with a CSRF token.
        study_id: Primary key of the study.

    Returns:
        ``{"status", "cancel_requested"}``, or 400 if the study has already
        ended.
    """
    study = get_object_or_404(HPOStudy, pk=study_id)
    if study.status not in ACTIVE_STATUSES:
        return _errors_response(
            {
                "status": [
                    f"The study is {_status_name(study.status, HPOStatus)}, nothing to cancel."
                ]
            }
        )
    hpo_service.request_cancel(study)
    return JsonResponse(
        {
            "status": _status_name(study.status, HPOStatus),
            "cancel_requested": bool(study.cancel_requested),
        }
    )


def _validated_params(algorithm: str, params: Mapping[str, Any]) -> None:
    """Check that ``params`` form a valid config of the algorithm key ``algorithm``.

    Raises:
        ValueError: If the algorithm is not registered, a key is unknown or a
            value is invalid.
    """
    try:
        algorithm_registry.get_config_class(algorithm).from_dict(dict(params))
    except KeyError as error:
        raise ValueError(str(error.args[0])) from error
    except TypeError as error:
        raise ValueError(str(error)) from error


def _apply_to_default(
    study: HPOStudy, algorithm: str, algo_params: dict[str, Any]
) -> dict[str, Any]:
    """Merge the best parameters into ``AlgorithmDefault`` of the algorithm.

    Raises:
        ValueError: If the merged parameters are invalid.
    """
    row = (
        AlgorithmDefault.objects.select_for_update()
        .filter(algorithm_type=algorithm)
        .first()
    )
    merged = {**(dict(row.params or {}) if row else {}), **algo_params}
    _validated_params(algorithm, merged)
    if row is None:
        AlgorithmDefault.objects.create(algorithm_type=algorithm, params=merged)
    else:
        row.params = merged
        row.save(update_fields=["params", "updated_at"])
    skipped = sorted(k for k in (study.best_params or {}) if k in FAILSTACK_SEARCH_KEYS)
    return {
        "target": "default",
        "params": merged,
        "agents_updated": [],
        "not_applied": skipped,
    }


def _apply_to_scenario(
    study: HPOStudy,
    algorithm: str,
    algo_params: dict[str, Any],
    switch_algorithm: bool,
) -> dict[str, Any]:
    """Write the best parameters to the agents of the study scenario.

    Agents that use the study algorithm get the parameters merged into their
    ``algorithm_params``. With ``switch_algorithm`` the other agents are set
    to the study algorithm with exactly these parameters, as in the
    evaluation runs (K7). For a study with target MATCHING only the agents
    of the study algorithm are updated and ``switch_algorithm`` is ignored
    (PLAN 2.15). The fail-stack settings of the study (fixed and searched)
    are written to every updated agent, as in the evaluation runs.

    Raises:
        ValueError: If no agent can be updated or parameters are invalid.
    """
    if HPOStudy.objects.filter(
        scenario_id=study.scenario_id, status__in=ACTIVE_STATUSES
    ).exists():
        raise ValueError(
            "Another study on this scenario is still running. Apply after it has ended."
        )
    market = getattr(study.scenario, "market", None)
    if market is None:
        raise ValueError("The scenario has no market.")
    _, agent_fields = split_params(study.best_params or {}, study.fixed_params or {})
    matching = study.target == HPOTarget.MATCHING
    if matching:
        switch_algorithm = False
    updated: list[str] = []
    for agent in Agent.objects.select_for_update().filter(market=market).order_by("id"):
        own = str(agent.algorithm_type)
        if own == algorithm:
            params = {**dict(agent.algorithm_params or {}), **algo_params}
        elif switch_algorithm:
            params = dict(algo_params)
            agent.algorithm_type = algorithm
        else:
            continue
        _validated_params(algorithm, params)
        agent.algorithm_params = params
        for key, value in agent_fields.items():
            setattr(agent, key, value)
        agent.save()
        updated.append(agent.name)
    if not updated and matching:
        raise ValueError(f"No agent of the scenario uses {algorithm}.")
    if not updated:
        raise ValueError(
            f"No agent of the scenario uses {algorithm}. Enable 'switch all "
            f"agents to {algorithm}' to apply the parameters."
        )
    return {
        "target": "scenario",
        "params": algo_params,
        "agent_fields": {
            k: str(v) if not isinstance(v, (bool, int, float)) else v
            for k, v in agent_fields.items()
        },
        "agents_updated": updated,
        "not_applied": [],
    }


@require_POST
def study_apply(request: HttpRequest, study_id: int) -> JsonResponse:
    """Apply the best parameters of a FINISHED study (OQ-9).

    Args:
        request: POST request with a CSRF token and ``target`` (``default`` or
            ``scenario``) and optional ``switch_algorithm``.
        study_id: Primary key of the study.

    Returns:
        JSON describing what was written, or 400 with ``errors``.
    """
    study = get_object_or_404(HPOStudy.objects.select_related("scenario"), pk=study_id)
    if study.status != HPOStatus.FINISHED or not study.best_params:
        return _errors_response(
            {"status": ["Only a FINISHED study with best parameters can be applied."]}
        )
    try:
        data = _request_data(request)
    except ValueError as error:
        return _errors_response({"__all__": [str(error)]})
    if isinstance(data.get("switch_algorithm"), str):
        data["switch_algorithm"] = data["switch_algorithm"].lower() in (
            "1",
            "true",
            "on",
        )
    form = HPOApplyForm(data)
    if not form.is_valid():
        return _errors_response(form.errors)
    algorithm = str(study.algorithm_type)
    algo_params = {
        k: v for k, v in study.best_params.items() if k not in FAILSTACK_SEARCH_KEYS
    }
    try:
        with transaction.atomic():
            if form.cleaned_data["target"] == HPOApplyForm.TARGET_DEFAULT:
                result = _apply_to_default(study, algorithm, algo_params)
            else:
                result = _apply_to_scenario(
                    study, algorithm, algo_params, form.cleaned_data["switch_algorithm"]
                )
    except ValueError as error:
        return _errors_response({"target": [str(error)]})
    result["algorithm"] = algorithm
    return JsonResponse(result)
