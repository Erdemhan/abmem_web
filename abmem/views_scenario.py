"""Helper endpoints of the scenario wizard (PLAN 2.6).

* ``demand_stats``: min, mean and max demand of periods ``1..N`` of the
  demand file.
* ``demand_preview``: the demand series of periods ``1..N`` for the demand
  file or a synthetic demand profile, with night mask and statistics
  (PLAN 2.8). The one preview endpoint of every page with the
  demand-profile component (chart, statistics, capacity warning).
* ``duplicate_scenario``: deep copy of a scenario template ("Duplicate" on
  the templates list).
* ``export_scenario_yaml``, ``import_scenario_preview`` and
  ``import_scenario_confirm``: scenario YAML export and two-phase import
  (PLAN 2.7, ``abmem.scenario_yaml``).
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import scenario_yaml
from .constants.paths import DATA_DIR
from .models import Simulation
from .services.agent import library_service
from .services.market.demand import DemandDataError, DemandSeries
from .services.market.demand_generator import (
    DemandProfile,
    generate_demand,
    night_mask,
    normalize_profile,
)
from .services.simulation.context import DEMAND_COLUMN, MARKET_DATA_FILE

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_demand_series() -> DemandSeries:
    """Read the demand series used by runs once per process.

    Returns:
        The series of ``marketData.xlsx`` (column ``DEMAND``).

    Raises:
        DemandDataError: The file cannot be read.
    """
    return DemandSeries.from_excel(DATA_DIR / MARKET_DATA_FILE, column=DEMAND_COLUMN)


def _parse_periods(raw: str | None) -> int:
    """Validate the ``periods`` query parameter.

    Args:
        raw: Query value.

    Returns:
        The positive number of periods.

    Raises:
        ValidationError: With key ``periods``.
    """
    text = (raw or "").strip()
    if not text.isdigit() or int(text) < 1:
        raise ValidationError(
            {"periods": ["periods must be a positive integer."]}
        )
    return int(text)


@require_GET
def demand_stats(request: HttpRequest) -> JsonResponse:
    """Return demand statistics of periods ``1..periods`` (same rows as a run).

    Args:
        request: GET with ``periods``.

    Returns:
        ``{"periods", "available_periods", "min", "mean", "max"}`` in MWh.
        HTTP 400 ``{"errors": {"periods": [...]}}`` when ``periods`` is not a
        positive integer or the horizon is not covered by valid demand rows,
        HTTP 503 when the demand file cannot be read.
    """
    try:
        periods = _parse_periods(request.GET.get("periods"))
    except ValidationError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    try:
        window, available = _file_window(periods)
    except _DemandError as exc:
        return exc.response
    return JsonResponse(
        {
            "periods": periods,
            "available_periods": available,
            "min": float(window.min()),
            "mean": float(window.mean()),
            "max": float(window.max()),
        }
    )


class _DemandError(Exception):
    """A demand request cannot be served; carries the HTTP error response."""

    def __init__(self, response: JsonResponse) -> None:
        """Store the response.

        Args:
            response: The HTTP 400 or 503 response.
        """
        super().__init__(response.status_code)
        self.response = response


def _file_window(periods: int) -> tuple[np.ndarray, int]:
    """Return the file demand of periods ``1..periods`` (the rows a run uses).

    Args:
        periods: Positive horizon.

    Returns:
        ``(window, available_periods)``.

    Raises:
        _DemandError: HTTP 503 when the file cannot be read, HTTP 400 with
            key ``periods`` when the horizon is not covered by valid rows.
    """
    try:
        series = load_demand_series()
    except DemandDataError as exc:
        logger.warning("Demand data could not be read: %s", exc)
        raise _DemandError(
            JsonResponse(
                {"errors": {"__all__": ["The demand data could not be read."]}},
                status=503,
            )
        ) from exc
    available = len(series.values)
    try:
        series.validate_horizon(periods)
    except DemandDataError as exc:
        raise _DemandError(
            JsonResponse(
                {
                    "errors": {
                        "periods": [
                            f"The demand data covers {available} periods; "
                            f"{periods} were requested."
                        ]
                    },
                    "available_periods": available,
                },
                status=400,
            )
        ) from exc
    return series.window(1, periods + 1), available


# Day hours used to shade a file preview when the request names none.
DEFAULT_DAY_START_HOUR: int = 8
DEFAULT_DAY_END_HOUR: int = 20
# Largest preview horizon (one year of hours plus a margin).
MAX_PREVIEW_PERIODS: int = 10_000


def _series_stats(values: np.ndarray) -> dict[str, float | int] | None:
    """Return count, min, mean, max and population std of a series.

    Args:
        values: Demand values in MWh.

    Returns:
        The statistics, or None for an empty series.
    """
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def _preview_params(request: HttpRequest) -> Mapping[str, Any]:
    """Return the parameters of a preview request.

    Args:
        request: GET (query parameters, ``profile`` as a JSON string) or POST
            (a JSON object).

    Returns:
        Mapping with the optional keys ``periods``, ``source``, ``profile``,
        ``day_start_hour`` and ``day_end_hour``.

    Raises:
        _DemandError: HTTP 400 for a body or ``profile`` that is not JSON.
    """

    def invalid(key: str) -> _DemandError:
        return _DemandError(JsonResponse({"errors": {key: ["Invalid JSON."]}}, status=400))

    if request.method == "POST":
        try:
            data = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            raise invalid("body") from None
        if not isinstance(data, dict):
            raise invalid("body")
        return data
    params: dict[str, Any] = {key: request.GET.get(key) for key in request.GET}
    if params.get("profile") not in (None, ""):
        try:
            params["profile"] = json.loads(params["profile"])
        except json.JSONDecodeError:
            raise invalid("demand_profile") from None
    else:
        params.pop("profile", None)
    return params


def _file_day_hours(params: Mapping[str, Any]) -> tuple[int, int]:
    """Return the day hours that shade a file preview.

    Args:
        params: Preview parameters.

    Returns:
        ``(day_start_hour, day_end_hour)`` of the request when valid, else
        ``(8, 20)``.
    """
    try:
        start = int(str(params.get("day_start_hour")))
        end = int(str(params.get("day_end_hour")))
    except ValueError:
        return DEFAULT_DAY_START_HOUR, DEFAULT_DAY_END_HOUR
    if 0 <= start < end <= 24 and end - start <= 23:
        return start, end
    return DEFAULT_DAY_START_HOUR, DEFAULT_DAY_END_HOUR


@require_http_methods(["GET", "POST"])
def demand_preview(request: HttpRequest) -> JsonResponse:
    """Return the demand of periods ``1..periods`` with night mask and statistics.

    The request (GET query or POST JSON) has ``periods`` and either
    ``source=file`` or a version-1 ``profile``; ``source`` defaults to
    ``generated`` when a profile is given, else ``file``. A file preview is
    shaded with ``day_start_hour``/``day_end_hour`` when given, else
    08:00-20:00 is day. The series is generated exactly like a run
    (``demand_generator``); nothing is written.

    Args:
        request: GET or POST.

    Returns:
        ``{"source", "periods", "values", "night_mask", "day_start_hour",
        "day_end_hour", "stats": {"count", "min", "mean", "max", "std",
        "day": {...} | None, "night": {...} | None}}`` in MWh; a file
        preview also has ``available_periods``. A profile with a
        summer/winter split also has ``season_boundary`` (the last summer
        period) and ``stats.summer`` / ``stats.winter`` over the periods
        whose winter weight is below / at least 0.5. HTTP 400 ``{"errors":
        {...}}`` with keys ``periods``, ``source``, ``body`` or
        ``demand_profile[.<field>]``; HTTP 503 when the demand file cannot be
        read.
    """
    try:
        params = _preview_params(request)
    except _DemandError as exc:
        return exc.response
    errors: dict[str, list[str]] = {}
    periods = 0
    raw_periods = params.get("periods")
    try:
        periods = _parse_periods(None if raw_periods is None else str(raw_periods))
    except ValidationError as exc:
        errors.update(exc.message_dict)
    else:
        if periods > MAX_PREVIEW_PERIODS:
            errors["periods"] = [
                f"A preview covers at most {MAX_PREVIEW_PERIODS} periods."
            ]
    raw_profile = params.get("profile")
    source = params.get("source") or ("generated" if raw_profile is not None else "file")
    profile: DemandProfile | None = None
    if source == "generated":
        if raw_profile is None:
            errors["demand_profile"] = ["A generated preview needs a demand profile."]
        else:
            try:
                profile = DemandProfile.from_dict(normalize_profile(raw_profile))
            except ValidationError as exc:
                errors.update(exc.message_dict)
    elif source != "file":
        errors["source"] = ["source must be 'file' or 'generated'."]
    if errors:
        return JsonResponse({"errors": errors}, status=400)

    body: dict[str, Any] = {"source": source, "periods": periods}
    if profile is not None:
        values = generate_demand(profile, periods)
        start, end = profile.day_start_hour, profile.day_end_hour
        mask = profile.night_mask(periods)
    else:
        try:
            values, available = _file_window(periods)
        except _DemandError as exc:
            return exc.response
        body["available_periods"] = available
        start, end = _file_day_hours(params)
        mask = night_mask(periods, start, end)
    stats: dict[str, Any] = dict(_series_stats(values) or {})
    stats["day"] = _series_stats(values[~mask])
    stats["night"] = _series_stats(values[mask])
    if profile is not None and profile.seasons is not None:
        winter = profile.seasons.winter_weight(periods) >= 0.5
        stats["summer"] = _series_stats(values[~winter])
        stats["winter"] = _series_stats(values[winter])
        body["season_boundary"] = profile.seasons.summer_end_period
    body.update(
        values=[float(value) for value in values],
        night_mask=[bool(flag) for flag in mask],
        day_start_hour=start,
        day_end_hour=end,
        stats=stats,
    )
    return JsonResponse(body)


@require_POST
def duplicate_scenario(request: HttpRequest, simulation_id: int) -> HttpResponse:
    """Copy a scenario template as ``'<name> (copy)'`` and open the copy in the wizard.

    Uses ``library_service.duplicate_scenario`` (simulation and market
    settings, agents with portfolios and plants, ``proxy=True``).

    Args:
        request: POST (CSRF protected).
        simulation_id: Id of the proxy scenario.

    Returns:
        Redirect to the wizard of the copy, or ``{"simulation_id",
        "edit_url"}`` for a JSON client. HTTP 400 for a run (non-proxy
        simulation) or a copy that cannot be saved.
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    try:
        duplicate = library_service.duplicate_scenario(simulation)
    except (ValueError, ValidationError) as exc:
        return JsonResponse(
            {"errors": {"__all__": [f"The scenario could not be copied: {exc}"]}},
            status=400,
        )
    edit_url = reverse("update_simulation", args=[duplicate.id])
    if "application/json" in request.headers.get("Accept", ""):
        return JsonResponse({"simulation_id": duplicate.id, "edit_url": edit_url})
    return redirect(edit_url)


@require_GET
def export_scenario_yaml(request: HttpRequest, simulation_id: int) -> HttpResponse:
    """Download a scenario template as a version-1 YAML file.

    Args:
        request: GET.
        simulation_id: Id of the proxy scenario.

    Returns:
        Attachment ``scenario_<slug>.yaml`` (``application/x-yaml``). HTTP 404
        for an unknown simulation or a run (non-proxy simulation).
    """
    simulation = get_object_or_404(Simulation, id=simulation_id)
    if not simulation.proxy:
        raise Http404("Only scenario templates can be exported.")
    text = scenario_yaml.export_scenario_yaml(simulation)
    slug = slugify(simulation.name) or str(simulation.id)
    response = HttpResponse(
        text.encode("utf-8"),
        content_type=f"{scenario_yaml.CONTENT_TYPE}; charset=utf-8",
    )
    response["Content-Disposition"] = f'attachment; filename="scenario_{slug}.yaml"'
    return response


def _uploaded_document(request: HttpRequest) -> object:
    """Read and parse the ``file`` field of a multipart import request.

    Args:
        request: POST with a ``file`` upload.

    Returns:
        The parsed YAML document.

    Raises:
        ScenarioImportError: No file, too large, not UTF-8 or invalid YAML.
    """
    upload = request.FILES.get("file")
    if upload is None:
        raise scenario_yaml.ScenarioImportError({"file": ["Select a YAML file."]})
    if upload.size is not None and upload.size > scenario_yaml.MAX_FILE_BYTES:
        raise scenario_yaml.ScenarioImportError(
            {"file": [f"The file is larger than {scenario_yaml.MAX_FILE_BYTES // 1024} KB."]}
        )
    return scenario_yaml.load_yaml_bytes(upload.read(scenario_yaml.MAX_FILE_BYTES + 1))


@require_POST
def import_scenario_preview(request: HttpRequest) -> JsonResponse:
    """Validate an uploaded scenario file and show what the import would do.

    Nothing is written.

    Args:
        request: Multipart POST with ``file`` (CSRF protected).

    Returns:
        ``{"scenario", "resources": {"reuse", "create"}, "agents": {"reuse",
        "create"}, "errors": {}}``, or HTTP 400 ``{"errors": {path: [messages]}}``.
    """
    try:
        plan = scenario_yaml.build_import_plan(_uploaded_document(request))
    except scenario_yaml.ScenarioImportError as exc:
        return JsonResponse({"errors": exc.errors}, status=400)
    return JsonResponse(plan.to_json())


@require_POST
def import_scenario_confirm(request: HttpRequest) -> JsonResponse:
    """Validate the uploaded file again and write the import in one transaction.

    The preview is not trusted: the file is parsed and validated from scratch.

    Args:
        request: Multipart POST with ``file`` (CSRF protected).

    Returns:
        ``{"simulation_id", "edit_url"}``, or HTTP 400 ``{"errors": {...}}``
        with nothing written.
    """
    try:
        _, simulation = scenario_yaml.import_scenario(_uploaded_document(request))
    except scenario_yaml.ScenarioImportError as exc:
        return JsonResponse({"errors": exc.errors}, status=400)
    except ValidationError as exc:
        return JsonResponse(
            {"errors": {"__all__": [f"The scenario could not be imported: {m}" for m in exc.messages]}},
            status=400,
        )
    except IntegrityError as exc:
        logger.warning("Scenario import rejected by a database constraint: %s", exc)
        return JsonResponse(
            {"errors": {"__all__": ["The scenario could not be imported because a database constraint rejected it."]}},
            status=400,
        )
    return JsonResponse(
        {
            "simulation_id": simulation.id,
            "edit_url": reverse("update_simulation", args=[simulation.id]),
        }
    )
