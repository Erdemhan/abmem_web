"""Helper endpoints of the scenario wizard (PLAN 2.6).

* ``demand_stats``: min, mean and max demand of periods ``1..N``, used by the
  wizard's capacity-versus-demand warning.
* ``duplicate_scenario``: deep copy of a scenario template ("Duplicate" on
  the templates list).
* ``export_scenario_yaml``, ``import_scenario_preview`` and
  ``import_scenario_confirm``: scenario YAML export and two-phase import
  (PLAN 2.7, ``abmem.scenario_yaml``).
"""

from __future__ import annotations

import functools
import logging

from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_POST

from . import scenario_yaml
from .constants.paths import DATA_DIR
from .models import Simulation
from .services.agent import library_service
from .services.market.demand import DemandDataError, DemandSeries
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
        series = load_demand_series()
    except DemandDataError as exc:
        logger.warning("Demand data could not be read: %s", exc)
        return JsonResponse(
            {"errors": {"__all__": ["The demand data could not be read."]}},
            status=503,
        )
    available = len(series.values)
    try:
        series.validate_horizon(periods)
    except DemandDataError:
        return JsonResponse(
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
    window = series.window(1, periods + 1)
    return JsonResponse(
        {
            "periods": periods,
            "available_periods": available,
            "min": float(window.min()),
            "mean": float(window.mean()),
            "max": float(window.max()),
        }
    )


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
