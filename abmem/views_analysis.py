"""Views of the run analysis tab (PLAN 2.5, spec run_analysis).

Routes (prefix ``/dev/``):

* ``simulation/<id>/analysis/`` analysis page,
* ``simulation/<id>/analysis/data/`` JSON for the charts and tables,
* ``simulation/<id>/export/<kind>.csv`` CSV export, ``kind`` in
  ``{periods, offers, steps}``.

All views are read-only GET views. An unknown simulation or a scenario
template (proxy simulation) returns 404. A run that is not FINISHED or has no
stored data returns 409 with a message.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views.decorators.http import require_GET

from .models import Simulation
from .services.analysis import exports
from .services.analysis.run_analysis import (
    RunAnalysisUnavailable,
    build_run_analysis,
    unavailable_reason,
)

CSV_EXPORTS: Final[dict[str, Callable[[Simulation], str]]] = {
    "periods": exports.periods_csv,
    "offers": exports.offers_csv,
    "steps": exports.step_logs_csv,
}
"""CSV export kind -> export function."""


def _get_run(sim_id: int) -> Simulation:
    """Return the run ``sim_id`` or raise 404.

    Args:
        sim_id: Primary key.

    Returns:
        The run with its market.

    Raises:
        Http404: Unknown simulation or a scenario template.
    """
    return get_object_or_404(
        Simulation.objects.select_related("market"), pk=sim_id, proxy=False
    )


@require_GET
def simulation_analysis(request: HttpRequest, sim_id: int) -> HttpResponse:
    """Render the analysis page of a run.

    Args:
        request: GET request.
        sim_id: Primary key of the run.

    Returns:
        The page (200), or the page with the reason and status 409 when the
        run has no analysis.
    """
    simulation = _get_run(sim_id)
    reason = unavailable_reason(simulation)
    data_url = reverse("simulation_analysis_data", args=[simulation.pk])
    context = {
        "simulation": simulation,
        "unavailable": reason,
        "page_config": {"data_url": data_url, "simulation_id": simulation.pk},
        "watch_url": reverse(
            "watch_simulation", kwargs={"simulation_id": simulation.pk}
        ),
        "csv_urls": {
            kind: reverse("simulation_export_csv", args=[simulation.pk, kind])
            for kind in CSV_EXPORTS
        },
    }
    return render(
        request,
        "pages/simulation-analysis.html",
        context,
        status=409 if reason else 200,
    )


@require_GET
def simulation_analysis_data(request: HttpRequest, sim_id: int) -> JsonResponse:
    """Return the analysis of a run as JSON.

    Args:
        request: GET request.
        sim_id: Primary key of the run.

    Returns:
        ``RunAnalysis.to_dict()`` (200), or ``{"error": message}`` with 409.
    """
    simulation = _get_run(sim_id)
    try:
        analysis = build_run_analysis(simulation)
    except RunAnalysisUnavailable as exc:
        return JsonResponse({"error": str(exc)}, status=409)
    return JsonResponse(analysis.to_dict())


@require_GET
def simulation_export_csv(request: HttpRequest, sim_id: int, kind: str) -> HttpResponse:
    """Download one CSV export of a run.

    Args:
        request: GET request.
        sim_id: Primary key of the run.
        kind: ``periods``, ``offers`` or ``steps``.

    Returns:
        The CSV as an attachment ``simulation_<id>_<kind>.csv``, or a plain
        text message with 409 when the run has no analysis.

    Raises:
        Http404: Unknown kind, unknown simulation or a scenario template.
    """
    export = CSV_EXPORTS.get(kind)
    if export is None:
        raise Http404(f"Unknown export {kind!r}.")
    simulation = _get_run(sim_id)
    reason = unavailable_reason(simulation)
    if reason:
        return HttpResponse(
            reason, status=409, content_type="text/plain; charset=utf-8"
        )
    response = HttpResponse(export(simulation), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = (
        f'attachment; filename="simulation_{simulation.pk}_{kind}.csv"'
    )
    return response
