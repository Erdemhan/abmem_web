"""Demand profile library pages and endpoints (PLAN 2.9, spec demand_profile_library).

* ``demand_profile_list``: library table with actions and the YAML import.
* ``demand_profile_create`` / ``demand_profile_edit``: editor page (GET) and
  JSON save (POST ``{"name", "description", "profile"}``). The wizard's
  "Save to library" action posts to ``demand_profile_create`` as well.
* ``demand_profile_duplicate``, ``demand_profile_delete``: list actions.
* ``demand_profile_export``, ``demand_profile_import_preview`` and
  ``demand_profile_import_confirm``: profile YAML (``abmem.demand_profile_yaml``)
  with the two-step import of the scenario YAML.
* ``demand_profile_detail_json``: one profile for the wizard.
* ``demand_profile_parse_yaml``: validates an uploaded profile file for the
  wizard editor. Nothing is written.

Every write goes through ``services.market.demand_library``. Profiles are
validated only by ``demand_generator.DemandProfile`` and previews use the
shared endpoint ``demand_preview``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from django.db import IntegrityError
from django.db.models import Count, Q
from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from . import demand_profile_yaml
from .models import DemandProfileTemplate
from .models.demand import DEMAND_PROFILE_NAME_MAX_LENGTH
from .services.market import demand_library
from .services.market.demand_library import DemandLibraryError
from .views_scenario import _uploaded_document

logger = logging.getLogger(__name__)

PREVIEW_PERIOD_CHOICES: tuple[int, ...] = (24, 168, 336, 720)
DEFAULT_PREVIEW_PERIODS: int = 168
_DISTRIBUTION_LABELS: dict[str, str] = {
    "NORMAL": "Normal",
    "UNIFORM": "Uniform",
    "LOGNORMAL": "Log-normal",
}


def _wants_json(request: HttpRequest) -> bool:
    """Return whether the client asked for a JSON reply.

    Args:
        request: The request.

    Returns:
        True when the ``Accept`` header names ``application/json``.
    """
    return "application/json" in request.headers.get("Accept", "")


def _number_text(value: Any) -> str:
    """Format a profile number for the list page.

    Args:
        value: Number of a stored profile.

    Returns:
        The number without a trailing ``.0``.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def segment_summary(segment: Any) -> str:
    """Describe one segment of a stored profile in one line.

    Args:
        segment: Segment mapping of ``DemandProfile.to_dict``.

    Returns:
        For example ``"Normal (mean 900, std 100) on [600, 1200] MWh"``.
    """
    if not isinstance(segment, dict):
        return "-"
    distribution = str(segment.get("distribution", ""))
    label = _DISTRIBUTION_LABELS.get(distribution, distribution or "-")
    bounds = (
        f"[{_number_text(segment.get('min'))}, {_number_text(segment.get('max'))}] MWh"
    )
    if "mean" in segment and "std" in segment:
        return (
            f"{label} (mean {_number_text(segment['mean'])}, "
            f"std {_number_text(segment['std'])}) on {bounds}"
        )
    return f"{label} on {bounds}"


def profile_row(template: DemandProfileTemplate) -> dict[str, Any]:
    """Return the list page row of a library profile.

    Args:
        template: Library profile, optionally annotated with
            ``scenario_count``.

    Returns:
        ``{"template", "day_hours", "day", "night", "scenario_count"}``.
    """
    profile = template.profile if isinstance(template.profile, dict) else {}
    start = profile.get("day_start_hour")
    end = profile.get("day_end_hour")
    day_hours = "-"
    if isinstance(start, int) and isinstance(end, int):
        day_hours = f"{start:02d}:00-{end:02d}:00 ({end - start} h day)"
    count = getattr(template, "scenario_count", None)
    if count is None:
        count = len(demand_library.scenarios_using(template))
    return {
        "template": template,
        "day_hours": day_hours,
        "day": segment_summary(profile.get("day")),
        "night": segment_summary(profile.get("night")),
        "scenario_count": count,
    }


@require_GET
def demand_profile_list(request: HttpRequest) -> HttpResponse:
    """Render the demand profile library.

    Args:
        request: GET.

    Returns:
        The page with one row per profile ordered by name.
    """
    templates = DemandProfileTemplate.objects.annotate(
        scenario_count=Count(
            "simulations", filter=Q(simulations__proxy=True), distinct=True
        )
    ).order_by("name", "id")
    return render(
        request,
        "pages/demand-profile-list.html",
        {"rows": [profile_row(template) for template in templates]},
    )


def _editor_context(template: DemandProfileTemplate | None) -> dict[str, Any]:
    """Return the context of the create and edit page.

    Args:
        template: The edited profile, or None on create.

    Returns:
        Context with ``template``, ``profile`` (None on create, the page
        uses the component defaults), ``editor_config`` (JSON for the page
        script) and the preview period choices.
    """
    config = {
        "save_url": (
            reverse("demand_profile_edit", args=[template.id])
            if template is not None
            else reverse("demand_profile_create")
        ),
        "list_url": reverse("demand_profile_list"),
        "demand_preview_url": reverse("demand_preview"),
        "is_update": template is not None,
        "name_max_length": DEMAND_PROFILE_NAME_MAX_LENGTH,
    }
    return {
        "template": template,
        "profile": template.profile if template is not None else None,
        "editor_config": config,
        "period_choices": PREVIEW_PERIOD_CHOICES,
        "default_periods": DEFAULT_PREVIEW_PERIODS,
        "name_max_length": DEMAND_PROFILE_NAME_MAX_LENGTH,
    }


def _json_body(request: HttpRequest) -> dict[str, Any] | JsonResponse:
    """Decode the JSON object of a POST.

    Args:
        request: POST with a JSON body.

    Returns:
        The object, or the HTTP 400 response for invalid JSON.
    """
    try:
        data = json.loads(request.body or b"{}")
    except json.JSONDecodeError:
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)
    if not isinstance(data, dict):
        return JsonResponse({"errors": {"body": ["Invalid JSON."]}}, status=400)
    return data


def _saved_response(template: DemandProfileTemplate, status: int = 200) -> JsonResponse:
    """Build the reply of a successful save.

    Args:
        template: The saved profile.
        status: HTTP status.

    Returns:
        ``{"id", "name", "description", "profile", "edit_url", "list_url"}``.
    """
    return JsonResponse(
        {
            **_detail(template),
            "edit_url": reverse("demand_profile_edit", args=[template.id]),
            "list_url": reverse("demand_profile_list"),
        },
        status=status,
    )


def _detail(template: DemandProfileTemplate) -> dict[str, Any]:
    """Return the JSON form of a library profile.

    Args:
        template: Library profile.

    Returns:
        ``{"id", "name", "description", "profile"}``.
    """
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "profile": template.profile,
    }


def _save(request: HttpRequest, template: DemandProfileTemplate | None) -> JsonResponse:
    """Validate a POSTed profile and create or update it.

    Args:
        request: POST ``{"name", "description", "profile"}``.
        template: The edited profile, or None to create one.

    Returns:
        ``_saved_response`` (HTTP 201 on create), or HTTP 400 ``{"errors":
        {field: [messages]}}`` with keys ``name``, ``description``,
        ``profile`` and ``profile.<field>``.
    """
    data = _json_body(request)
    if isinstance(data, JsonResponse):
        return data
    name = data.get("name")
    profile = data.get("profile")
    description = data.get("description", "")
    try:
        if template is None:
            saved = demand_library.create_profile(name, profile, description)
        else:
            saved = demand_library.update_profile(template, name, profile, description)
    except DemandLibraryError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    except IntegrityError as exc:
        logger.warning("Demand profile write rejected by a constraint: %s", exc)
        return JsonResponse(
            {"errors": {"name": ["A demand profile with this name already exists."]}},
            status=400,
        )
    return _saved_response(saved, status=201 if template is None else 200)


@require_http_methods(["GET", "POST"])
def demand_profile_create(request: HttpRequest) -> HttpResponse:
    """Render the editor of a new profile (GET) or create a profile (POST JSON).

    Args:
        request: GET, or POST ``{"name", "description", "profile"}`` (CSRF
            protected).

    Returns:
        The page, or the ``_save`` reply.
    """
    if request.method == "POST":
        return _save(request, None)
    return render(request, "pages/demand-profile-edit.html", _editor_context(None))


@require_http_methods(["GET", "POST"])
def demand_profile_edit(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Render the editor of a profile (GET) or update it (POST JSON).

    Args:
        request: GET, or POST ``{"name", "description", "profile"}`` (CSRF
            protected).
        profile_id: Id of the library profile.

    Returns:
        The page, or the ``_save`` reply. HTTP 404 for an unknown id.
    """
    template = get_object_or_404(DemandProfileTemplate, id=profile_id)
    if request.method == "POST":
        return _save(request, template)
    return render(request, "pages/demand-profile-edit.html", _editor_context(template))


@require_POST
def demand_profile_duplicate(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Copy a profile as ``'<name> (copy)'`` and open the copy in the editor.

    Args:
        request: POST (CSRF protected).
        profile_id: Id of the library profile.

    Returns:
        Redirect to the editor of the copy, or the ``_saved_response`` of the
        copy (HTTP 201) for a JSON client.
    """
    template = get_object_or_404(DemandProfileTemplate, id=profile_id)
    try:
        duplicate = demand_library.duplicate_profile(template)
    except DemandLibraryError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    if _wants_json(request):
        return _saved_response(duplicate, status=201)
    return redirect("demand_profile_edit", profile_id=duplicate.id)


@require_POST
def demand_profile_delete(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Delete a profile unless a scenario uses it.

    Args:
        request: POST (CSRF protected).
        profile_id: Id of the library profile.

    Returns:
        Redirect to the list (``{"status": "success"}`` for a JSON client),
        or HTTP 400 ``{"errors": {"profile": [...]}}`` naming the scenarios
        that use the profile.
    """
    template = get_object_or_404(DemandProfileTemplate, id=profile_id)
    try:
        demand_library.delete_profile(template)
    except DemandLibraryError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    if _wants_json(request):
        return JsonResponse({"status": "success"})
    return redirect("demand_profile_list")


@require_GET
def demand_profile_detail_json(request: HttpRequest, profile_id: int) -> JsonResponse:
    """Return one library profile for the wizard.

    Args:
        request: GET.
        profile_id: Id of the library profile.

    Returns:
        ``{"id", "name", "description", "profile"}``, HTTP 404
        ``{"errors": {"demand_profile_source": [...]}}`` for an unknown id.
    """
    template = DemandProfileTemplate.objects.filter(id=profile_id).first()
    if template is None:
        return JsonResponse(
            {
                "errors": {
                    "demand_profile_source": [
                        "The selected demand profile is not in the library any more."
                    ]
                }
            },
            status=404,
        )
    return JsonResponse(_detail(template))


@require_GET
def demand_profile_export(request: HttpRequest, profile_id: int) -> HttpResponse:
    """Download a profile as a version-1 YAML file.

    Args:
        request: GET.
        profile_id: Id of the library profile.

    Returns:
        Attachment ``demand_profile_<slug>.yaml``. HTTP 404 for an unknown id.
    """
    template = DemandProfileTemplate.objects.filter(id=profile_id).first()
    if template is None:
        raise Http404("Unknown demand profile.")
    text = demand_profile_yaml.export_profile_yaml(template)
    slug = slugify(template.name) or str(template.id)
    response = HttpResponse(
        text.encode("utf-8"),
        content_type=f"{demand_profile_yaml.CONTENT_TYPE}; charset=utf-8",
    )
    response["Content-Disposition"] = (
        f'attachment; filename="demand_profile_{slug}.yaml"'
    )
    return response


@require_POST
def demand_profile_import_preview(request: HttpRequest) -> JsonResponse:
    """Validate an uploaded profile file and show what the import would do.

    Nothing is written.

    Args:
        request: Multipart POST with ``file`` (CSRF protected).

    Returns:
        ``ProfileImportPlan.to_json()`` with ``status`` ``new`` or ``reuse``.
        HTTP 400 ``{"errors": {...}}`` for an invalid file, and the plan with
        status ``conflict`` and its ``errors`` for a name conflict.
    """
    try:
        plan = demand_profile_yaml.build_import_plan(_uploaded_document(request))
    except demand_profile_yaml.ProfileImportError as exc:
        return JsonResponse({"errors": exc.errors}, status=400)
    body = plan.to_json()
    return JsonResponse(body, status=400 if body["errors"] else 200)


@require_POST
def demand_profile_import_confirm(request: HttpRequest) -> JsonResponse:
    """Validate the uploaded file again and add it to the library.

    The preview is not trusted: the file is parsed and validated from scratch.

    Args:
        request: Multipart POST with ``file`` (CSRF protected).

    Returns:
        ``{"id", "name", "status", "edit_url", "list_url"}`` (``status`` is
        ``new`` or ``reuse``), or HTTP 400 ``{"errors": {...}}`` with nothing
        written.
    """
    try:
        plan, template = demand_profile_yaml.import_profile(_uploaded_document(request))
    except demand_profile_yaml.ProfileImportError as exc:
        return JsonResponse({"errors": exc.errors}, status=400)
    except DemandLibraryError as exc:
        return JsonResponse({"errors": exc.message_dict}, status=400)
    except IntegrityError as exc:
        logger.warning("Demand profile import rejected by a constraint: %s", exc)
        return JsonResponse(
            {"errors": {"name": ["A demand profile with this name already exists."]}},
            status=400,
        )
    return JsonResponse(
        {
            "id": template.id,
            "name": template.name,
            "status": plan.status,
            "edit_url": reverse("demand_profile_edit", args=[template.id]),
            "list_url": reverse("demand_profile_list"),
        }
    )


@require_POST
def demand_profile_parse_yaml(request: HttpRequest) -> JsonResponse:
    """Validate an uploaded profile file for the wizard editor. Nothing is written.

    Args:
        request: Multipart POST with ``file`` (CSRF protected).

    Returns:
        ``{"name", "description", "profile"}`` with the normalised profile,
        or HTTP 400 ``{"errors": {...}}``.
    """
    try:
        parsed = demand_profile_yaml.parse_profile_document(_uploaded_document(request))
    except demand_profile_yaml.ProfileImportError as exc:
        return JsonResponse({"errors": exc.errors}, status=400)
    return JsonResponse(
        {
            "name": parsed.name,
            "description": parsed.description,
            "profile": parsed.profile,
        }
    )
