"""Views of the Algorithms page (PLAN 2.12, spec algorithms_tab).

Routes (prefix ``/dev/``):

* ``algorithms/`` (``algorithm_list``): agent algorithms and market clearing
  algorithms in two columns,
* ``algorithms/<kind>/<key>/`` (``algorithm_params``): GET returns the
  parameter form as JSON, POST ``{"params": {...}}`` saves the global
  default (HTTP 400 ``{"errors": {field: [messages]}}`` when invalid),
* ``algorithms/<kind>/<key>/reset/`` (``algorithm_params_reset``): POST
  deletes the stored default so the published defaults apply.

``kind`` is ``agent`` or ``clearing``. An unknown key returns 404.
"""

from __future__ import annotations

import json
from typing import Any

from django.http import Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_http_methods, require_POST

from .services import algorithm_catalog
from .services.algorithm_catalog import AGENT, CLEARING, AlgorithmKind
from .services.param_schema import NON_FIELD_ERRORS

KINDS: tuple[AlgorithmKind, ...] = (AGENT, CLEARING)


def _kind(kind: str) -> AlgorithmKind:
    """Return ``kind`` as an algorithm kind or raise 404."""
    if kind not in KINDS:
        raise Http404(f"Unknown algorithm kind {kind!r}.")
    return kind  # type: ignore[return-value]


@require_GET
def algorithm_list(request: HttpRequest) -> HttpResponse:
    """Render the Algorithms page.

    Args:
        request: GET request.

    Returns:
        The page with ``agent_cards``, ``clearing_cards``, ``agent_errors``
        and ``clearing_errors`` (modules that failed to import), the same
        data grouped as ``columns`` and ``page_config`` (URL templates for
        the page script).
    """
    agent_cards, agent_errors = algorithm_catalog.agent_cards()
    clearing_cards, clearing_errors = algorithm_catalog.clearing_cards()
    context = {
        "agent_cards": agent_cards,
        "clearing_cards": clearing_cards,
        "agent_errors": agent_errors,
        "clearing_errors": clearing_errors,
        "columns": [
            {
                "kind": AGENT,
                "title": "Agent algorithms",
                "help": "Bidding policies of the agents: learning algorithms "
                "and fixed baselines.",
                "cards": agent_cards,
                "errors": agent_errors,
            },
            {
                "kind": CLEARING,
                "title": "Market clearing algorithms",
                "help": "Rules that compute the accepted quantities and the "
                "clearing price of a period.",
                "cards": clearing_cards,
                "errors": clearing_errors,
            },
        ],
        "page_config": {
            # The page replaces KIND and KEY.
            "params_url": reverse("algorithm_params", args=["KIND", "KEY"]),
            "reset_url": reverse("algorithm_params_reset", args=["KIND", "KEY"]),
        },
    }
    return render(request, "pages/algorithm-list.html", context)


def _form_or_404(kind: AlgorithmKind, key: str) -> dict[str, Any] | JsonResponse:
    """Return the parameter form of ``key``.

    Returns:
        The form, or a 409 response when the config class cannot be
        described (unsupported field type or invalid class defaults).

    Raises:
        Http404: Unknown key.
    """
    try:
        return algorithm_catalog.param_form(kind, key)
    except KeyError as error:
        raise Http404(str(error.args[0]) if error.args else str(error)) from None
    except (TypeError, ValueError) as error:
        return JsonResponse(
            {
                "errors": {
                    NON_FIELD_ERRORS: [f"The parameters cannot be edited: {error}"]
                }
            },
            status=409,
        )


@require_http_methods(["GET", "POST"])
def algorithm_params(request: HttpRequest, kind: str, key: str) -> JsonResponse:
    """Return or save the global default parameters of one algorithm.

    Args:
        request: GET, or POST with a JSON body ``{"params": {name: value}}``.
        kind: ``agent`` or ``clearing``.
        key: Registry key.

    Returns:
        The form JSON (200, also after a successful save) or
        ``{"errors": {field: [messages]}}`` with 400.

    Raises:
        Http404: Unknown kind or key.
    """
    algorithm_kind = _kind(kind)
    form = _form_or_404(algorithm_kind, key)
    if isinstance(form, JsonResponse):
        return form
    if request.method == "GET":
        return JsonResponse(form)
    try:
        body = json.loads(request.body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse(
            {"errors": {NON_FIELD_ERRORS: ["The request body is not valid JSON."]}},
            status=400,
        )
    params = body.get("params") if isinstance(body, dict) else None
    if not isinstance(params, dict):
        return JsonResponse(
            {"errors": {NON_FIELD_ERRORS: ["'params' must be an object."]}},
            status=400,
        )
    validation = algorithm_catalog.save_defaults(algorithm_kind, key, params)
    if not validation.is_valid:
        return JsonResponse({"errors": validation.errors}, status=400)
    return JsonResponse(algorithm_catalog.param_form(algorithm_kind, key))


@require_POST
def algorithm_params_reset(request: HttpRequest, kind: str, key: str) -> JsonResponse:
    """Delete the stored default of one algorithm.

    Args:
        request: POST request.
        kind: ``agent`` or ``clearing``.
        key: Registry key.

    Returns:
        The form JSON with the published defaults (200).

    Raises:
        Http404: Unknown kind or key.
    """
    algorithm_kind = _kind(kind)
    form = _form_or_404(algorithm_kind, key)
    if isinstance(form, JsonResponse):
        return form
    algorithm_catalog.reset_defaults(algorithm_kind, key)
    return JsonResponse(algorithm_catalog.param_form(algorithm_kind, key))
