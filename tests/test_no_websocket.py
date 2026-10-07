"""Guards for PLAN 1.13 (K10): the WebSocket/Channels layer is removed.

The UI obtains simulation progress through HTTP polling, so the project must
run as a plain Django HTTP application without ``channels``.
"""

import importlib.util
import pathlib

import pytest
from django.conf import settings
from django.urls import get_resolver, reverse

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every named HTTP route of abmem/urls.py with sample arguments. Removing the
# WebSocket route must leave all of these resolvable under the same names.
NAMED_ROUTES: list[tuple[str, dict[str, object]]] = [
    ("resource_create", {}),
    ("resource_list", {}),
    ("resource_update", {"resource_id": 1}),
    ("resource_delete", {"resource_id": 1}),
    ("simulation_list_proxy", {}),
    ("simulation_delete", {"simulation_id": 1}),
    ("simulation_list", {}),
    ("simulation_view", {"sim_id": 1}),
    ("get_agents", {}),
    ("agent_create", {}),
    ("agent_list", {}),
    ("agent_update", {"agent_id": 1}),
    ("agent_delete", {"agent_id": 1}),
    ("blank", {}),
    ("dashboard", {}),
    ("create_simulation", {}),
    ("run_simulation", {"simulation_id": 1}),
    ("update_simulation", {"simulation_id": 1}),
    ("watch_simulation", {"simulation_id": "1"}),
    ("get_simulation_status", {}),
    ("nextPeriod", {}),
]


@pytest.mark.parametrize(("name", "kwargs"), NAMED_ROUTES)
def test_named_route_still_resolves(name: str, kwargs: dict[str, object]) -> None:
    """Each named HTTP route reverses to a path under /dev/."""
    url = reverse(name, kwargs=kwargs or None)
    assert url.startswith("/dev/")


def test_no_websocket_route_registered() -> None:
    """No URL pattern under the root resolver points to a ws/ path."""

    def _patterns(resolver, prefix: str = "") -> list[str]:
        out: list[str] = []
        for entry in resolver.url_patterns:
            route = prefix + str(entry.pattern)
            if hasattr(entry, "url_patterns"):
                out.extend(_patterns(entry, route))
            else:
                out.append(route)
        return out

    routes = _patterns(get_resolver())
    assert routes, "the root resolver must expose at least one route"
    assert not [r for r in routes if "ws/" in r]


def test_consumers_and_routing_modules_removed() -> None:
    """abmem.consumers and abmem.routing no longer exist."""
    assert importlib.util.find_spec("abmem.consumers") is None
    assert importlib.util.find_spec("abmem.routing") is None


def test_channels_not_configured() -> None:
    """Settings carry no Channels configuration."""
    assert "channels" not in settings.INSTALLED_APPS
    assert not hasattr(settings, "CHANNEL_LAYERS")
    assert not hasattr(settings, "ASGI_APPLICATION")


def test_no_source_imports_channels() -> None:
    """No project Python module imports channels or abmem.consumers."""
    offenders: list[str] = []
    for folder in ("abmem", "web_project"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if "migrations" in path.parts:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith(("import channels", "from channels")) or (
                    "abmem.consumers" in stripped
                    and stripped.startswith(("import ", "from "))
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {stripped}")
    assert offenders == []


def test_requirements_drop_channels() -> None:
    """requirements.txt lists no Channels, channels-redis or daphne package."""
    names = {
        line.split("==")[0].strip().lower()
        for line in (REPO_ROOT / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert not names & {"channels", "channels-redis", "channels_redis", "daphne"}


def test_asgi_application_is_plain_django_handler() -> None:
    """web_project.asgi.application is Django's own ASGIHandler."""
    from django.core.handlers.asgi import ASGIHandler

    from web_project.asgi import application

    assert isinstance(application, ASGIHandler)


@pytest.mark.django_db
@pytest.mark.parametrize("path", ["/dev/", "/dev/simulation/list"])
def test_http_pages_still_served(client, path: str) -> None:
    """The dashboard and simulation list answer over plain HTTP."""
    response = client.get(path)
    assert response.status_code == 200
