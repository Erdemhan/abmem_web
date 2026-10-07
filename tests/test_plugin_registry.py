"""Unit tests of the shared plugin registry helper (PLAN 2.10)."""

from __future__ import annotations

import abc
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from django.core.exceptions import ImproperlyConfigured

from abmem.services.plugin_registry import (
    KEY_PATTERN,
    PluginDiscoveryError,
    PluginRegistry,
)


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Yield the directory of a fresh importable package with a subpackage."""
    name = f"abmem_test_registry_{uuid.uuid4().hex[:8]}"
    root = tmp_path / name
    (root / "sub").mkdir(parents=True)
    (root / "skipped").mkdir()
    for directory in (root, root / "sub", root / "skipped"):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield root
    finally:
        for module in [m for m in sys.modules if m.startswith(name)]:
            del sys.modules[module]


def _plugin_base(plugins: PluginRegistry) -> type:
    """Return an abstract base class that registers its subclasses in ``plugins``."""

    class Base(abc.ABC):
        key = ""

        def __init_subclass__(cls, **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            plugins.register(cls)

        @abc.abstractmethod
        def run(self) -> str:
            """Abstract method."""

    return Base


def test_register_get_unregister_and_is_known() -> None:
    plugins: PluginRegistry = PluginRegistry("thing", packages=())
    base = _plugin_base(plugins)

    class Alpha(base):  # type: ignore[misc, valid-type]
        key = "ALPHA"

        def run(self) -> str:
            return "a"

    class NoKey(base):  # type: ignore[misc, valid-type]
        def run(self) -> str:
            return "n"

    assert plugins.classes() == {"ALPHA": Alpha}
    assert plugins.get("ALPHA") is Alpha
    assert plugins.is_known("ALPHA")
    for value in (None, 1, ["ALPHA"], "alpha"):
        assert not plugins.is_known(value)
    with pytest.raises(
        KeyError, match=r"unknown thing 'BETA', known keys: \['ALPHA'\]"
    ):
        plugins.get("BETA")
    plugins.unregister("ALPHA")
    plugins.unregister("ALPHA")
    assert plugins.classes() == {}


def test_same_qualified_name_replaces_and_different_class_clashes() -> None:
    plugins: PluginRegistry = PluginRegistry("thing", packages=())
    base = _plugin_base(plugins)

    def make() -> type:
        class Alpha(base):  # type: ignore[misc, valid-type]
            key = "ALPHA"

            def run(self) -> str:
                return "a"

        return Alpha

    first, second = make(), make()
    assert plugins.get("ALPHA") is second is not first
    with pytest.raises(ImproperlyConfigured, match="duplicate thing key 'ALPHA'"):

        class Other(base):  # type: ignore[misc, valid-type]
            key = "ALPHA"

            def run(self) -> str:
                return "o"


def test_key_pattern() -> None:
    assert KEY_PATTERN.fullmatch("A")
    assert KEY_PATTERN.fullmatch("MERIT_ORDER_2")
    assert KEY_PATTERN.fullmatch("A" * 64)
    for bad in ("", "a", "_A", "1A", "A-B", "A" * 65):
        assert not KEY_PATTERN.fullmatch(bad)


def test_discover_imports_once_skips_private_and_excluded(package: Path) -> None:
    name = package.name
    (package / "one.py").write_text(
        "import builtins\nbuiltins.COUNT_ONE = getattr(builtins, 'COUNT_ONE', 0) + 1\n",
        "utf-8",
    )
    (package / "sub" / "two.py").write_text("VALUE = 2\n", "utf-8")
    (package / "_hidden.py").write_text("raise RuntimeError('private')\n", "utf-8")
    (package / "skipped" / "bad.py").write_text(
        "raise RuntimeError('excluded')\n", "utf-8"
    )
    plugins: PluginRegistry = PluginRegistry(
        "thing", packages=(name,), exclude=(f"{name}.skipped",)
    )
    import builtins

    builtins.COUNT_ONE = 0  # type: ignore[attr-defined]
    try:
        plugins.classes()
        plugins.classes()
        assert builtins.COUNT_ONE == 1  # type: ignore[attr-defined]
        assert f"{name}.sub.two" in sys.modules
        assert f"{name}._hidden" not in sys.modules
        assert f"{name}.skipped.bad" not in sys.modules
    finally:
        del builtins.COUNT_ONE  # type: ignore[attr-defined]


def test_failed_discovery_is_not_cached(package: Path) -> None:
    name = package.name
    broken = package / "broken.py"
    broken.write_text("raise RuntimeError('first attempt')\n", "utf-8")
    plugins: PluginRegistry = PluginRegistry("thing", packages=(name,))
    with pytest.raises(PluginDiscoveryError) as info:
        plugins.classes()
    assert f"{name}.broken" in str(info.value)
    assert "first attempt" in str(info.value)
    assert isinstance(info.value.__cause__, RuntimeError)
    broken.write_text("VALUE = 1\n", "utf-8")
    sys.modules.pop(f"{name}.broken", None)
    assert plugins.classes() == {}


def test_scan_reports_failed_modules_and_keeps_the_others(package: Path) -> None:
    name = package.name
    plugins: PluginRegistry = PluginRegistry("thing", packages=(name,))
    base = _plugin_base(plugins)
    import builtins

    builtins.SCAN_BASE = base  # type: ignore[attr-defined]
    try:
        (package / "good.py").write_text(
            "import builtins\n"
            "class Good(builtins.SCAN_BASE):\n"
            "    key = 'GOOD'\n"
            "    def run(self):\n"
            "        return 'g'\n",
            "utf-8",
        )
        broken = package / "sub" / "broken.py"
        broken.write_text("raise RuntimeError('scan failure')\n", "utf-8")
        classes, errors = plugins.scan()
        assert list(classes) == ["GOOD"]
        assert list(errors) == [f"{name}.sub.broken"]
        assert "RuntimeError: scan failure" in errors[f"{name}.sub.broken"]
        # Discovery is not marked complete, so a normal query still raises.
        with pytest.raises(PluginDiscoveryError):
            plugins.classes()
        broken.write_text("VALUE = 1\n", "utf-8")
        sys.modules.pop(f"{name}.sub.broken", None)
        classes, errors = plugins.scan()
        assert errors == {} and list(classes) == ["GOOD"]
        assert list(plugins.classes()) == ["GOOD"]
    finally:
        del builtins.SCAN_BASE  # type: ignore[attr-defined]


def test_scan_reports_a_missing_package() -> None:
    plugins: PluginRegistry = PluginRegistry(
        "thing", packages=("abmem_no_such_package_for_scan",)
    )
    classes, errors = plugins.scan()
    assert classes == {}
    assert "ModuleNotFoundError" in errors["abmem_no_such_package_for_scan"]
