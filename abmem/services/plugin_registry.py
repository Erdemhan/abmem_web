"""Registry of plugin classes discovered from the modules of a package (PLAN 2.10).

Agent algorithms and market clearing algorithms are both plain subclasses of
a base class. The base class calls :meth:`PluginRegistry.register` from its
``__init_subclass__`` hook, so every concrete subclass that declares its own
``key`` is registered when its module is imported. :meth:`PluginRegistry.discover`
imports every module of the configured packages once, so a new class dropped
into the right package is found without editing any table.

Discovery runs lazily on the first registry query, never at Django start, so
importing this module or a registry does not import torch.

The module has no Django model imports and no torch imports.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import threading
import traceback
from collections.abc import Iterable, Sequence
from typing import Final, Generic, TypeVar

from django.core.exceptions import ImproperlyConfigured

__all__ = ["KEY_PATTERN", "PluginDiscoveryError", "PluginRegistry"]

KEY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

T = TypeVar("T", bound=type)


class PluginDiscoveryError(ImproperlyConfigured):
    """Raised when a module of a plugin package cannot be imported."""


def _qualified_name(cls: type) -> str:
    """Return ``module.QualName`` of ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


class PluginRegistry(Generic[T]):
    """Key-to-class registry filled by subclass registration and package discovery.

    Attributes:
        kind: Human-readable name of the plugin kind used in error messages,
            for example ``"agent algorithm"``.
    """

    def __init__(
        self, kind: str, packages: Sequence[str], exclude: Iterable[str] = ()
    ) -> None:
        """Create an empty registry.

        Args:
            kind: Name of the plugin kind used in error messages.
            packages: Dotted names of the packages whose modules are imported
                by :meth:`discover`.
            exclude: Dotted names of modules or subpackages that discovery
                does not import.
        """
        self.kind = kind
        self._packages: tuple[str, ...] = tuple(packages)
        self._exclude: frozenset[str] = frozenset(exclude)
        self._classes: dict[str, T] = {}
        self._discovered = False
        self._lock = threading.RLock()

    def register(self, cls: T) -> bool:
        """Register ``cls`` when it is concrete and declares its own ``key``.

        A class that only inherits ``key`` from a parent is not registered,
        so subclassing an existing algorithm does not collide with its key.
        Registering a class with the same module and qualified name as the
        registered one replaces it (module reload).

        Args:
            cls: The subclass being created.

        Returns:
            True if ``cls`` was registered.

        Raises:
            ImproperlyConfigured: If the key does not match :data:`KEY_PATTERN`
                or another class already uses it.
        """
        key = cls.__dict__.get("key")
        if not key or inspect.isabstract(cls):
            return False
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
            raise ImproperlyConfigured(
                f"{self.kind} {_qualified_name(cls)} has an invalid key {key!r}, "
                f"expected the format {KEY_PATTERN.pattern}"
            )
        with self._lock:
            existing = self._classes.get(key)
            if existing is not None and _qualified_name(existing) != _qualified_name(
                cls
            ):
                raise ImproperlyConfigured(
                    f"duplicate {self.kind} key {key!r}: "
                    f"{_qualified_name(existing)} and {_qualified_name(cls)}"
                )
            self._classes[key] = cls
        return True

    def unregister(self, key: str) -> None:
        """Remove ``key`` from the registry if present.

        Used by tests that register temporary plugin classes.

        Args:
            key: The key to remove.
        """
        with self._lock:
            self._classes.pop(key, None)

    def discover_package(
        self, package: str, errors: dict[str, str] | None = None
    ) -> None:
        """Import every public module of ``package`` and of its subpackages.

        Modules whose last name part starts with ``_`` and the names in
        ``exclude`` are skipped.

        Args:
            package: Dotted package name.
            errors: When given, a module that fails to import is recorded here
                (module name to traceback text) and the other modules are
                still imported, instead of raising.

        Raises:
            PluginDiscoveryError: If a module fails to import and ``errors``
                is None. The message contains the module name and the
                traceback.
        """
        root = importlib.import_module(package)
        for info in pkgutil.iter_modules(getattr(root, "__path__", []), package + "."):
            short_name = info.name.rsplit(".", 1)[-1]
            if short_name.startswith("_") or info.name in self._exclude:
                continue
            try:
                importlib.import_module(info.name)
            except Exception as error:
                if errors is not None:
                    errors[info.name] = traceback.format_exc()
                    continue
                raise PluginDiscoveryError(
                    f"{self.kind} discovery: module {info.name} failed to import.\n"
                    f"{traceback.format_exc()}"
                ) from error
            if info.ispkg:
                self.discover_package(info.name, errors)

    def scan(self) -> tuple[dict[str, T], dict[str, str]]:
        """Import every module and report import failures instead of raising.

        Used by the Algorithms page (PLAN 2.12), which lists the registered
        classes next to the modules that failed. Discovery is marked complete
        only when no module failed, so other queries still raise.

        Returns:
            The registered classes by key and the import errors by module name
            (traceback text).
        """
        errors: dict[str, str] = {}
        with self._lock:
            for package in self._packages:
                try:
                    self.discover_package(package, errors)
                except Exception:
                    errors[package] = traceback.format_exc()
            if not errors:
                self._discovered = True
            return dict(self._classes), errors

    def discover(self) -> None:
        """Import the modules of the configured packages once (cached).

        A failed discovery is not cached, so the next query retries.

        Raises:
            PluginDiscoveryError: If a module fails to import.
        """
        if self._discovered:
            return
        with self._lock:
            if self._discovered:
                return
            for package in self._packages:
                self.discover_package(package)
            self._discovered = True

    def classes(self) -> dict[str, T]:
        """Return a copy of the key-to-class mapping after discovery.

        Returns:
            Registered classes by key.
        """
        self.discover()
        with self._lock:
            return dict(self._classes)

    def get(self, key: str) -> T:
        """Return the class registered under ``key``.

        Args:
            key: Plugin key.

        Returns:
            The registered class.

        Raises:
            KeyError: If ``key`` is unknown. The message lists the known keys.
        """
        classes = self.classes()
        try:
            return classes[key]
        except (KeyError, TypeError):
            raise KeyError(
                f"unknown {self.kind} {key!r}, known keys: {sorted(classes)}"
            ) from None

    def is_known(self, key: object) -> bool:
        """Return True if ``key`` is a registered key.

        Args:
            key: Any value. Non-strings are never known.
        """
        return isinstance(key, str) and key in self.classes()
