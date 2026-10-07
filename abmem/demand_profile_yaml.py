"""Demand profile YAML export and import (PLAN 2.9, spec demand_profile_library).

File format, version 1 (exact keys)::

    format_version: 1
    kind: demand_profile
    name: <library name>
    description: <optional text>
    demand: {version: 1, seed, day_start_hour, day_end_hour, autocorrelation,
             day: {distribution, mean, std, min, max}, night: {...}}

``demand`` holds the settings only (``DemandProfile.to_dict``), never
generated values. It is validated by ``DemandProfile.from_dict`` through
``demand_library.validate_profile``.

Import rules (same as the agent rule of the scenario import):

* a new name is created on confirm,
* the same name (case-insensitive) with an equal normalised profile reuses
  the library profile,
* the same name with a different profile is a conflict, nothing is written.

File loading (size limit, UTF-8, ``yaml.safe_load``), the error type and the
dumper are shared with ``abmem.scenario_yaml``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import yaml
from django.db import transaction
from django.utils import timezone

from .models import DemandProfileTemplate
from .scenario_yaml import (
    CONTENT_TYPE,
    MAX_FILE_BYTES,
    ScenarioImportError,
    _ExportDumper,
    load_yaml_bytes,
)
from .services.market import demand_library
from .services.market.demand_library import DemandLibraryError

__all__ = [
    "CONTENT_TYPE",
    "FORMAT_VERSION",
    "KIND",
    "MAX_FILE_BYTES",
    "ProfileImportError",
    "ProfileImportPlan",
    "build_import_plan",
    "export_profile_yaml",
    "import_profile",
    "load_yaml_bytes",
    "parse_profile_document",
    "profile_document",
]

FORMAT_VERSION: Final[int] = 1
KIND: Final[str] = "demand_profile"
TOP_KEYS: Final[tuple[str, ...]] = (
    "format_version",
    "kind",
    "name",
    "description",
    "demand",
)
REQUIRED_KEYS: Final[tuple[str, ...]] = ("format_version", "kind", "name", "demand")

STATUS_NEW: Final[str] = "new"
STATUS_REUSE: Final[str] = "reuse"
STATUS_CONFLICT: Final[str] = "conflict"

ProfileImportError = ScenarioImportError
"""The uploaded document cannot be imported. Nothing was written.

The same error type as the scenario import (path-keyed ``errors``), so the
upload helpers of both imports are shared.
"""


# ---------------------------------------------------------------- export


def profile_document(template: DemandProfileTemplate) -> dict[str, Any]:
    """Build the version-1 document of a library profile.

    Args:
        template: Library profile.

    Returns:
        Mapping with the keys of ``TOP_KEYS`` in file order.
    """
    return {
        "format_version": FORMAT_VERSION,
        "kind": KIND,
        "name": template.name,
        "description": template.description,
        "demand": copy.deepcopy(template.profile),
    }


def export_profile_yaml(
    template: DemandProfileTemplate, *, now: datetime | None = None
) -> str:
    """Serialise a library profile as YAML text with a header comment.

    Args:
        template: Library profile.
        now: Export time for the header, ``timezone.now()`` when None.

    Returns:
        The YAML text (keys in file order).
    """
    stamp = (now or timezone.now()).isoformat(timespec="seconds")
    header = (
        "# ABMEM demand profile export\n"
        f"# Exported: {stamp}\n"
        f"# Source profile id: {template.pk}\n"
    )
    body = yaml.dump(
        profile_document(template),
        Dumper=_ExportDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return header + body


# ---------------------------------------------------------------- import


@dataclass(frozen=True)
class ParsedProfile:
    """A validated demand profile document.

    Attributes:
        name: Stripped library name (not checked against the library).
        description: Stripped description, empty when absent.
        profile: Normalised profile (``DemandProfile.to_dict``).
    """

    name: str
    description: str
    profile: dict[str, Any]


def parse_profile_document(document: Any) -> ParsedProfile:
    """Validate a parsed demand profile document without touching the database.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        The validated content.

    Raises:
        ProfileImportError: Path-keyed errors (``file``, ``format_version``,
            ``kind``, ``name``, ``description``, ``demand``,
            ``demand.<field>``).
    """
    if not isinstance(document, dict):
        raise ProfileImportError(
            {"file": ["The file must contain a mapping with the demand profile keys."]}
        )
    errors: dict[str, list[str]] = {}

    def add(key: str, message: str) -> None:
        errors.setdefault(key, []).append(message)

    unknown = [str(key) for key in document if key not in TOP_KEYS]
    if unknown:
        add(
            "file",
            f"Unknown key(s): {', '.join(unknown)}. Allowed keys: {', '.join(TOP_KEYS)}.",
        )
    missing = [key for key in REQUIRED_KEYS if key not in document]
    if missing:
        add("file", f"Missing key(s): {', '.join(missing)}.")
    version = document.get("format_version")
    if "format_version" in document and (
        isinstance(version, bool) or version != FORMAT_VERSION
    ):
        add(
            "format_version",
            f"Unsupported format_version {version!r}, expected {FORMAT_VERSION}.",
        )
    if "kind" in document and document.get("kind") != KIND:
        add("kind", f"kind must be '{KIND}', got {document.get('kind')!r}.")
    if errors:
        raise ProfileImportError(errors)

    name = document.get("name")
    if not isinstance(name, str) or not name.strip():
        add("name", "name must be a non-empty text.")
    elif len(name.strip()) > DemandProfileTemplate._meta.get_field("name").max_length:
        add(
            "name",
            "name must have at most "
            f"{DemandProfileTemplate._meta.get_field('name').max_length} characters.",
        )
    description = document.get("description")
    if description is not None and not isinstance(description, str):
        add("description", "description must be text.")
    profile: dict[str, Any] = {}
    if document.get("demand") is None:
        add("demand", "demand is required and must be a mapping.")
    else:
        try:
            profile = demand_library.validate_profile(document["demand"], key="demand")
        except DemandLibraryError as exc:
            for key, messages in exc.message_dict.items():
                errors.setdefault(key, []).extend(messages)
    if errors:
        raise ProfileImportError(errors)
    return ParsedProfile(
        name=name.strip(),
        description=(description or "").strip(),
        profile=profile,
    )


@dataclass(frozen=True)
class ProfileImportPlan:
    """What an import would do.

    Attributes:
        parsed: The validated document.
        existing: Library profile with the same name (case-insensitive), or
            None.
        status: ``new``, ``reuse`` (same name, equal profile) or ``conflict``
            (same name, different profile).
    """

    parsed: ParsedProfile
    existing: DemandProfileTemplate | None
    status: str

    def conflict_errors(self) -> dict[str, list[str]]:
        """Return the errors of a conflicting plan.

        Returns:
            ``{"name": [...]}`` for status ``conflict``, else ``{}``.
        """
        if self.status != STATUS_CONFLICT or self.existing is None:
            return {}
        return {
            "name": [
                f"The library already has a demand profile named "
                f"'{self.existing.name}' with different settings. Rename the "
                "profile in the file or change the library profile first."
            ]
        }

    def to_json(self) -> dict[str, Any]:
        """Return the preview shown before confirmation.

        Returns:
            ``{"name", "description", "demand", "status", "existing_id",
            "errors"}``.
        """
        return {
            "name": self.parsed.name,
            "description": self.parsed.description,
            "demand": copy.deepcopy(self.parsed.profile),
            "status": self.status,
            "existing_id": self.existing.id if self.existing is not None else None,
            "errors": self.conflict_errors(),
        }


def build_import_plan(document: Any) -> ProfileImportPlan:
    """Validate a document and compare it with the library. Nothing is written.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        The plan, possibly with status ``conflict``.

    Raises:
        ProfileImportError: The document is invalid.
    """
    parsed = parse_profile_document(document)
    existing = demand_library.find_by_name(parsed.name)
    if existing is None:
        status = STATUS_NEW
    elif demand_library.validate_profile(existing.profile) == parsed.profile:
        status = STATUS_REUSE
    else:
        status = STATUS_CONFLICT
    return ProfileImportPlan(parsed=parsed, existing=existing, status=status)


def import_profile(document: Any) -> tuple[ProfileImportPlan, DemandProfileTemplate]:
    """Validate a document and write it to the library in one transaction.

    Args:
        document: Output of ``load_yaml_bytes``.

    Returns:
        ``(plan, template)``: the new profile, or the reused library profile.

    Raises:
        ProfileImportError: Invalid document or name conflict (nothing
            written).
        DemandLibraryError: Rejected by the library service during the
            write (nothing written).
    """
    with transaction.atomic():
        plan = build_import_plan(document)
        if plan.status == STATUS_CONFLICT:
            raise ProfileImportError(plan.conflict_errors())
        if plan.status == STATUS_REUSE and plan.existing is not None:
            return plan, plan.existing
        template = demand_library.create_profile(
            plan.parsed.name, plan.parsed.profile, plan.parsed.description
        )
    return plan, template
