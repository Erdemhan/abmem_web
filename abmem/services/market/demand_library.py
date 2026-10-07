"""Demand profile library service (PLAN 2.9, spec demand_profile_library).

Every write of a ``DemandProfileTemplate`` goes through this module.

* Names are stripped, required, at most ``DEMAND_PROFILE_NAME_MAX_LENGTH``
  characters and unique case-insensitively as computed by the database
  ``LOWER`` function (constraint ``uniq_demand_profile_name_ci``), the same
  comparison as the agent library (``library_service``).
* Profiles are parsed and validated only by
  ``demand_generator.DemandProfile.from_dict`` and stored as ``to_dict()``.
* A scenario keeps its own copy of a library profile
  (``Simulation.demand_profile``). ``Simulation.demand_profile_source`` is an
  informational link, so updating a library profile does not change a
  scenario. A profile linked by a proxy scenario cannot be deleted. Runs do
  not block a deletion, their link becomes NULL.

All validation failures raise ``DemandLibraryError``, a ``ValidationError``
subclass whose ``message_dict`` is keyed by field (``name``, ``profile``,
``profile.day.std``, ``description``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from django.core.exceptions import ValidationError
from django.db import transaction
from django.db.models import Value
from django.db.models.functions import Lower

from ...models.demand import DEMAND_PROFILE_NAME_MAX_LENGTH, DemandProfileTemplate
from ...models.simulation import Simulation
from .demand_generator import DemandProfile

__all__ = [
    "DemandLibraryError",
    "PROFILE_KEY",
    "copy_profile_name",
    "create_profile",
    "delete_profile",
    "duplicate_profile",
    "find_by_name",
    "scenarios_using",
    "update_profile",
    "validate_profile",
]

PROFILE_KEY: Final[str] = "profile"
COPY_SUFFIX: Final[str] = " (copy)"


class DemandLibraryError(ValidationError):
    """Deterministic, field-keyed validation error of the demand profile library.

    Subclass of ``ValidationError`` like
    ``library_service.LibraryValidationError``. Views map it to HTTP 400.
    """


def _error(field: str, message: str, code: str) -> DemandLibraryError:
    """Build a single-field ``DemandLibraryError``.

    Args:
        field: Field key.
        message: Human-readable message.
        code: Machine-readable error code.

    Returns:
        The error instance.
    """
    return DemandLibraryError({field: [ValidationError(message, code=code)]})


def _same_name(name: str) -> Any:
    """Return the profiles whose name equals ``name`` under the database ``LOWER``.

    Args:
        name: Stripped name.

    Returns:
        Queryset of matching profiles.
    """
    return DemandProfileTemplate.objects.annotate(_lname=Lower("name")).filter(
        _lname=Lower(Value(name))
    )


def find_by_name(name: str) -> DemandProfileTemplate | None:
    """Return the library profile named ``name`` (case-insensitive), if any.

    Args:
        name: Name to look up. Surrounding whitespace is ignored.

    Returns:
        The profile, or None for an unknown or empty name.
    """
    if not isinstance(name, str) or not name.strip():
        return None
    return _same_name(name.strip()).first()


def _validate_name(name: Any, *, exclude_id: int | None) -> str:
    """Strip and validate a library profile name.

    Args:
        name: Raw name.
        exclude_id: Profile id to ignore in the uniqueness check.

    Returns:
        The stripped name.

    Raises:
        DemandLibraryError: Not a string, empty, too long or already taken
            (key ``name``).
    """
    if not isinstance(name, str):
        raise _error("name", "Profile name must be a string.", "invalid")
    stripped = name.strip()
    if not stripped:
        raise _error("name", "Profile name is required.", "required")
    if len(stripped) > DEMAND_PROFILE_NAME_MAX_LENGTH:
        raise _error(
            "name",
            f"Profile name must have at most {DEMAND_PROFILE_NAME_MAX_LENGTH} "
            f"characters ({len(stripped)} given).",
            "max_length",
        )
    taken = _same_name(stripped)
    if exclude_id is not None:
        taken = taken.exclude(id=exclude_id)
    if taken.exists():
        raise _error(
            "name",
            f"A demand profile named '{stripped}' already exists "
            "(names are compared case-insensitively).",
            "name_taken",
        )
    return stripped


def validate_profile(profile: Any, key: str = PROFILE_KEY) -> dict[str, Any]:
    """Validate a profile mapping with ``DemandProfile.from_dict``.

    Args:
        profile: Raw profile mapping (``version`` optional).
        key: Error key of the whole object. Field errors are keyed
            ``"<key>.<field>"`` (``profile.day.std``).

    Returns:
        The normalised profile (``DemandProfile.to_dict()``).

    Raises:
        DemandLibraryError: Field-keyed errors as described above.
    """
    try:
        return DemandProfile.from_dict(profile).to_dict()
    except ValidationError as exc:
        raise DemandLibraryError(
            {
                (key if name == "demand_profile" else f"{key}.{name}"): messages
                for name, messages in exc.message_dict.items()
            }
        ) from None


def _validate_description(description: Any) -> str:
    """Validate the optional description.

    Args:
        description: Raw description (None is treated as empty).

    Returns:
        The stripped description.

    Raises:
        DemandLibraryError: Not a string (key ``description``).
    """
    if description is None:
        return ""
    if not isinstance(description, str):
        raise _error("description", "Description must be text.", "invalid")
    return description.strip()


def _validated(
    name: Any, profile: Any, description: Any, *, exclude_id: int | None
) -> tuple[str, dict[str, Any], str]:
    """Validate all fields and report every error at once.

    Args:
        name: Raw name.
        profile: Raw profile mapping.
        description: Raw description.
        exclude_id: Profile id to ignore in the name check.

    Returns:
        ``(name, normalised_profile, description)``.

    Raises:
        DemandLibraryError: Field-keyed errors of all fields.
    """
    errors: dict[str, list[str]] = {}
    values: dict[str, Any] = {}
    for field, check in (
        ("name", lambda: _validate_name(name, exclude_id=exclude_id)),
        ("profile", lambda: validate_profile(profile)),
        ("description", lambda: _validate_description(description)),
    ):
        try:
            values[field] = check()
        except DemandLibraryError as exc:
            for key, messages in exc.message_dict.items():
                errors.setdefault(key, []).extend(messages)
    if errors:
        raise DemandLibraryError(errors)
    return values["name"], values["profile"], values["description"]


@transaction.atomic
def create_profile(
    name: str, profile: Mapping[str, Any], description: str = ""
) -> DemandProfileTemplate:
    """Add a profile to the library.

    Args:
        name: Library name (stripped, unique case-insensitively).
        profile: Version-1 profile mapping.
        description: Optional text.

    Returns:
        The saved profile with the normalised ``profile``.

    Raises:
        DemandLibraryError: Field-keyed errors (``name``, ``profile[.<field>]``,
            ``description``). Nothing is written.
    """
    clean_name, clean_profile, clean_description = _validated(
        name, profile, description, exclude_id=None
    )
    return DemandProfileTemplate.objects.create(
        name=clean_name, profile=clean_profile, description=clean_description
    )


@transaction.atomic
def update_profile(
    template: DemandProfileTemplate,
    name: str,
    profile: Mapping[str, Any],
    description: str = "",
) -> DemandProfileTemplate:
    """Change a library profile. Scenarios keep their own copies.

    Args:
        template: The profile to change.
        name: New name (the profile itself is ignored in the name check).
        profile: New version-1 profile mapping.
        description: New description.

    Returns:
        ``template`` after saving.

    Raises:
        DemandLibraryError: Field-keyed errors. Nothing is written.
    """
    clean_name, clean_profile, clean_description = _validated(
        name, profile, description, exclude_id=template.pk
    )
    template.name = clean_name
    template.profile = clean_profile
    template.description = clean_description
    template.save()
    return template


def copy_profile_name(name: str) -> str:
    """Return the first free copy name ``'<name> (copy)'``, ``'<name> (copy 2)'``, ...

    The base name is shortened so that the result fits the name column.

    Args:
        name: Name of the original profile.

    Returns:
        A name that no library profile uses (case-insensitive).
    """
    number = 1
    while True:
        suffix = COPY_SUFFIX if number == 1 else f" (copy {number})"
        base = name.strip()[
            : max(DEMAND_PROFILE_NAME_MAX_LENGTH - len(suffix), 0)
        ].rstrip()
        candidate = f"{base}{suffix}"
        if not _same_name(candidate).exists():
            return candidate
        number += 1


@transaction.atomic
def duplicate_profile(template: DemandProfileTemplate) -> DemandProfileTemplate:
    """Copy a library profile under the first free copy name.

    Args:
        template: The profile to copy.

    Returns:
        The new profile with the same settings and description.
    """
    return create_profile(
        copy_profile_name(template.name), template.profile, template.description
    )


def scenarios_using(template: DemandProfileTemplate) -> list[Simulation]:
    """Return the proxy scenarios linked to a library profile.

    Runs (non-proxy simulations) are not included.

    Args:
        template: A library profile.

    Returns:
        Distinct proxy simulations ordered by id.
    """
    return list(
        Simulation.objects.filter(proxy=True, demand_profile_source=template)
        .distinct()
        .order_by("id")
    )


@transaction.atomic
def delete_profile(template: DemandProfileTemplate) -> None:
    """Delete a library profile unless a scenario is linked to it.

    Runs linked to the profile keep their own copy and lose the link.

    Args:
        template: The profile to delete.

    Raises:
        DemandLibraryError: One or more proxy scenarios are linked to the
            profile (key ``profile``, code ``in_use``, the message names the
            scenarios). Nothing is deleted.
    """
    used = scenarios_using(template)
    if used:
        names = ", ".join(f"'{s.name}' (#{s.id})" for s in used)
        raise _error(
            "profile",
            f"Demand profile '{template.name}' is used by the scenario(s) "
            f"{names}. Change their demand source or delete them first.",
            "in_use",
        )
    template.delete()
