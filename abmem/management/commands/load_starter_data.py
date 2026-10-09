"""Load the starter resources, scenarios and demand profiles into an empty database."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Final

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from abmem.demand_profile_yaml import import_profile
from abmem.models import Agent, Resource, Simulation
from abmem.scenario_yaml import (
    ScenarioImportError,
    import_resources,
    import_scenario,
    load_yaml_bytes,
)

STARTER_DIR: Final[Path] = Path(__file__).resolve().parents[2] / "starter_data"


def database_is_empty() -> bool:
    """Return True when the database holds no resource, agent or simulation."""
    return not (
        Resource.objects.exists()
        or Agent.objects.exists()
        or Simulation.objects.exists()
    )


def load_starter_data(directory: Path = STARTER_DIR) -> dict[str, int]:
    """Import the starter files of ``directory`` in one transaction.

    Order: ``resources.yaml``, then ``demand_profiles/*.yaml``, then
    ``scenarios/*.yaml`` (file name order). Every file goes through the same
    import functions as the web interface, so the same validation and reuse
    rules apply.

    Args:
        directory: Folder with the starter files.

    Returns:
        Counts ``{"resources", "demand_profiles", "scenarios"}`` of the
        imported documents.

    Raises:
        CommandError: A file is invalid or conflicts with existing data.
            Nothing is written in that case.
    """
    counts = {"resources": 0, "demand_profiles": 0, "scenarios": 0}
    with transaction.atomic():
        resources_file = directory / "resources.yaml"
        if resources_file.is_file():
            counts["resources"] = len(
                import_resources(_load(resources_file))
            )
        for path in sorted((directory / "demand_profiles").glob("*.yaml")):
            import_profile(_load(path))
            counts["demand_profiles"] += 1
        for path in sorted((directory / "scenarios").glob("*.yaml")):
            try:
                import_scenario(_load(path))
            except ScenarioImportError as exc:
                raise CommandError(f"{path.name}: {exc}") from exc
            counts["scenarios"] += 1
    return counts


def _load(path: Path) -> Any:
    """Read and parse one YAML file.

    Args:
        path: File to read.

    Returns:
        The parsed document.

    Raises:
        CommandError: The file is not valid YAML.
    """
    try:
        return load_yaml_bytes(path.read_bytes())
    except ScenarioImportError as exc:
        raise CommandError(f"{path.name}: {exc}") from exc


class Command(BaseCommand):
    """Import the starter data shipped in ``abmem/starter_data``."""

    help = (
        "Import the starter resources, demand profiles and scenario templates. "
        "By default only into an empty database."
    )

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register command options.

        Args:
            parser: The argument parser of the command.
        """
        parser.add_argument(
            "--force",
            action="store_true",
            help="Import even when the database already holds data.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Import the starter data or report that it was skipped.

        Args:
            *args: Unused positional arguments.
            **options: Parsed options, including ``force``.
        """
        if not options["force"] and not database_is_empty():
            self.stdout.write("Database already holds data, starter data skipped.")
            return
        counts = load_starter_data()
        self.stdout.write(
            self.style.SUCCESS(
                "Starter data loaded: {resources} resources, {demand_profiles} "
                "demand profiles, {scenarios} scenarios.".format(**counts)
            )
        )
