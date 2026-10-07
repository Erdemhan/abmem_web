"""Run a hyperparameter study in the foreground (PLAN 2.4).

Examples:

    python manage.py run_hpo --study 12
    python manage.py run_hpo --config study.json

``--config`` reads a JSON object in the input format of
``hpo_service.validate_study_input``, creates the study and runs it.
"""

from __future__ import annotations

import json
import pathlib
from argparse import ArgumentParser
from typing import Any

from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    """Create (optionally) and run one HPO study in this process."""

    help = "Run a hyperparameter study (existing --study ID or new from --config JSON)."

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register command options.

        Args:
            parser: The argument parser of the command.
        """
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("--study", type=int, help="Id of a CREATED study.")
        group.add_argument(
            "--config", type=pathlib.Path, help="JSON file describing a new study."
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Create the study if requested, run it and print the outcome.

        Args:
            *args: Unused positional arguments.
            **options: Parsed options ``study`` and ``config``.

        Raises:
            CommandError: If the config is invalid, the study does not exist
                or the study did not finish.
        """
        from abmem.models import HPOStudy
        from abmem.models.enums import HPOStatus
        from abmem.services.hpo import hpo_service

        if options.get("config") is not None:
            path: pathlib.Path = options["config"]
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CommandError(f"cannot read {path}: {error}") from error
            try:
                study = hpo_service.create_study(data)
            except ValidationError as error:
                raise CommandError(f"invalid study: {error.message_dict}") from error
            self.stdout.write(f"Created study {study.pk}")
        else:
            study = HPOStudy.objects.filter(pk=options["study"]).first()
            if study is None:
                raise CommandError(f"study {options['study']} does not exist")

        hpo_service.run_study(study.pk)
        study.refresh_from_db()
        self.stdout.write(
            f"Study {study.pk}: {study.status.name}, best_score={study.best_score}, "
            f"best_params={study.best_params}"
        )
        if study.status != HPOStatus.FINISHED:
            raise CommandError(
                f"study {study.pk} ended {study.status.name}: {study.error}"
            )
