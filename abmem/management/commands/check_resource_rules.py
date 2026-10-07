"""Report agents whose plants break the agent-type resource rule (PLAN 2.2)."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from abmem.services.agent.resource_rules import find_violations


class Command(BaseCommand):
    """List agents that own plants whose resource is not allowed for their type."""

    help = "List agents whose plants violate the agent-type resource rule."

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Register command options.

        Args:
            parser: The argument parser of the command.
        """
        parser.add_argument(
            "--fail-on-violation",
            action="store_true",
            help="Exit with status 1 when at least one violation exists.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Print every violation and optionally fail.

        Args:
            *args: Unused positional arguments.
            **options: Parsed options, including ``fail_on_violation``.

        Raises:
            CommandError: With return code 1 when ``--fail-on-violation`` is
                set and violations exist.
        """
        violations = find_violations()
        for agent_id, agent_name, type_name, resources in violations:
            self.stdout.write(
                f"Agent {agent_id} '{agent_name}' ({type_name}): "
                f"not allowed resources: {', '.join(resources)}"
            )
        self.stdout.write(f"{len(violations)} violation(s) found.")
        if violations and options["fail_on_violation"]:
            raise CommandError(
                f"{len(violations)} resource rule violation(s).", returncode=1
            )
