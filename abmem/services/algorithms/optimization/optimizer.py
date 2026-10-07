"""Command-line entry point for the hyperparameter search (PLAN 2.4).

Thin wrapper around the ``run_hpo`` management command. Run from the
repository root:

    python -m abmem.services.algorithms.optimization.optimizer --study 12
    python -m abmem.services.algorithms.optimization.optimizer --config study.json
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from abmem.bootstrap import setup_django


def main(argv: Sequence[str] | None = None) -> None:
    """Configure Django and run ``run_hpo`` with the given arguments.

    Args:
        argv: Command arguments, ``sys.argv[1:]`` when None.
    """
    setup_django()
    from django.core.management import call_command

    args = list(sys.argv[1:] if argv is None else argv)
    call_command("run_hpo", *args)


if __name__ == "__main__":
    main()
