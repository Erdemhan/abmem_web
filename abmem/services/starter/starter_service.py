"""Start of a simulation run in a separate process (ARCHITECTURE.md 7.1)."""

from __future__ import annotations

import multiprocessing

from django.db import connections

from ... import bootstrap
from ...models import Simulation


def start(simulation: Simulation) -> int:
    """Start a non-proxy simulation in a separate spawned process.

    The child process sets up Django itself and loads the simulation by id
    (see ``bootstrap.run_simulation_process``). Proxy simulations are
    templates and are never started.

    Args:
        simulation: Saved simulation to run.

    Returns:
        The id of the simulation.
    """
    if not simulation.proxy:
        # Inherited open connections must not be shared with the child.
        connections.close_all()
        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=bootstrap.run_simulation_process, args=(simulation.id,)
        )
        process.start()
    return simulation.id
