"""Process entry points and one-time Django setup (PLAN 0.5).

This module must not import ``abmem.models`` or any ORM-dependent service at
module level, because it is imported by freshly spawned child processes
before Django is configured.
"""

import os

DEFAULT_SETTINGS_MODULE = "web_project.settings"


def setup_django(settings_module: str = DEFAULT_SETTINGS_MODULE) -> None:
    """Configure Django once for the current process.

    Sets ``DJANGO_SETTINGS_MODULE`` if it is not already set and calls
    ``django.setup()``. Repeated calls are no-ops because ``django.setup``
    only populates the app registry once.

    Args:
        settings_module: Dotted path of the settings module.
    """
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    from django.apps import apps

    if apps.ready:
        return
    import django

    django.setup()


def run_simulation_process(simulation_id: int) -> None:
    """Run one simulation in a spawned child process.

    Top-level and picklable so it can be used as a
    ``multiprocessing.Process`` target under the ``spawn`` start method.

    Args:
        simulation_id: Primary key of the (non-proxy) simulation to run.
    """
    setup_django()
    from abmem.models import Simulation
    from abmem.services.simulation import simulation_service

    simulation = Simulation.objects.get(id=simulation_id)
    simulation_service.run(simulation)


def init_hpo_worker() -> None:
    """Initialise a hyperparameter-search pool worker.

    Configures Django, limits torch to one intra-op thread so that parallel
    workers do not oversubscribe the CPU, and drops any inherited database
    connections.
    """
    setup_django()
    import torch
    from django.db import connections

    torch.set_num_threads(1)
    connections.close_all()
