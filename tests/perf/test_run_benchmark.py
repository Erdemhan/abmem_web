"""Wall-time benchmark of a full run (PLAN 1.12): 8 agents x 336 periods.

Skipped in the default run. Enable with ``ABMEM_RUN_SLOW=1``. The measured
time is printed and attached to the test report as a user property, it is
not compared against a threshold.
"""

from __future__ import annotations

import time
from decimal import Decimal

import pytest

from abmem.models.enums import SimulationState
from abmem.services.simulation import simulation_service
from tests.runtime.conftest import make_run, resources

pytestmark = [pytest.mark.slow, pytest.mark.django_db]

ALGORITHMS = [
    "DDPG",
    "TD3",
    "SAC",
    "PPO",
    "RDPG",
    "RANDOM",
    "MARGINAL_COST",
    "FIXED_MARKUP",
]


def test_run_benchmark_8_agents_336_periods(record_property) -> None:
    r = resources(8)
    simulation = make_run(
        [
            {
                "plants": [(r[i], 5000 + 100 * i)],
                "algorithm": algorithm,
                "budget": Decimal(0),
            }
            for i, algorithm in enumerate(ALGORITHMS)
        ],
        period_number=336,
    )
    start = time.perf_counter()
    simulation_service.run(simulation, export_json=False)
    elapsed = time.perf_counter() - start
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    record_property("wall_time_s", round(elapsed, 3))
    print(f"\n8 agents x 336 periods: {elapsed:.2f} s")
