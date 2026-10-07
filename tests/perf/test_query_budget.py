"""Database statement budget of the period loop (PLAN 1.12, A10).

A period writes one period row, one offer ``bulk_create``, one agent
``bulk_update``, one step-log ``bulk_create`` (from period 2 on) and one
market state update, independent of the number of agents and plants. The
test database wraps each test in a transaction, so ``transaction.atomic``
adds a SAVEPOINT and a RELEASE statement to every period.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from django.db import connection
from django.test.utils import CaptureQueriesContext

from abmem.models import Agent
from abmem.models.enums import AgentState
from abmem.services.market import market_service
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import SimulationContext
from tests.runtime.conftest import (  # noqa: F401  fixtures used by name
    SMALL_DDPG,
    _restore_torch_determinism,
    constant_demand,
    fast_demand,
    make_run,
    resources,
)

pytestmark = pytest.mark.django_db

SAVEPOINT_STATEMENTS = 2
PERIOD_ONE_BUDGET = 4 + SAVEPOINT_STATEMENTS
PERIOD_BUDGET = 5 + SAVEPOINT_STATEMENTS
RUN_LOOP_STATEMENTS_PER_PERIOD = 1
PERIODS = 30


def _context(n_agents: int) -> SimulationContext:
    """Build a context with learning and baseline agents of two plants each.

    Args:
        n_agents: Number of agents.

    Returns:
        The context, no period run yet.
    """
    r = resources(n_agents + 1)
    algorithms = ["DDPG", "RANDOM", "TD3"]
    simulation = make_run(
        [
            {
                "plants": [(r[i], 300), (r[i + 1], 100)],
                "algorithm": algorithms[i % len(algorithms)],
                "params": SMALL_DDPG if i % len(algorithms) != 1 else {},
            }
            for i in range(n_agents)
        ],
        period_number=PERIODS,
    )
    return SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False
    )


@pytest.mark.parametrize("n_agents", [2, 6])
def test_statements_per_period_within_budget(n_agents: int) -> None:
    ctx = _context(n_agents)
    counts = []
    for t in range(1, PERIODS + 1):
        with CaptureQueriesContext(connection) as captured:
            market_service.run_period(ctx, t)
        counts.append(len(captured.captured_queries))
    assert counts[0] <= PERIOD_ONE_BUDGET
    assert max(counts[1:]) <= PERIOD_BUDGET


def test_period_loop_issues_no_reads() -> None:
    ctx = _context(3)
    for t in range(1, PERIODS + 1):
        with CaptureQueriesContext(connection) as captured:
            market_service.run_period(ctx, t)
        selects = [
            q["sql"] for q in captured.captured_queries if q["sql"].startswith("SELECT")
        ]
        assert selects == [], f"period {t}: {selects}"


def test_agent_state_and_budget_persisted() -> None:
    ctx = _context(3)
    for t in range(1, 4):
        market_service.run_period(ctx, t)
    for rt in ctx.agents:
        row = Agent.objects.get(pk=rt.agent.pk)
        assert row.state == AgentState.WAITING
        assert row.budget == rt.agent.budget


def test_failed_period_keeps_memory_equal_to_rows() -> None:
    ctx = _context(2)
    market_service.run_period(ctx, 1)
    budgets = [rt.agent.budget for rt in ctx.agents]

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("write failure")

    original = ctx.step_logger.flush
    ctx.step_logger.flush = _boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="write failure"):
        market_service.run_period(ctx, 2)
    ctx.step_logger.flush = original  # type: ignore[method-assign]
    assert [rt.agent.budget for rt in ctx.agents] == budgets
    for rt in ctx.agents:
        assert Agent.objects.get(pk=rt.agent.pk).budget == rt.agent.budget


def test_full_run_statement_budget(fast_demand) -> None:
    r = resources(4)
    simulation = make_run(
        [
            {"plants": [(r[i], 300), (r[i + 1], 100)], "budget": Decimal(0)}
            for i in range(3)
        ],
        period_number=PERIODS,
    )
    with CaptureQueriesContext(connection) as captured:
        simulation_service.run(simulation, export_json=False)
    per_period = PERIOD_BUDGET + RUN_LOOP_STATEMENTS_PER_PERIOD
    setup_allowance = 20
    assert len(captured.captured_queries) <= PERIODS * per_period + setup_allowance
