"""Result fingerprint of a fixed-seed run (PLAN 1.12).

The database-write optimisations of PLAN 1.12 must not change any result.
This test runs a small fixed-seed scenario through ``simulation_service.run``
and compares offers, PTFs, budgets and step-log rows with
``fingerprint_small_run.json``, which was produced by the code before the
optimisation (commit 72b5552).

Floats are stored with ``repr`` and Decimals with ``str``, so the comparison
is exact. The learning policies run on CPU with the deterministic seeding of
``seed_everything``. A different CPU, BLAS build or torch version may change
the last bits of the network outputs. In that case regenerate the fixture
with ``ABMEM_REGEN_FINGERPRINT=1`` on a commit whose results are trusted.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import pytest

from abmem.models import AgentStepLog, Simulation
from abmem.models.enums import AlgorithmType, MarketStrategy
from abmem.models.models import Offer, Period
from abmem.services.simulation import simulation_service
from tests.runtime.conftest import SMALL_DDPG, make_run, resources

pytestmark = pytest.mark.django_db

FIXTURE = pathlib.Path(__file__).with_name("fingerprint_small_run.json")


def _optional_str(value: Any) -> str | None:
    """Return ``str(value)``, or None for None.

    Args:
        value: Any value.

    Returns:
        The string or None.
    """
    return None if value is None else str(value)


def _optional_repr(value: float | None) -> str | None:
    """Return ``repr(float(value))``, or None for None.

    Args:
        value: Float or None.

    Returns:
        The exact float representation or None.
    """
    return None if value is None else repr(float(value))


def _scenario() -> Simulation:
    """Create the fixed scenario: learning, baseline and fail-stack agents.

    Returns:
        The saved run in state CREATED.
    """
    r = resources(4)
    return make_run(
        [
            {
                "plants": [(r[0], 300), (r[1], 200)],
                "algorithm": AlgorithmType.DDPG,
                "params": SMALL_DDPG,
                "failstack": {"failstack_enabled": True, "failstack_limit": 2},
            },
            {"plants": [(r[2], 400)], "algorithm": AlgorithmType.RANDOM},
            {
                "plants": [(r[1], 150)],
                "algorithm": AlgorithmType.TD3,
                "params": SMALL_DDPG,
            },
            {"plants": [(r[3], 250)], "algorithm": AlgorithmType.MARGINAL_COST},
        ],
        period_number=40,
        seed=21,
        strategy=MarketStrategy.PAYASBID,
        evaluation_periods=6,
    )


def fingerprint(simulation: Simulation) -> dict[str, Any]:
    """Collect every result of a run in an id-free, exactly comparable form.

    Agents are identified by name and plants by their position in the
    agent's offers of a period (plant order), so database ids do not enter.

    Args:
        simulation: Finished run.

    Returns:
        JSON-serialisable dict with keys ``periods``, ``offers``,
        ``budgets`` and ``step_logs``.
    """
    market = simulation.market
    periods = [
        [p.periodNumber, str(p.demand), str(p.metDemand), p.marketVolume, str(p.ptf)]
        for p in Period.objects.filter(market=market).order_by("periodNumber")
    ]
    offers = [
        [
            o.period.periodNumber,
            o.agent.name,
            o.amount,
            str(o.offerPrice),
            bool(o.acceptance),
            _optional_str(o.acceptancePrice),
            _optional_str(o.acceptanceAmount),
        ]
        for o in Offer.objects.filter(period__market=market)
        .select_related("period", "agent")
        .order_by("period__periodNumber", "agent__name", "id")
    ]
    budgets = [
        [name, str(budget)]
        for name, budget in market.agent_set.order_by("name").values_list(
            "name", "budget"
        )
    ]
    step_logs = [
        [
            row.period.periodNumber,
            row.agent.name,
            _optional_repr(row.reward),
            _optional_repr(row.critic_loss),
            _optional_repr(row.actor_loss),
            _optional_repr(row.mean_q),
            _optional_repr(row.exploration_scale),
            list(row.intervened_plants),
            bool(row.learning_active),
            bool(row.evaluation_phase),
            {k: repr(v) if isinstance(v, float) else v for k, v in row.extra.items()},
        ]
        for row in AgentStepLog.objects.filter(period__market=market)
        .select_related("period", "agent")
        .order_by("period__periodNumber", "agent__name")
    ]
    return {
        "periods": periods,
        "offers": offers,
        "budgets": budgets,
        "step_logs": step_logs,
    }


def test_fixed_seed_results_match_stored_fingerprint(fast_demand) -> None:
    simulation = _scenario()
    simulation_service.run(simulation, export_json=False)
    actual = json.loads(json.dumps(fingerprint(simulation)))
    if os.environ.get("ABMEM_REGEN_FINGERPRINT") == "1":
        FIXTURE.write_text(json.dumps(actual, indent=1) + "\n", encoding="utf-8")
        pytest.skip(f"fingerprint regenerated at {FIXTURE}")
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert actual["periods"] == expected["periods"]
    assert actual["budgets"] == expected["budgets"]
    assert actual["offers"] == expected["offers"]
    assert actual["step_logs"] == expected["step_logs"]
