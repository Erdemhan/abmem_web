"""Two runs in one process share no agents or policies (PLAN 1.8, B10)."""

from __future__ import annotations

import pytest

from abmem.models import AgentStepLog
from abmem.models.enums import AlgorithmType, SimulationState
from abmem.models.models import Offer
from abmem.services.simulation import simulation_service
from tests.runtime.conftest import SMALL_DDPG, make_run, resources

pytestmark = pytest.mark.django_db


def test_back_to_back_runs_keep_offers_in_their_market(fast_demand) -> None:
    r = resources(5)
    first = make_run(
        [
            {
                "plants": [(r[0], 300)],
                "algorithm": AlgorithmType.DDPG,
                "params": SMALL_DDPG,
            },
            {"plants": [(r[1], 300), (r[2], 200)], "algorithm": AlgorithmType.RANDOM},
        ],
        period_number=30,
    )
    second = make_run(
        [
            {"plants": [(r[3], 500)], "algorithm": AlgorithmType.MARGINAL_COST},
            {
                "plants": [(r[4], 100)],
                "algorithm": AlgorithmType.SAC,
                "params": SMALL_DDPG,
            },
            {"plants": [(r[0], 250)], "algorithm": AlgorithmType.RANDOM},
        ],
        period_number=20,
    )

    simulation_service.run(first, export_json=False)
    simulation_service.run(second, export_json=False)

    for simulation, n_agents, n_plants, periods in (
        (first, 2, 3, 30),
        (second, 3, 3, 20),
    ):
        simulation.refresh_from_db()
        assert simulation.state == SimulationState.FINISHED
        offers = Offer.objects.filter(period__market=simulation.market)
        assert offers.count() == n_plants * periods
        assert (
            offers.values("agent_id").distinct().count() == n_agents
        ), "offers of foreign agents in this market"
        assert AgentStepLog.objects.filter(
            agent__market=simulation.market
        ).count() == n_agents * (periods - 1)

    mismatched = [
        offer.id
        for offer in Offer.objects.select_related("period", "agent")
        if offer.period.market_id != offer.agent.market_id
    ]
    assert mismatched == []
