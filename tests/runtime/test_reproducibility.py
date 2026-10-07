"""Same seed gives identical runs on CPU, different seeds differ (PLAN 1.10, C9)."""

from __future__ import annotations

import numpy as np
import pytest

from abmem.models.models import Offer, Period
from abmem.services.simulation import simulation_service
from abmem.services.simulation.seeding import seed_everything, spawn_agent_rngs
from tests.runtime.conftest import SMALL_DDPG, make_run, resources

pytestmark = pytest.mark.django_db


def _scenario(r, seed: int):
    return make_run(
        [
            {
                "plants": [(r[0], 300), (r[1], 200)],
                "algorithm": "DDPG",
                "params": SMALL_DDPG,
            },
            {"plants": [(r[2], 400)], "algorithm": "RANDOM"},
            {
                "plants": [(r[1], 150)],
                "algorithm": "TD3",
                "params": SMALL_DDPG,
                "failstack": {"failstack_enabled": True, "failstack_limit": 2},
            },
        ],
        period_number=45,
        seed=seed,
    )


def _trace(simulation) -> tuple[list, list, list]:
    market = simulation.market
    prices = list(
        Offer.objects.filter(period__market=market)
        .order_by("period__periodNumber", "agent__name", "resource_id", "id")
        .values_list("offerPrice", flat=True)
    )
    ptfs = list(
        Period.objects.filter(market=market)
        .order_by("periodNumber")
        .values_list("ptf", flat=True)
    )
    budgets = list(market.agent_set.order_by("name").values_list("budget", flat=True))
    return prices, ptfs, budgets


def test_same_seed_identical_results(fast_demand) -> None:
    r = resources(3)
    a = _scenario(r, seed=11)
    b = _scenario(r, seed=11)
    simulation_service.run(a, export_json=False)
    simulation_service.run(b, export_json=False)
    assert _trace(a) == _trace(b)


def test_different_seed_changes_random_prices(fast_demand) -> None:
    r = resources(3)
    a = _scenario(r, seed=11)
    b = _scenario(r, seed=12)
    simulation_service.run(a, export_json=False)
    simulation_service.run(b, export_json=False)
    random_prices = []
    for simulation in (a, b):
        random_prices.append(
            list(
                Offer.objects.filter(period__market=simulation.market, agent__name="a1")
                .order_by("period__periodNumber")
                .values_list("offerPrice", flat=True)
            )
        )
    assert random_prices[0] != random_prices[1]


def test_spawn_agent_rngs_is_deterministic_and_independent() -> None:
    first = [g.random(4) for g in spawn_agent_rngs(5, 3)]
    second = [g.random(4) for g in spawn_agent_rngs(5, 3)]
    for x, y in zip(first, second):
        assert np.array_equal(x, y)
    assert not np.array_equal(first[0], first[1])
    assert spawn_agent_rngs(5, 0) == []


@pytest.mark.parametrize("bad", [-1, 2**32, 1.5, True])
def test_seed_validation(bad) -> None:
    with pytest.raises(ValueError):
        seed_everything(bad, deterministic=False)
    with pytest.raises(ValueError):
        spawn_agent_rngs(bad, 1)


def test_seed_everything_seeds_global_generators() -> None:
    import random

    import torch

    seed_everything(3, deterministic=False)
    a = (random.random(), np.random.random(), torch.rand(1).item())
    seed_everything(3, deterministic=False)
    b = (random.random(), np.random.random(), torch.rand(1).item())
    assert a == b
