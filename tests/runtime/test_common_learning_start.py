"""Common learning start of the learning agents of a run (PLAN 2.14)."""

from __future__ import annotations

from typing import Any

import pytest

from abmem.models import Agent
from abmem.models.diagnostics import AgentStepLog
from abmem.services.algorithms.algorithm_config import (
    DDPGConfig,
    PPOConfig,
    RDPGConfig,
)
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import SimulationContext, SimulationValidationError
from tests.runtime.conftest import make_run, resources

pytestmark = pytest.mark.django_db

LEARNERS = ["DDPG", "TD3", "SAC", "PPO", "RDPG"]


def _run(
    params: dict[str, dict[str, Any]] | None = None,
    *,
    learning_starts: int | None = None,
    periods: int = 260,
    lags: tuple[int, ...] = (1, 24),
    extra: list[str] | None = None,
):
    algorithms = LEARNERS + (extra or [])
    r = resources(len(algorithms))
    simulation = make_run(
        [
            {"plants": [(r[i], 300)], "algorithm": a, "params": (params or {}).get(a, {})}
            for i, a in enumerate(algorithms)
        ],
        period_number=periods,
        observation_lags=lags,
    )
    simulation.learning_starts = learning_starts
    simulation.save(update_fields=["learning_starts"])
    return simulation


def _first_updates(simulation) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for agent in Agent.objects.filter(market__simulation=simulation).order_by("id"):
        first = (
            AgentStepLog.objects.filter(agent=agent)
            .exclude(critic_loss=None, actor_loss=None)
            .order_by("period__periodNumber")
            .values_list("period__periodNumber", flat=True)
            .first()
        )
        result[str(agent.algorithm_type)] = first
    return result


# ------------------------------------------------------------------ configs


def test_first_update_transitions_per_config() -> None:
    assert DDPGConfig(learning_starts=80, batch_size=64).first_update_transitions() == 80
    assert RDPGConfig(learning_starts=8, batch_size=8, seq_len=24).first_update_transitions() == 24
    assert PPOConfig(learning_starts=8, batch_size=8, minibatch_size=8, rollout_length=64).first_update_transitions() == 64
    assert DDPGConfig().learning_starts_limit() == DDPGConfig().buffer_size
    assert PPOConfig().learning_starts_limit() is None


# ------------------------------------------------------------------ runs


@pytest.mark.parametrize(("lags", "expected"), [((1, 24), 89), ((1, 24, 168), 233)])
def test_defaults_keep_the_same_first_update(fast_demand, lags, expected) -> None:
    simulation = _run(lags=lags, periods=expected + 20)
    simulation_service.run(simulation, export_json=False)
    assert set(_first_updates(simulation).values()) == {expected}


def test_automatic_value_is_the_largest_requirement(fast_demand) -> None:
    small = {"learning_starts": 8, "batch_size": 8}
    params = {a: dict(small) for a in LEARNERS}
    params["PPO"] = {**small, "minibatch_size": 8}  # rollout_length stays 64
    simulation = _run(params)
    simulation_service.run(simulation, export_json=False)
    # Largest requirement is PPO's rollout of 64 transitions, first stored
    # transition in period 26, so every agent first updates in period 89.
    assert set(_first_updates(simulation).values()) == {89}


def test_explicit_value_applies_to_every_algorithm(fast_demand) -> None:
    simulation = _run(learning_starts=100, periods=200)
    simulation_service.run(simulation, export_json=False)
    assert set(_first_updates(simulation).values()) == {125}


def test_snapshot_records_the_common_value(fast_demand) -> None:
    simulation = _run(learning_starts=100, periods=30, extra=["MARGINAL_COST"])
    SimulationContext.build(simulation, export_json=False)
    for agent in Agent.objects.filter(market__simulation=simulation):
        if str(agent.algorithm_type) in LEARNERS:
            assert agent.effective_params["learning_starts"] == 100
        else:
            assert agent.effective_params.get("learning_starts") != 100


def test_value_below_a_requirement_is_rejected(fast_demand) -> None:
    simulation = _run(learning_starts=30, periods=30)
    with pytest.raises(SimulationValidationError, match="learning start 30"):
        SimulationContext.build(simulation, export_json=False)


def test_value_above_a_buffer_is_rejected(fast_demand) -> None:
    simulation = _run({"DDPG": {"buffer_size": 128}}, learning_starts=200, periods=30)
    with pytest.raises(SimulationValidationError, match="DDPG"):
        SimulationContext.build(simulation, export_json=False)
