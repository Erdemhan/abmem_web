"""Saving policies of finished runs and warm-starting new runs (finding A9)."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pytest
import torch

from abmem.models import AgentStepLog, HPOStudy, HPOTrial, PolicyCheckpoint, Simulation
from abmem.models.enums import HPOPhase, SimulationState
from abmem.services.algorithms import persistence
from abmem.services.market import market_service
from abmem.services.simulation import checkpoints, simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
)
from tests.factories import make_simulation
from tests.runtime.conftest import SMALL_DDPG, make_run, resources

pytestmark = pytest.mark.django_db

PERIODS = 40


@pytest.fixture
def scenario() -> Simulation:
    """A proxy scenario row that the test runs point to."""
    simulation = make_simulation(name="ckpt-sc")
    simulation.proxy = True
    simulation.save()
    return simulation


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[SimulationContext]:
    """Keep the context of every run whose checkpoints are saved."""
    contexts: list[SimulationContext] = []
    original = checkpoints.save_run_checkpoints

    def spy(ctx: SimulationContext) -> list[PolicyCheckpoint]:
        contexts.append(ctx)
        return original(ctx)

    monkeypatch.setattr(simulation_service.checkpoints, "save_run_checkpoints", spy)
    return contexts


def _agents(r: list[Any], ddpg_plants: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "plants": [(r[i], 100) for i in range(ddpg_plants)],
            "algorithm": "DDPG",
            "params": SMALL_DDPG,
        },
        {"plants": [(r[2], 50)], "algorithm": "RANDOM"},
    ]


def _run(
    scenario: Simulation,
    r: list[Any],
    *,
    source: Simulation | None = None,
    freeze: bool = False,
    ddpg_plants: int = 2,
    evaluation_periods: int = 0,
    seed: int = 7,
) -> Simulation:
    simulation = make_run(
        _agents(r, ddpg_plants),
        PERIODS,
        seed=seed,
        evaluation_periods=evaluation_periods,
    )
    simulation.scenario = scenario
    simulation.warm_start_from = source
    simulation.freeze_policies = freeze
    simulation.save()
    return simulation


def _ddpg_checkpoint(simulation: Simulation) -> PolicyCheckpoint:
    return PolicyCheckpoint.objects.get(simulation=simulation)


def test_finished_run_saves_one_checkpoint_per_learning_agent(
    fast_demand: Any, scenario: Simulation, checkpoint_dir: pathlib.Path
) -> None:
    r = resources(3)
    simulation = _run(scenario, r, evaluation_periods=5)
    simulation_service.run(simulation, export_json=False)

    rows = list(PolicyCheckpoint.objects.filter(simulation=simulation))
    assert len(rows) == 1
    row = rows[0]
    agent = simulation.market.agent_set.get(name="a0")
    assert row.agent == agent
    assert row.algorithm_type == "DDPG"
    assert row.periods_trained == PERIODS - 5
    assert row.parent is None
    assert row.format_version == persistence.FORMAT_VERSION
    path = checkpoints.checkpoint_path(row)
    assert path.is_file() and checkpoint_dir in path.parents
    assert row.file_size == path.stat().st_size
    assert row.sha256 == persistence.file_sha256(path)
    data = persistence.load_checkpoint(path, expected_sha256=row.sha256)
    assert data["meta"]["simulation_id"] == simulation.pk
    assert data["meta"]["agent_name"] == "a0"
    assert data["meta"]["periods_trained"] == PERIODS - 5
    assert len(data["meta"]["normalizer_mean"]) == 3
    assert data["spec"]["action_dim"] == 2
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    assert simulation.error == ""


def test_hpo_trial_runs_save_no_checkpoint(
    fast_demand: Any, scenario: Simulation, checkpoint_dir: pathlib.Path
) -> None:
    r = resources(3)
    simulation = _run(scenario, r)
    study = HPOStudy.objects.create(
        name="s",
        algorithm_type="DDPG",
        scenario=scenario,
        param_bounds={"tau": {"low": 0.001, "high": 0.01, "scale": "log"}},
        colony_size=2,
        max_iter=1,
        limit=5,
        search_seeds=[1],
        evaluation_seeds=[2],
        objective={"mode": "weighted", "weights": [1.0, 0.0]},
    )
    HPOTrial.objects.create(
        study=study,
        iteration=0,
        phase=HPOPhase.INIT,
        source_index=0,
        params={},
        seed=7,
        simulation=simulation,
    )
    simulation_service.run(simulation, export_json=False)
    assert not PolicyCheckpoint.objects.exists()
    assert not checkpoint_dir.exists() or not any(checkpoint_dir.rglob("*.pt"))


def test_failed_run_saves_no_checkpoint(
    fast_demand: Any,
    scenario: Simulation,
    checkpoint_dir: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = resources(3)
    simulation = _run(scenario, r)
    original = market_service.run_period

    def failing(ctx: SimulationContext, t: int) -> None:
        if t == 30:
            raise RuntimeError("boom")
        original(ctx, t)

    monkeypatch.setattr(simulation_service.market_service, "run_period", failing)
    with pytest.raises(RuntimeError):
        simulation_service.run(simulation, export_json=False)
    assert not PolicyCheckpoint.objects.exists()
    assert not checkpoint_dir.exists() or not any(checkpoint_dir.rglob("*.pt"))


def test_save_failure_keeps_the_run_finished(
    fast_demand: Any, scenario: Simulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = resources(3)
    simulation = _run(scenario, r)

    def failing_save(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(persistence, "save_checkpoint", failing_save)
    simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FINISHED
    assert "disk full" in simulation.error
    assert not PolicyCheckpoint.objects.exists()


def test_warm_started_run_acts_like_the_end_of_the_source_run(
    fast_demand: Any, scenario: Simulation, captured: list[SimulationContext]
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    trained = captured[0].agents[0].policy

    second = _run(scenario, r, source=first, seed=99)
    ctx = SimulationContext.build(second, export_json=False)
    checkpoints.apply_warm_start(ctx)
    loaded = ctx.agents[0].policy
    probe_rng = np.random.default_rng(3)
    for _ in range(5):
        obs = probe_rng.normal(size=3).astype(np.float32)
        np.testing.assert_array_equal(
            trained.select_action(obs, explore=False).u,
            loaded.select_action(obs, explore=False).u,
        )
    assert ctx.warm_start_parents == {ctx.agents[0].agent.pk: _ddpg_checkpoint(first)}
    assert ctx.warm_start_warnings == []
    assert not loaded.frozen
    assert len(loaded.buffer) == 0
    # The new run keeps the generator derived from its own seed.
    assert ctx.agents[0].rng is loaded.rng


def test_warm_started_run_records_its_origin(
    fast_demand: Any, scenario: Simulation
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    second = _run(scenario, r, source=first, seed=11)
    simulation_service.run(second, export_json=False)
    parent = _ddpg_checkpoint(first)
    child = _ddpg_checkpoint(second)
    assert child.parent == parent
    assert child.periods_trained == parent.periods_trained + PERIODS


def test_frozen_warm_start_does_not_learn(
    fast_demand: Any, scenario: Simulation, captured: list[SimulationContext]
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    second = _run(scenario, r, source=first, freeze=True, seed=12)
    simulation_service.run(second, export_json=False)
    frozen_runtime = captured[1].agents[0]
    assert frozen_runtime.evaluate_only and frozen_runtime.policy.frozen
    assert len(frozen_runtime.policy.buffer) == 0
    assert len(captured[0].agents[0].policy.buffer) > 0

    agent = second.market.agent_set.get(name="a0")
    logs = list(AgentStepLog.objects.filter(agent=agent))
    assert logs
    assert all(log.evaluation_phase for log in logs)
    assert not any(log.learning_active for log in logs)
    assert all(log.critic_loss is None for log in logs)

    parent = _ddpg_checkpoint(first)
    child = _ddpg_checkpoint(second)
    assert child.periods_trained == parent.periods_trained
    before = persistence.load_checkpoint(checkpoints.checkpoint_path(parent))
    after = persistence.load_checkpoint(checkpoints.checkpoint_path(child))
    for name, state in before["state"]["modules"].items():
        for key, tensor in state.items():
            assert torch.equal(tensor, after["state"]["modules"][name][key])


def test_shape_mismatch_starts_fresh_with_a_warning(
    fast_demand: Any, scenario: Simulation
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    second = _run(scenario, r, source=first, ddpg_plants=1)
    ctx = SimulationContext.build(second, export_json=False)
    checkpoints.apply_warm_start(ctx)
    assert ctx.warm_start_parents == {}
    assert len(ctx.warm_start_warnings) == 1
    assert "action_dim" in ctx.warm_start_warnings[0]


def test_shape_mismatch_with_freeze_fails_the_run(
    fast_demand: Any, scenario: Simulation
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    second = _run(scenario, r, source=first, ddpg_plants=1, freeze=True)
    with pytest.raises(SimulationValidationError, match="action_dim"):
        simulation_service.run(second, export_json=False)
    second.refresh_from_db()
    assert second.state == SimulationState.FAILED


def test_corrupted_checkpoint_fails_the_run(
    fast_demand: Any, scenario: Simulation
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    path = checkpoints.checkpoint_path(_ddpg_checkpoint(first))
    with path.open("ab") as handle:
        handle.write(b"\0")
    second = _run(scenario, r, source=first)
    with pytest.raises(SimulationValidationError, match="corrupted"):
        simulation_service.run(second, export_json=False)


def test_invalid_sources_are_rejected(fast_demand: Any, scenario: Simulation) -> None:
    r = resources(3)
    other = make_simulation(name="other-sc")
    other.proxy = True
    other.save()
    finished = _run(scenario, r)
    simulation_service.run(finished, export_json=False)
    target = _run(scenario, r)

    unfinished = _run(scenario, r)
    with pytest.raises(checkpoints.WarmStartError, match="FINISHED"):
        checkpoints.resolve_warm_start(target, unfinished)
    foreign = _run(other, r)
    foreign.state = SimulationState.FINISHED
    foreign.save()
    with pytest.raises(checkpoints.WarmStartError, match="scenario"):
        checkpoints.resolve_warm_start(target, foreign)
    PolicyCheckpoint.objects.all().delete()
    with pytest.raises(checkpoints.WarmStartError, match="no saved policies"):
        checkpoints.resolve_warm_start(target, finished)
    with pytest.raises(checkpoints.WarmStartError, match="scenario, not a run"):
        checkpoints.resolve_warm_start(target, scenario)


def test_agents_are_matched_by_source_then_by_name(
    fast_demand: Any, scenario: Simulation
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    target = _run(scenario, r)
    plan = checkpoints.resolve_warm_start(target, first)
    a0 = target.market.agent_set.get(name="a0")
    assert plan.matches == {a0.pk: _ddpg_checkpoint(first)}
    assert plan.unmatched == []  # the RANDOM baseline is not reported

    # Any agent row serves as the library source here.
    a0.source = first.market.agent_set.get(name="a1")
    a0.save()
    plan = checkpoints.resolve_warm_start(target, first)
    assert plan.matches == {}
    assert plan.unmatched == ["a0"]


def test_deleting_a_run_deletes_its_files(
    fast_demand: Any,
    scenario: Simulation,
    django_capture_on_commit_callbacks: Any,
) -> None:
    r = resources(3)
    first = _run(scenario, r)
    simulation_service.run(first, export_json=False)
    path = checkpoints.checkpoint_path(_ddpg_checkpoint(first))
    assert path.is_file()
    with django_capture_on_commit_callbacks(execute=True):
        first.delete()
    assert not path.exists()
    assert not path.parent.exists()
    assert not PolicyCheckpoint.objects.exists()


def test_run_without_warm_start_is_untouched(
    fast_demand: Any, scenario: Simulation, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = resources(3)
    simulation = _run(scenario, r)
    ctx = SimulationContext.build(simulation, export_json=False)
    state = ctx.agents[0].rng.bit_generator.state

    def no_load(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("no checkpoint may be read")

    monkeypatch.setattr(persistence, "load_checkpoint", no_load)
    checkpoints.apply_warm_start(ctx)
    assert ctx.agents[0].rng.bit_generator.state == state
    assert ctx.warm_start_parents == {}
    assert not any(rt.evaluate_only for rt in ctx.agents)
