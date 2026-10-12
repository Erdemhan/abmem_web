"""HPO overrides restricted to one algorithm key (PLAN 2.15, target MATCHING)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    agent_params,
    overrides_apply,
)
from tests.runtime.conftest import SMALL_DDPG, make_run, resources

pytestmark = pytest.mark.django_db


@pytest.mark.parametrize(
    ("algorithm", "only", "expected"),
    [
        ("DDPG", None, True),
        ("DDPG", "DDPG", True),
        ("FIXED_MARKUP", "DDPG", False),
        ("", "DDPG", False),
    ],
)
def test_overrides_apply(algorithm: str, only: str | None, expected: bool) -> None:
    assert overrides_apply(algorithm, only) is expected


def test_agent_params_with_override_algorithm() -> None:
    agent = SimpleNamespace(algorithm_type="FIXED_MARKUP", algorithm_params={"markup": 0.3})
    defaults = {"FIXED_MARKUP": {"markup": 0.1}, "DDPG": {"gamma": 0.99}}
    # Restricted to another algorithm: the agent keeps its own parameters.
    assert agent_params(
        agent, "FIXED_MARKUP", defaults, {"markup": 0.9}, override_algorithm="DDPG"
    ) == {"markup": 0.3}
    # Restricted to the agent's algorithm: the overrides win.
    assert agent_params(
        agent,
        "FIXED_MARKUP",
        defaults,
        {"markup": 0.9},
        override_algorithm="FIXED_MARKUP",
    ) == {"markup": 0.9}
    # Not restricted: unchanged behaviour.
    assert agent_params(agent, "FIXED_MARKUP", defaults, {"markup": 0.9}) == {
        "markup": 0.9
    }
    assert agent_params(agent, "FIXED_MARKUP", defaults, None, override_algorithm="DDPG") == {
        "markup": 0.3
    }


def test_build_applies_overrides_only_to_the_override_algorithm(fast_demand) -> None:
    r = resources(2)
    simulation = make_run(
        [
            {"plants": [(r[0], 100)], "algorithm": "DDPG", "params": dict(SMALL_DDPG)},
            {
                "plants": [(r[1], 100)],
                "algorithm": "FIXED_MARKUP",
                "params": {"markup": 0.3},
            },
        ]
    )
    ctx = SimulationContext.build(
        simulation, {"gamma": 0.8}, override_algorithm="DDPG"
    )
    ddpg, fixed = ctx.agents
    assert ddpg.policy.key == "DDPG" and ddpg.policy.config.gamma == 0.8
    assert fixed.policy.key == "FIXED_MARKUP"
    assert fixed.policy.config.markup == 0.3
    assert fixed.agent.effective_params["gamma"] != 0.8


def test_run_passes_override_algorithm(fast_demand) -> None:
    r = resources(2)
    simulation = make_run(
        [
            {"plants": [(r[0], 100)], "algorithm": "DDPG", "params": dict(SMALL_DDPG)},
            {
                "plants": [(r[1], 100)],
                "algorithm": "FIXED_MARKUP",
                "params": {"markup": 0.3},
            },
        ],
        period_number=3,
    )
    simulation_service.run(
        simulation,
        algorithm_overrides={"gamma": 0.8},
        override_algorithm="DDPG",
        export_json=False,
    )
    a0, a1 = simulation.market.agent_set.order_by("id")
    assert a0.effective_params["gamma"] == 0.8
    assert a1.effective_params["markup"] == 0.3
    assert a1.effective_params["gamma"] != 0.8
