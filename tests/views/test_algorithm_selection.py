"""View tests for per-agent algorithm selection and the fail-stack switch (PLAN 2.1).

Spec ``algorithm_selection_ui``: create/update scenario payloads, AgentForm
paths, the status JSON, the rendered pages, and the PLAN 2.1 validation run
(two algorithms, each run agent starts with the right policy class and
fail-stack parameters).
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from django.template import Context, Template
from django.test import Client
from django.urls import reverse

from abmem.models import Agent, Simulation
from abmem.models.enums import AgentType, AlgorithmType, EnergyType
from abmem.models.models import Resource
from abmem.services.agent.failstack import FailStackParams
from abmem.services.algorithms.DDPG import DDPGAlgorithm
from abmem.services.algorithms.ppo import PPOAlgorithm
from abmem.services.algorithms.registry import BASELINE_TYPES
from abmem.services.simulation.context import SimulationContext
from abmem.services.starter import starter_service
from abmem.templatetags.abmem_choices import algorithm_choices, failstack_defaults
from tests.factories import make_agent_with_plants, make_market, make_resource


@pytest.fixture
def fossil() -> Resource:
    """One saved fossil resource."""
    return make_resource(name="alg-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)


def _agent(
    resource: Resource, name: str, algorithm: Any = None, failstack: Any = None
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": name,
        "type": AgentType.FOSSIL.value,
        "budget": 0,
        "plants": [{"resource": resource.id, "capacity": 20000}],
    }
    if algorithm is not None:
        data["algorithm_type"] = algorithm
    if failstack is not None:
        data["failstack"] = failstack
    return data


def _payload(agents: list[dict[str, Any]], name: str = "alg-sim") -> dict[str, Any]:
    return {
        "simulation_name": name,
        "simulationMode": "1",
        "periodNumber": "5",
        "periodType": "0",
        "strategy": "0",
        "lowerBound": "0",
        "upperBound": "200",
        "agents": agents,
    }


def _post_create(client: Client, agents: list[dict[str, Any]]) -> Any:
    return client.post(
        reverse("create_simulation"),
        json.dumps(_payload(agents)),
        content_type="application/json",
    )


# create_simulation


@pytest.mark.django_db
def test_create_stores_algorithm_and_failstack(
    client: Client, fossil: Resource
) -> None:
    response = _post_create(
        client,
        [
            _agent(fossil, "ddpg", AlgorithmType.DDPG.value),
            _agent(
                fossil,
                "ppo",
                str(AlgorithmType.PPO.value),
                {"enabled": True, "limit": 3, "delta": "7.5", "min_margin": "-0.25"},
            ),
        ],
    )
    assert response.status_code == 200, response.content
    ddpg = Agent.objects.get(name="ddpg")
    ppo = Agent.objects.get(name="ppo")
    assert ddpg.algorithm_type == AlgorithmType.DDPG
    assert ddpg.failstack_enabled is False
    assert ppo.algorithm_type == AlgorithmType.PPO
    assert ppo.failstack_enabled is True
    assert ppo.failstack_limit == 3
    assert ppo.failstack_delta == Decimal("7.5")
    assert ppo.failstack_min_margin == Decimal("-0.25")


@pytest.mark.django_db
def test_create_defaults_without_algorithm_keys(
    client: Client, fossil: Resource
) -> None:
    response = _post_create(client, [_agent(fossil, "plain")])
    assert response.status_code == 200, response.content
    agent = Agent.objects.get(name="plain")
    assert agent.algorithm_type == AlgorithmType.DDPG
    assert agent.failstack_enabled is False
    assert (
        agent.failstack_limit,
        agent.failstack_delta,
        agent.failstack_min_margin,
    ) == (
        6,
        Decimal("15"),
        Decimal("0"),
    )


@pytest.mark.django_db
def test_create_accepts_algorithm_name(client: Client, fossil: Resource) -> None:
    response = _post_create(client, [_agent(fossil, "td3", "TD3")])
    assert response.status_code == 200, response.content
    assert Agent.objects.get(name="td3").algorithm_type == AlgorithmType.TD3


@pytest.mark.django_db
def test_disabled_failstack_values_are_stored(client: Client, fossil: Resource) -> None:
    response = _post_create(
        client,
        [
            _agent(
                fossil,
                "rnd",
                AlgorithmType.RANDOM.value,
                {"enabled": False, "limit": 9, "delta": 4, "min_margin": 0.5},
            )
        ],
    )
    assert response.status_code == 200, response.content
    agent = Agent.objects.get(name="rnd")
    assert agent.failstack_enabled is False
    assert agent.failstack_limit == 9
    assert agent.failstack_delta == Decimal("4")
    assert agent.failstack_min_margin == Decimal("0.5")
    assert FailStackParams.from_agent(agent) is None


@pytest.mark.django_db
@pytest.mark.parametrize("baseline", sorted(BASELINE_TYPES, key=lambda t: t.value))
def test_create_rejects_baseline_with_failstack(
    client: Client, fossil: Resource, baseline: AlgorithmType
) -> None:
    response = _post_create(
        client,
        [
            _agent(fossil, "ok", AlgorithmType.SAC.value),
            _agent(fossil, "bad", baseline.value, {"enabled": True}),
        ],
    )
    assert response.status_code == 400
    messages = response.json()["errors"]["failstack_enabled"]
    assert baseline.name in messages[0]
    assert "Agent 2 (bad)" in messages[0]
    assert not Simulation.objects.exists()
    assert not Agent.objects.exists()


@pytest.mark.django_db
@pytest.mark.parametrize(
    "algorithm,failstack,error_key",
    [
        (99, None, "algorithm_type"),
        ("NOPE", None, "algorithm_type"),
        (True, None, "algorithm_type"),
        (-1, None, "algorithm_type"),
        (0, {"enabled": True, "limit": 0}, "failstack_limit"),
        (0, {"enabled": True, "limit": -2}, "failstack_limit"),
        (0, {"enabled": True, "limit": 2.5}, "failstack_limit"),
        (0, {"enabled": True, "limit": "x"}, "failstack_limit"),
        (0, {"enabled": True, "delta": -1}, "failstack_delta"),
        (0, {"enabled": True, "delta": "NaN"}, "failstack_delta"),
        (0, {"enabled": True, "delta": "Infinity"}, "failstack_delta"),
        (0, {"enabled": True, "min_margin": "nan"}, "failstack_min_margin"),
        (0, {"enabled": True, "min_margin": "abc"}, "failstack_min_margin"),
        (0, {"enabled": True, "min_margin": 1000}, "failstack_min_margin"),
        (0, {"enabled": "maybe"}, "failstack_enabled"),
        (0, [1, 2], "failstack"),
    ],
)
def test_create_rejects_invalid_values(
    client: Client,
    fossil: Resource,
    algorithm: Any,
    failstack: Any,
    error_key: str,
) -> None:
    response = _post_create(client, [_agent(fossil, "bad", algorithm, failstack)])
    assert response.status_code == 400, response.content
    assert error_key in response.json()["errors"]
    assert not Simulation.objects.exists()


@pytest.mark.django_db
def test_create_rejects_nan_json_literal(client: Client, fossil: Resource) -> None:
    body = json.dumps(
        _payload([_agent(fossil, "bad", 0, {"enabled": True, "delta": float("nan")})])
    )
    assert "NaN" in body
    response = client.post(
        reverse("create_simulation"), body, content_type="application/json"
    )
    assert response.status_code == 400
    assert "failstack_delta" in response.json()["errors"]


# update_simulation


def _scenario(resource: Resource) -> Simulation:
    market = make_market()
    simulation = market.simulation
    simulation.proxy = True
    simulation.save()
    market.proxy = True
    market.save()
    agent = make_agent_with_plants(market, [(resource, 100)], name="stored")
    agent.algorithm_type = AlgorithmType.SAC
    agent.failstack_enabled = True
    agent.failstack_limit = 4
    agent.save()
    return simulation


@pytest.mark.django_db
def test_update_changes_algorithm(client: Client, fossil: Resource) -> None:
    simulation = _scenario(fossil)
    response = client.post(
        reverse("update_simulation", args=[simulation.id]),
        json.dumps(
            _payload(
                [
                    _agent(
                        fossil,
                        "stored",
                        AlgorithmType.RDPG.value,
                        {"enabled": True, "limit": 2, "delta": 1, "min_margin": 0},
                    )
                ]
            )
        ),
        content_type="application/json",
    )
    assert response.status_code == 200, response.content
    agent = simulation.market.agent_set.get()
    assert agent.algorithm_type == AlgorithmType.RDPG
    assert (agent.failstack_enabled, agent.failstack_limit) == (True, 2)


@pytest.mark.django_db
def test_update_rejects_baseline_failstack_atomically(
    client: Client, fossil: Resource
) -> None:
    simulation = _scenario(fossil)
    response = client.post(
        reverse("update_simulation", args=[simulation.id]),
        json.dumps(
            _payload(
                [
                    _agent(
                        fossil,
                        "changed",
                        AlgorithmType.FIXED_MARKUP.value,
                        {"enabled": True},
                    )
                ]
            )
        ),
        content_type="application/json",
    )
    assert response.status_code == 400
    assert "failstack_enabled" in response.json()["errors"]
    agent = simulation.market.agent_set.get()
    assert (agent.name, agent.algorithm_type, agent.failstack_limit) == (
        "stored",
        AlgorithmType.SAC,
        4,
    )


@pytest.mark.django_db
def test_update_page_prefills_algorithm(client: Client, fossil: Resource) -> None:
    simulation = _scenario(fossil)
    response = client.get(reverse("update_simulation", args=[simulation.id]))
    assert response.status_code == 200
    html = response.content.decode()
    assert (
        f'<option value="{AlgorithmType.SAC.value}" data-baseline="0" selected>SAC'
        in html
    )
    assert (
        'class="form-check-input agent-failstack-enabled" type="checkbox" value="on" data-default="0" checked'
        in html
    )
    assert 'class="form-control failstack-limit" value="4" data-default="6"' in html
    assert "function readAlgorithmFields" in html


# create_agent_and_related / update_agent (AgentForm)


def _form_data(resource: Resource, **extra: str) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "lib-alg",
        "budget": "100",
        "type": str(AgentType.FOSSIL.value),
        "proxy": "True",
        "resources[]": [str(resource.id)],
        "capacities[]": ["10"],
    }
    data.update(extra)
    return data


@pytest.mark.django_db
def test_agent_form_stores_algorithm(client: Client, fossil: Resource) -> None:
    response = client.post(
        reverse("agent_create"),
        _form_data(
            fossil,
            algorithm_type=str(AlgorithmType.TD3.value),
            failstack_enabled="on",
            failstack_limit="5",
            failstack_delta="2.5",
            failstack_min_margin="0.1",
        ),
    )
    assert response.status_code == 302, response.content
    agent = Agent.objects.get(name="lib-alg")
    assert agent.algorithm_type == AlgorithmType.TD3
    assert (agent.failstack_enabled, agent.failstack_limit) == (True, 5)
    assert agent.failstack_delta == Decimal("2.5")
    assert agent.failstack_min_margin == Decimal("0.1")


@pytest.mark.django_db
def test_agent_form_defaults_without_algorithm(
    client: Client, fossil: Resource
) -> None:
    response = client.post(reverse("agent_create"), _form_data(fossil))
    assert response.status_code == 302, response.content
    agent = Agent.objects.get(name="lib-alg")
    assert agent.algorithm_type == AlgorithmType.DDPG
    assert agent.failstack_enabled is False
    assert agent.failstack_limit == 6


@pytest.mark.django_db
@pytest.mark.parametrize(
    "extra,error_key",
    [
        (
            {
                "algorithm_type": str(AlgorithmType.MARGINAL_COST.value),
                "failstack_enabled": "on",
            },
            "failstack_enabled",
        ),
        ({"algorithm_type": "42"}, "algorithm_type"),
        ({"failstack_limit": "0"}, "failstack_limit"),
        ({"failstack_delta": "-3"}, "failstack_delta"),
        ({"failstack_min_margin": "nan"}, "failstack_min_margin"),
    ],
)
def test_agent_form_rejects_invalid(
    client: Client, fossil: Resource, extra: dict[str, str], error_key: str
) -> None:
    response = client.post(reverse("agent_create"), _form_data(fossil, **extra))
    assert response.status_code == 400
    assert error_key in response.json()["errors"]
    assert not Agent.objects.filter(name="lib-alg").exists()


@pytest.mark.django_db
def test_update_agent_keeps_algorithm_when_omitted(
    client: Client, fossil: Resource
) -> None:
    agent = make_agent_with_plants(make_market(), [(fossil, 7)], name="lib-old")
    agent.proxy = True
    agent.algorithm_type = AlgorithmType.PPO
    agent.save()
    response = client.post(reverse("agent_update", args=[agent.id]), _form_data(fossil))
    assert response.status_code == 302, response.content
    agent.refresh_from_db()
    assert agent.algorithm_type == AlgorithmType.PPO


@pytest.mark.django_db
def test_agent_list_shows_algorithm(client: Client, fossil: Resource) -> None:
    agent = make_agent_with_plants(make_market(), [(fossil, 7)], name="lib-list")
    agent.proxy = True
    agent.algorithm_type = AlgorithmType.RANDOM
    agent.failstack_enabled = False
    agent.save()
    response = client.get(reverse("agent_list"))
    assert response.status_code == 200
    html = response.content.decode()
    assert "RANDOM" in html
    assert 'name="algorithm_type" id="algorithm_type"' in html
    assert 'name="failstack_enabled" id="failstack_enabled"' in html
    assert "setAlgorithmFields" in html


@pytest.mark.django_db
def test_create_page_renders_algorithm_select(client: Client) -> None:
    response = client.get(reverse("create_simulation"))
    assert response.status_code == 200
    html = response.content.decode()
    for value, label, is_baseline in algorithm_choices():
        expected = f'<option value="{value}" data-baseline="{int(is_baseline)}"'
        assert expected in html, label
    assert "applyAlgorithmFields" in html


# template tags


def test_algorithm_choices_marks_baselines() -> None:
    choices = algorithm_choices()
    assert [value for value, _, _ in choices] == [t.value for t in AlgorithmType]
    baselines = {label for _, label, is_baseline in choices if is_baseline}
    assert baselines == {"RANDOM", "MARGINAL_COST", "FIXED_MARKUP"}


def test_failstack_defaults_equal_model_defaults() -> None:
    assert failstack_defaults() == {
        "failstack_limit": 6,
        "failstack_delta": "15",
        "failstack_min_margin": "0",
    }


def test_algorithm_fields_default_markup_has_no_template_literal_breakers() -> None:
    html = Template("{% load abmem_choices %}{% algorithm_fields %}").render(Context())
    assert "`" not in html and "${" not in html
    assert '<option value="0" data-baseline="0" selected>DDPG</option>' in html
    assert 'style="display: none;"' in html


# status endpoint and PLAN 2.1 validation run


class _SyncProcess:
    """Stand-in for ``multiprocessing.Process`` running the target in-process."""

    def __init__(self, target: Any, args: tuple = ()) -> None:
        self.target = target
        self.args = args

    def start(self) -> None:
        self.target(*self.args)


class _SyncContext:
    Process = _SyncProcess


@pytest.mark.django_db(transaction=True)
def test_two_algorithm_scenario_runs_with_right_policies(
    monkeypatch: pytest.MonkeyPatch, settings: Any, tmp_path: Any
) -> None:
    settings.ABMEM_SIM_DATA_DIR = str(tmp_path)
    monkeypatch.setattr(
        starter_service.multiprocessing, "get_context", lambda method: _SyncContext()
    )
    contexts: list[SimulationContext] = []
    original_build = SimulationContext.build.__func__

    def recording_build(cls: type[SimulationContext], *args: Any, **kwargs: Any) -> Any:
        ctx = original_build(cls, *args, **kwargs)
        contexts.append(ctx)
        return ctx

    monkeypatch.setattr(SimulationContext, "build", classmethod(recording_build))
    gas = make_resource(name="run-gas", energy_type=EnergyType.FOSSIL, fuel_cost=20)
    client = Client()
    response = _post_create(
        client,
        [
            _agent(gas, "ddpg", AlgorithmType.DDPG.value),
            _agent(
                gas,
                "ppo",
                AlgorithmType.PPO.value,
                {"enabled": True, "limit": 3, "delta": 10, "min_margin": 0},
            ),
        ],
    )
    assert response.status_code == 200, response.content
    proxy = Simulation.objects.get(proxy=True, name="alg-sim")

    response = client.get(reverse("run_simulation", args=[proxy.id]))
    assert response.status_code == 302

    run = Simulation.objects.get(proxy=False)
    assert run.error == ""
    # copy test: run agents carry the scenario agents' algorithm settings
    for scenario_agent in proxy.market.agent_set.all():
        run_agent = run.market.agent_set.get(name=scenario_agent.name)
        for field in (
            "algorithm_type",
            "failstack_enabled",
            "failstack_limit",
            "failstack_delta",
            "failstack_min_margin",
        ):
            assert getattr(run_agent, field) == getattr(scenario_agent, field), field

    assert len(contexts) == 1
    states = {state.agent.name: state for state in contexts[0].agents}
    assert type(states["ddpg"].policy) is DDPGAlgorithm
    assert type(states["ppo"].policy) is PPOAlgorithm
    assert states["ddpg"].failstack_params is None
    assert states["ppo"].failstack_params == FailStackParams(
        limit=3, delta=Decimal("10"), min_margin=Decimal("0")
    )

    status = client.get(reverse("get_simulation_status"), {"simulation_id": run.id})
    assert status.status_code == 200
    body = status.json()
    agents = {agent["name"]: agent for agent in body["agents"]}
    assert agents["ddpg"]["algorithm"] == "DDPG"
    assert agents["ddpg"]["failstack_enabled"] is False
    assert agents["ppo"]["algorithm"] == "PPO"
    assert agents["ppo"]["failstack_enabled"] is True
    assert set(agents["ppo"]) >= {"name", "state", "budget"}
    assert body["error"] == ""


@pytest.mark.django_db
def test_watch_page_renders_algorithm_columns(client: Client, fossil: Resource) -> None:
    market = make_market()
    response = client.get(
        reverse("watch_simulation", kwargs={"simulation_id": market.simulation.id})
    )
    assert response.status_code == 200
    html = response.content.decode()
    assert "Algorithm</th>" in html and "Fail-stack</th>" in html
    assert "agent.algorithm" in html and "agent.failstack_enabled" in html
