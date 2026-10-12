"""Tests of the agent library service (agent_library spec, package service)."""

from __future__ import annotations

import dataclasses
from decimal import Decimal
from typing import Any

import pytest
from django.db import DataError, connection, transaction

from abmem.models import Agent, Market, Simulation
from abmem.models.enums import (
    AgentState,
    AgentType,
    EnergyType,
    MarketState,
    SimulationState,
)
from abmem.models.models import Plant, Portfolio, Resource
from abmem.services.agent import library_service as ls
from abmem.services.agent.library_service import (
    COPIED_AGENT_FIELDS,
    NAME_MAX_LENGTH,
    NOT_COPIED_AGENT_FIELDS,
    AgentInput,
    LibraryValidationError,
    PlantInput,
)
from tests.factories import make_agent_with_plants, make_market, make_resource

pytestmark = pytest.mark.django_db


@pytest.fixture
def fossil() -> Resource:
    return make_resource(energy_type=EnergyType.FOSSIL)


@pytest.fixture
def renewable() -> Resource:
    return make_resource(energy_type=EnergyType.RENEWABLE)


def _input(resource: Resource, **overrides: Any) -> AgentInput:
    values: dict[str, Any] = {
        "name": "Coal",
        "budget": Decimal("1000.50"),
        "type": AgentType.FOSSIL,
        "algorithm_type": "TD3",
        "failstack_enabled": True,
        "failstack_limit": 4,
        "failstack_delta": Decimal("12.50"),
        "failstack_min_margin": Decimal("0.0500"),
        "plants": (PlantInput(resource.id, 300), PlantInput(resource.id, 150)),
        "algorithm_params": {"actor_lr": 0.001},
    }
    values.update(overrides)
    return AgentInput(**values)


def _scenario_market() -> Market:
    market = make_market()
    market.simulation.proxy = True
    market.simulation.save()
    market.proxy = True
    market.save()
    return market


def _plants(agent: Agent) -> list[tuple[int, int]]:
    return sorted(
        Plant.objects.filter(portfolio__agent=agent).values_list(
            "resource_id", "capacity"
        )
    )


def _codes(exc: LibraryValidationError, field: str) -> list[str]:
    return [error.code for error in exc.error_dict[field]]


# ---------------------------------------------------------------- field list


def test_copied_fields_cover_every_concrete_agent_field() -> None:
    concrete = {field.name for field in Agent._meta.concrete_fields}
    assert set(COPIED_AGENT_FIELDS) == concrete - NOT_COPIED_AGENT_FIELDS
    assert NOT_COPIED_AGENT_FIELDS <= concrete


# ------------------------------------------------------------- name helpers


def test_normalize_name_strips_only_whitespace() -> None:
    assert ls.normalize_name("  Coal Plant \t") == "Coal Plant"


def test_check_name_available_empty_is_false() -> None:
    assert ls.check_name_available("   ") is False


def test_check_name_available_library_case_insensitive(fossil: Resource) -> None:
    agent = ls.create_library_agent(_input(fossil, name="Coal"))
    assert ls.check_name_available("COAL") is False
    assert ls.check_name_available(" coal ") is False
    assert ls.check_name_available("coal", exclude_id=agent.id) is True
    assert ls.check_name_available("Gas") is True


def test_check_name_available_market_case_sensitive() -> None:
    market = make_market()
    make_agent_with_plants(market, name="CCGT")
    assert ls.check_name_available("CCGT", market=market) is False
    assert ls.check_name_available("ccgt", market=market) is True
    assert ls.check_name_available("CCGT") is True


def test_name_check_uses_database_lower_ascii() -> None:
    Agent.objects.create(name="ABC", market=None, proxy=True)
    assert ls.check_name_available("abc") is False
    assert ls.check_name_available("aBc") is False


def test_name_check_matches_database_lower_for_dotted_capital_i() -> None:
    """The service accepts 'izmir' exactly when the database LOWER('İzmir') != 'izmir'."""
    try:
        with transaction.atomic(), connection.cursor() as cursor:
            cursor.execute("SELECT LOWER(%s)", ["İzmir"])
            db_lower = cursor.fetchone()[0]
    except DataError:
        pytest.skip("The test database encoding cannot represent U+0130.")
    Agent.objects.create(name="İzmir", market=None, proxy=True)
    assert ls.check_name_available("izmir") is (db_lower != "izmir")


# ------------------------------------------------------- create_library_agent


def test_create_library_agent(fossil: Resource) -> None:
    agent = ls.create_library_agent(_input(fossil, name="  Coal  "))
    agent.refresh_from_db()
    assert agent.is_library
    assert agent.market_id is None and agent.proxy is True
    assert agent.name == "Coal"
    assert agent.budget == Decimal("1000.50")
    assert agent.type == AgentType.FOSSIL
    assert agent.algorithm_type == "TD3"
    assert agent.failstack_enabled is True
    assert agent.failstack_limit == 4
    assert agent.failstack_delta == Decimal("12.50")
    assert agent.failstack_min_margin == Decimal("0.0500")
    assert agent.algorithm_params == {"actor_lr": 0.001}
    assert agent.source_id is None
    assert agent.portfolio.proxy is True
    assert _plants(agent) == [(fossil.id, 150), (fossil.id, 300)]


@pytest.mark.parametrize("variant", ["Coal", "COAL", "coal", " cOaL "])
def test_duplicate_library_name_rejected(fossil: Resource, variant: str) -> None:
    ls.create_library_agent(_input(fossil, name="Coal"))
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, name=variant))
    assert _codes(info.value, "name") == ["name_taken"]
    assert Agent.objects.filter(market__isnull=True).count() == 1


def test_library_name_may_equal_scenario_agent_name(fossil: Resource) -> None:
    make_agent_with_plants(make_market(), name="Coal")
    agent = ls.create_library_agent(_input(fossil, name="Coal"))
    assert agent.is_library


@pytest.mark.parametrize(
    ("name", "code"), [("", "required"), ("   ", "required"), ("x" * 21, "max_length")]
)
def test_invalid_name_rejected(fossil: Resource, name: str, code: str) -> None:
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, name=name))
    assert _codes(info.value, "name") == [code]
    assert not Agent.objects.exists()


def test_name_of_exactly_20_characters_accepted(fossil: Resource) -> None:
    agent = ls.create_library_agent(_input(fossil, name="x" * 20))
    assert len(agent.name) == 20


def test_resource_rule_violation_rejected(
    fossil: Resource, renewable: Resource
) -> None:
    data = _input(
        fossil, plants=(PlantInput(fossil.id, 10), PlantInput(renewable.id, 10))
    )
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(data)
    assert _codes(info.value, "plants") == ["resource_not_allowed"]
    assert not Agent.objects.exists()
    assert not Portfolio.objects.exists()


def test_hybrid_agent_may_own_every_energy_type(
    fossil: Resource, renewable: Resource
) -> None:
    data = _input(
        fossil,
        type=AgentType.HYBRID,
        plants=(PlantInput(fossil.id, 10), PlantInput(renewable.id, 10)),
    )
    assert len(_plants(ls.create_library_agent(data))) == 2


@pytest.mark.parametrize(
    ("plants", "code"),
    [
        ((), "no_plants"),
        ("zero", "invalid_capacity"),
        ("negative", "invalid_capacity"),
        ("unknown", "unknown_resource"),
    ],
)
def test_invalid_plants_rejected(fossil: Resource, plants: Any, code: str) -> None:
    rows = {
        "zero": (PlantInput(fossil.id, 0),),
        "negative": (PlantInput(fossil.id, -5),),
        "unknown": (PlantInput(fossil.id + 9999, 10),),
    }.get(plants, plants)
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, plants=rows))
    assert _codes(info.value, "plants") == [code]
    assert not Agent.objects.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failstack_limit", 0),
        ("failstack_delta", Decimal("-1")),
        ("failstack_delta", Decimal("NaN")),
        ("failstack_min_margin", Decimal("Infinity")),
        ("budget", Decimal("NaN")),
        ("type", 99),
        ("algorithm_type", 99),
    ],
)
def test_invalid_scalar_fields_rejected(
    fossil: Resource, field: str, value: Any
) -> None:
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, **{field: value}))
    assert field in info.value.error_dict
    assert not Agent.objects.exists()


def test_decimal_digit_limits_rejected_before_database(fossil: Resource) -> None:
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, failstack_min_margin=Decimal("100")))
    assert "failstack_min_margin" in info.value.error_dict
    assert not Agent.objects.exists()


def test_library_validation_error_is_validation_error() -> None:
    from django.core.exceptions import ValidationError

    assert issubclass(LibraryValidationError, ValidationError)


def test_integrity_race_converted_to_name_error(
    fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    Agent.objects.create(name="Coal", market=None, proxy=True)
    original = ls.check_name_available
    calls = {"n": 0}

    def stale_first_check(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        return True if calls["n"] == 1 else original(*args, **kwargs)

    monkeypatch.setattr(ls, "check_name_available", stale_first_check)
    with pytest.raises(LibraryValidationError) as info:
        ls.create_library_agent(_input(fossil, name="coal"))
    assert _codes(info.value, "name") == ["name_taken"]
    assert Agent.objects.count() == 1


def test_transaction_rollback_leaves_no_partial_rows(
    fossil: Resource, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_bulk_create(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("plant insert failed")

    monkeypatch.setattr(Plant.objects, "bulk_create", failing_bulk_create)
    with pytest.raises(RuntimeError):
        ls.create_library_agent(_input(fossil))
    assert not Agent.objects.exists()
    assert not Portfolio.objects.exists()


# ------------------------------------------------------- update_library_agent


def test_update_library_agent_replaces_plants(fossil: Resource) -> None:
    agent = ls.create_library_agent(_input(fossil))
    other = make_resource(energy_type=EnergyType.FOSSIL)
    ls.update_library_agent(
        agent,
        _input(
            fossil,
            name="coal",
            budget=Decimal("5"),
            plants=(PlantInput(other.id, 7),),
            algorithm_params=None,
        ),
    )
    agent.refresh_from_db()
    assert agent.name == "coal"
    assert agent.budget == Decimal("5")
    assert agent.algorithm_params == {"actor_lr": 0.001}
    assert _plants(agent) == [(other.id, 7)]


def test_update_library_agent_name_conflict(fossil: Resource) -> None:
    ls.create_library_agent(_input(fossil, name="Gas"))
    agent = ls.create_library_agent(_input(fossil, name="Coal"))
    with pytest.raises(LibraryValidationError):
        ls.update_library_agent(agent, _input(fossil, name="GAS"))


def test_update_rejects_non_library_agent(fossil: Resource) -> None:
    agent = make_agent_with_plants(make_market(), [(fossil, 10)])
    with pytest.raises(ValueError):
        ls.update_library_agent(agent, _input(fossil))


# ----------------------------------------------------------------- copy_agent


def test_copy_agent_is_complete(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    market = _scenario_market()
    copy = ls.copy_agent(library, market, proxy=True)
    copy.refresh_from_db()
    for field in COPIED_AGENT_FIELDS:
        assert getattr(copy, field) == getattr(library, field), field
    assert copy.id != library.id
    assert copy.market_id == market.id
    assert copy.proxy is True
    assert copy.state == AgentState.CREATED
    assert copy.source_id == library.id
    assert copy.portfolio.id != library.portfolio.id
    assert copy.portfolio.proxy is True
    assert _plants(copy) == _plants(library)


def test_copy_agent_resets_state_and_custom_name(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    library.state = AgentState.RUNNING
    library.save()
    copy = ls.copy_agent(library, make_market(), proxy=False, name=" Renamed ")
    assert copy.state == AgentState.CREATED
    assert copy.name == "Renamed"
    assert copy.proxy is False
    assert copy.portfolio.plant_set.filter(proxy=True).count() == 0


def test_copy_is_independent_of_later_library_edits(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    copy = ls.copy_agent(library, _scenario_market(), proxy=True)
    before = (
        {f: getattr(copy, f) for f in COPIED_AGENT_FIELDS},
        _plants(copy),
    )
    other = make_resource(energy_type=EnergyType.FOSSIL)
    ls.update_library_agent(
        library,
        _input(
            fossil,
            name="Changed",
            budget=Decimal("1"),
            algorithm_type="SAC",
            failstack_limit=9,
            plants=(PlantInput(other.id, 1),),
            algorithm_params={"actor_lr": 0.5},
        ),
    )
    copy.refresh_from_db()
    assert ({f: getattr(copy, f) for f in COPIED_AGENT_FIELDS}, _plants(copy)) == before


def test_algorithm_params_are_not_shared(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    copy = ls.copy_agent(library, _scenario_market(), proxy=True)
    copy.algorithm_params["actor_lr"] = 9
    assert library.algorithm_params == {"actor_lr": 0.001}


def test_source_chain(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    scenario_copy = ls.copy_agent(library, _scenario_market(), proxy=True)
    run_copy = ls.copy_agent(scenario_copy, make_market(), proxy=False)
    second_run = ls.copy_agent(run_copy, make_market(), proxy=False)
    assert scenario_copy.source_id == library.id
    assert run_copy.source_id == library.id
    assert second_run.source_id == library.id


def test_copy_of_agent_without_source_has_no_source(fossil: Resource) -> None:
    agent = make_agent_with_plants(make_market(), [(fossil, 10)])
    copy = ls.copy_agent(agent, make_market(), proxy=False)
    assert copy.source_id is None
    assert _plants(copy) == [(fossil.id, 10)]


def test_library_deletion_keeps_copies(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    copy = ls.copy_agent(library, _scenario_market(), proxy=True)
    library.delete()
    copy.refresh_from_db()
    assert copy.source_id is None
    assert _plants(copy) == [(fossil.id, 150), (fossil.id, 300)]


@pytest.mark.parametrize("times", [2, 3])
def test_same_library_agent_copied_several_times(fossil: Resource, times: int) -> None:
    """OQ-14 (2026-10-12): every copy is a scenario agent with its own name."""
    library = ls.create_library_agent(_input(fossil))
    market = _scenario_market()
    copies = [ls.copy_agent(library, market, proxy=True) for _ in range(times)]
    assert [copy.name for copy in copies] == ["Coal", "Coal_2", "Coal_3"][:times]
    assert {copy.source_id for copy in copies} == {library.id}
    for copy in copies:
        assert _plants(copy) == _plants(library)
    assert Agent.objects.filter(market=market, source=library).count() == times


def test_copy_with_given_name_next_to_an_earlier_copy(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    market = _scenario_market()
    ls.copy_agent(library, market, proxy=True)
    other = ls.copy_agent(library, market, proxy=True, name="Other")
    assert other.name == "Other"
    assert other.source_id == library.id


def test_copy_without_name_takes_free_name_case_insensitively(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    market = _scenario_market()
    make_agent_with_plants(market, [(fossil, 1)], name="coal")
    make_agent_with_plants(market, [(fossil, 1)], name="COAL_2")
    copy = ls.copy_agent(library, market, proxy=True)
    assert copy.name == "Coal_3"


def test_copy_name_conflict_in_market_rejected(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    market = _scenario_market()
    make_agent_with_plants(market, [(fossil, 1)], name="Coal")
    with pytest.raises(LibraryValidationError) as info:
        ls.copy_agent(library, market, proxy=True, name="Coal")
    assert _codes(info.value, "name") == ["name_taken"]
    assert Agent.objects.filter(market=market).count() == 1


def test_copy_name_shortened_at_length_limit(fossil: Resource) -> None:
    long_name = "L" * NAME_MAX_LENGTH
    library = ls.create_library_agent(_input(fossil, name=long_name))
    market = _scenario_market()
    names = [ls.copy_agent(library, market, proxy=True).name for _ in range(3)]
    assert names == [
        long_name,
        "L" * (NAME_MAX_LENGTH - 2) + "_2",
        "L" * (NAME_MAX_LENGTH - 2) + "_3",
    ]
    assert all(len(name) <= NAME_MAX_LENGTH for name in names)


@pytest.mark.parametrize(
    ("base", "taken", "expected"),
    [
        ("Gas", [], "Gas"),
        ("Gas", ["gas"], "Gas_2"),
        ("Gas", ["Gas", "GAS_2", "gas_3"], "Gas_4"),
        (" Gas ", ["Gas "], "Gas_2"),
        ("A" * 19 + " ", ["A" * 19], "A" * 18 + "_2"),
        ("B" * 20, ["B" * 20] + [f"{'B' * 18}_{n}" for n in range(2, 10)], "B" * 17 + "_10"),
        ("Ab c", ["ab c"], "Ab c_2"),
    ],
)
def test_free_agent_name(base: str, taken: list[str], expected: str) -> None:
    assert ls.free_agent_name(base, taken) == expected
    assert len(ls.free_agent_name(base, taken)) <= NAME_MAX_LENGTH


def test_copy_name_too_long_rejected(fossil: Resource) -> None:
    library = ls.create_library_agent(_input(fossil))
    with pytest.raises(LibraryValidationError) as info:
        ls.copy_agent(library, _scenario_market(), proxy=True, name="y" * 21)
    assert _codes(info.value, "name") == ["max_length"]


# --------------------------------------------------- add_new_agent_to_scenario


def test_add_new_agent_to_scenario(fossil: Resource) -> None:
    market = _scenario_market()
    copy = ls.add_new_agent_to_scenario(_input(fossil), market)
    library = Agent.objects.get(market__isnull=True)
    assert copy.source_id == library.id
    assert copy.market_id == market.id and copy.proxy is True
    assert _plants(copy) == _plants(library)


def test_add_new_agent_rolls_back_library_agent_on_copy_failure(
    fossil: Resource,
) -> None:
    market = _scenario_market()
    make_agent_with_plants(market, [(fossil, 1)], name="Coal")
    with pytest.raises(LibraryValidationError):
        ls.add_new_agent_to_scenario(_input(fossil, name="Coal"), market)
    assert not Agent.objects.filter(market__isnull=True).exists()


# ---------------------------------------------------- clone_scenario_for_run


def _scenario_with_agents(fossil: Resource) -> Simulation:
    market = _scenario_market()
    simulation = market.simulation
    simulation.seed = 42
    simulation.evaluation_periods = 3
    simulation.save()
    ls.add_new_agent_to_scenario(_input(fossil, name="A"), market)
    ls.add_new_agent_to_scenario(
        _input(fossil, name="B", algorithm_type="DDPG"), market
    )
    return simulation


def test_clone_scenario_for_run(fossil: Resource) -> None:
    proxy_sim = _scenario_with_agents(fossil)
    run = ls.clone_scenario_for_run(proxy_sim)
    run.refresh_from_db()
    assert run.id != proxy_sim.id
    assert run.proxy is False
    assert run.state == SimulationState.CREATED
    assert run.currentPeriod == -1
    assert run.seed == 42
    assert run.evaluation_periods == 3
    assert run.periodNumber == proxy_sim.periodNumber
    assert run.market.proxy is False
    assert run.market.state == MarketState.CREATED
    assert run.market.strategy == proxy_sim.market.strategy
    scenario_agents = list(proxy_sim.market.agent_set.order_by("name"))
    run_agents = list(run.market.agent_set.order_by("name"))
    assert [a.name for a in run_agents] == ["A", "B"]
    for src, dst in zip(scenario_agents, run_agents):
        assert dst.proxy is False
        assert dst.source_id == src.source_id
        assert dst.algorithm_type == src.algorithm_type
        assert _plants(dst) == _plants(src)
    assert proxy_sim.market.agent_set.count() == 2


def test_clone_scenario_seed_override_including_zero(fossil: Resource) -> None:
    proxy_sim = _scenario_with_agents(fossil)
    assert ls.clone_scenario_for_run(proxy_sim, seed=7).seed == 7
    assert ls.clone_scenario_for_run(proxy_sim, seed=0).seed == 0


def test_clone_scenario_forced_algorithm_and_overrides(fossil: Resource) -> None:
    proxy_sim = _scenario_with_agents(fossil)
    run = ls.clone_scenario_for_run(
        proxy_sim,
        forced_algorithm="PPO",
        algorithm_overrides={"gamma": 0.9},
        fixed_agent_fields={"failstack_enabled": False, "failstack_limit": 2},
    )
    for agent in Agent.objects.filter(market=run.market):
        assert agent.algorithm_type == "PPO"
        assert agent.algorithm_params == {"actor_lr": 0.001, "gamma": 0.9}
        assert agent.failstack_enabled is False
        assert agent.failstack_limit == 2
    for agent in proxy_sim.market.agent_set.all():
        assert agent.algorithm_type != "PPO"
        assert agent.failstack_limit == 4


def test_clone_scenario_rejects_invalid_fixed_fields(fossil: Resource) -> None:
    proxy_sim = _scenario_with_agents(fossil)
    with pytest.raises(ValueError):
        ls.clone_scenario_for_run(proxy_sim, fixed_agent_fields={"name": "x"})
    with pytest.raises(LibraryValidationError):
        ls.clone_scenario_for_run(proxy_sim, fixed_agent_fields={"failstack_limit": 0})
    assert Simulation.objects.filter(proxy=False).count() == 0


def test_clone_scenario_requires_proxy(fossil: Resource) -> None:
    market = make_market()
    with pytest.raises(ValueError):
        ls.clone_scenario_for_run(market.simulation)


def test_agent_input_is_frozen(fossil: Resource) -> None:
    data = _input(fossil)
    with pytest.raises(dataclasses.FrozenInstanceError):
        data.name = "x"  # type: ignore[misc]
