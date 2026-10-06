"""Agent name, library role and fail-stack constraints (data_model spec, 0006/0009)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from django.db import IntegrityError, transaction

from abmem.models import Agent
from abmem.models.enums import AlgorithmType
from tests.factories import make_market

pytestmark = pytest.mark.django_db


def _library(name: str, **fields: Any) -> Agent:
    return Agent.objects.create(name=name, market=None, proxy=True, **fields)


def _assert_rejected(**fields: Any) -> None:
    with pytest.raises(IntegrityError), transaction.atomic():
        Agent.objects.create(**fields)


def test_library_names_unique_case_insensitive() -> None:
    _library("Coal")
    _assert_rejected(name="COAL", market=None, proxy=True)


def test_duplicate_name_in_one_market_rejected() -> None:
    market = make_market()
    Agent.objects.create(name="CCGT", market=market)
    _assert_rejected(name="CCGT", market=market)


def test_case_variants_allowed_in_one_market() -> None:
    market = make_market()
    Agent.objects.create(name="CCGT", market=market)
    Agent.objects.create(name="ccgt", market=market)
    assert Agent.objects.filter(market=market).count() == 2


def test_same_name_in_two_markets_and_in_library() -> None:
    Agent.objects.create(name="CCGT", market=make_market())
    Agent.objects.create(name="CCGT", market=make_market())
    _library("CCGT")
    assert Agent.objects.filter(name="CCGT").count() == 3


def test_library_agent_must_be_proxy() -> None:
    _assert_rejected(name="lib", market=None, proxy=False)


@pytest.mark.parametrize(
    "fields",
    [
        {"failstack_limit": 0},
        {"failstack_delta": Decimal("-0.01")},
    ],
)
def test_failstack_constraints(fields: dict[str, Any]) -> None:
    _assert_rejected(name="fs", market=make_market(), **fields)


def test_defaults_and_role_properties() -> None:
    agent = Agent.objects.create(name="a", market=make_market())
    agent.refresh_from_db()
    assert agent.algorithm_type == AlgorithmType.DDPG
    assert agent.algorithm_params == {}
    assert agent.failstack_enabled is False
    assert agent.failstack_limit == 6
    assert agent.failstack_delta == Decimal("15")
    assert agent.failstack_min_margin == Decimal("0")
    assert agent.source is None
    assert agent.is_library is False
    assert agent.is_baseline is False
    library = _library("lib", algorithm_type=AlgorithmType.MARGINAL_COST)
    assert library.is_library is True
    assert library.is_baseline is True


def test_runtime_policy_is_not_persisted() -> None:
    agent = Agent.objects.create(name="a", market=make_market())
    agent.runtime_policy = object()
    agent.save()
    assert Agent.objects.get(id=agent.id).runtime_policy is None
    assert "runtime_policy" not in {f.name for f in Agent._meta.get_fields()}


def test_algorithm_alias_maps_to_runtime_policy() -> None:
    agent = Agent(name="a")
    policy = object()
    agent.algorithm = policy
    assert agent.runtime_policy is policy
    assert agent.algorithm is policy


def test_deleting_library_agent_keeps_copies() -> None:
    library = _library("lib")
    copy = Agent.objects.create(
        name="lib", market=make_market(), proxy=True, source=library
    )
    library.delete()
    copy.refresh_from_db()
    assert copy.source is None
