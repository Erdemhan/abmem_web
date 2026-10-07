"""Tests of the clearing algorithm registry and the clearing result contract (PLAN 2.10)."""

from __future__ import annotations

import sys
import textwrap
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from django.core.exceptions import ImproperlyConfigured

from abmem.services.market.clearing import (
    Bid,
    ClearingContractError,
    ClearingResult,
    clear_market,
    validate_clearing_result,
)
from abmem.services.market.clearing_algorithms import registry
from abmem.services.market.clearing_algorithms.base import BaseClearingAlgorithm
from abmem.services.market.clearing_algorithms.merit_order import MeritOrderClearing
from abmem.services.plugin_registry import PluginDiscoveryError

D = Decimal
Q = D("0.001")
LOW, HIGH = D(0), D(200)

# Registry ---------------------------------------------------------------------


def test_merit_order_is_the_only_and_default_algorithm() -> None:
    infos = registry.available_clearing_algorithms()
    assert [info.key for info in infos] == ["MERIT_ORDER"]
    assert infos[0].label == "Uniform-price merit order"
    assert infos[0].description
    assert registry.DEFAULT_CLEARING_ALGORITHM == "MERIT_ORDER"
    algorithm = registry.get_clearing_algorithm("MERIT_ORDER")
    assert isinstance(algorithm, MeritOrderClearing)
    assert registry.is_known_clearing("MERIT_ORDER")
    assert registry.clearing_label_for("MERIT_ORDER") == "Uniform-price merit order"


@pytest.mark.parametrize("bad", ["merit_order", "", None, 0, "PAY_AS_CLEAR"])
def test_unknown_clearing_key(bad: object) -> None:
    assert not registry.is_known_clearing(bad)
    with pytest.raises(KeyError, match="known keys"):
        registry.get_clearing_algorithm(bad)  # type: ignore[arg-type]


def test_clear_market_wrapper_delegates_to_merit_order() -> None:
    bids = [Bid(0, D(60), D(100)), Bid(1, D(60), D(10)), Bid(2, D(10), D(5))]
    assert clear_market(bids, D(60)) == MeritOrderClearing().clear(bids, D(60), Q)


def test_duplicate_clearing_key_names_both_classes() -> None:
    with pytest.raises(ImproperlyConfigured) as info:

        class OtherMeritOrder(BaseClearingAlgorithm):
            key = "MERIT_ORDER"

            def clear(
                self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal
            ) -> ClearingResult:
                raise NotImplementedError

    assert "MeritOrderClearing" in str(info.value)
    assert "OtherMeritOrder" in str(info.value)
    assert type(registry.get_clearing_algorithm("MERIT_ORDER")) is MeritOrderClearing


def test_abstract_clearing_subclass_is_skipped() -> None:
    class StillAbstract(BaseClearingAlgorithm):
        key = "TEST_STILL_ABSTRACT"

    assert not registry.is_known_clearing("TEST_STILL_ABSTRACT")


PLUGIN_SOURCE = """
from abmem.services.market.clearing_algorithms.merit_order import MeritOrderClearing


class CopyOfMeritOrder(MeritOrderClearing):
    key = "TEST_COPY_MERIT_ORDER"
    label = "Copy (test)"
    description = "Same as the merit order."
"""


@pytest.fixture
def plugin_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Yield a fresh importable package and remove what it registered afterwards."""
    name = f"abmem_test_clearing_{uuid.uuid4().hex[:8]}"
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(registry.CLEARING_ALGORITHMS.classes())
    try:
        yield name
    finally:
        for key in set(registry.CLEARING_ALGORITHMS.classes()) - before:
            registry.CLEARING_ALGORITHMS.unregister(key)
        for module in [m for m in sys.modules if m.startswith(name)]:
            del sys.modules[module]


def test_plugin_module_is_discovered(plugin_package: str, tmp_path: Path) -> None:
    (tmp_path / plugin_package / "copy_algo.py").write_text(
        textwrap.dedent(PLUGIN_SOURCE), "utf-8"
    )
    assert not registry.is_known_clearing("TEST_COPY_MERIT_ORDER")
    registry.CLEARING_ALGORITHMS.discover_package(plugin_package)
    keys = [info.key for info in registry.available_clearing_algorithms()]
    assert keys == ["MERIT_ORDER", "TEST_COPY_MERIT_ORDER"]
    bids = [Bid(0, D(50), D(10)), Bid(1, D(40), D(10))]
    copy = registry.get_clearing_algorithm("TEST_COPY_MERIT_ORDER")
    assert copy.clear(bids, D(15), Q) == clear_market(bids, D(15))


def test_failing_plugin_module_is_reported(plugin_package: str, tmp_path: Path) -> None:
    (tmp_path / plugin_package / "broken.py").write_text(
        "raise ImportError('missing dependency xyz')\n", "utf-8"
    )
    with pytest.raises(PluginDiscoveryError, match="missing dependency xyz"):
        registry.CLEARING_ALGORITHMS.discover_package(plugin_package)


# Result contract --------------------------------------------------------------

BIDS = [Bid(0, D(20), D(10)), Bid(1, D(30), D(10)), Bid(2, D(40), D(10))]
DEMAND = D(15)


def _valid() -> ClearingResult:
    """Return the merit-order result of ``BIDS`` for ``DEMAND``."""
    return MeritOrderClearing().clear(BIDS, DEMAND, Q)


def _check(result: object, low: Decimal = LOW, high: Decimal = HIGH) -> None:
    validate_clearing_result(result, BIDS, DEMAND, Q, low, high, algorithm="FAKE")


def test_valid_result_passes_unchanged() -> None:
    result = _valid()
    assert result.ptf == D(30) and result.met_demand == D(15)
    assert validate_clearing_result(result, BIDS, DEMAND, Q, LOW, HIGH) is result


def test_zero_demand_result_passes() -> None:
    result = MeritOrderClearing().clear(BIDS, D(0), Q)
    assert validate_clearing_result(result, BIDS, D(0), Q, LOW, HIGH) is result


@pytest.mark.parametrize(
    ("broken", "rule"),
    [
        (lambda r: "not a result", "expected ClearingResult"),
        (
            lambda r: replace(r, accepted={0: D(10), 1: D(5)}),
            "exactly the bid keys",
        ),
        (
            lambda r: replace(r, accepted={**r.accepted, 9: D(0)}),
            "exactly the bid keys",
        ),
        (
            lambda r: replace(r, accepted={**r.accepted, 1: 5}),
            "must be a finite Decimal",
        ),
        (
            lambda r: replace(r, accepted={**r.accepted, 1: D("NaN")}),
            "must be a finite Decimal",
        ),
        (
            lambda r: replace(
                r, accepted={0: D(10), 1: D(-5), 2: D(10)}, met_demand=D(15)
            ),
            "outside [0, 10]",
        ),
        (
            lambda r: replace(r, accepted={0: D(15), 1: D(0), 2: D(0)}),
            "outside [0, 10]",
        ),
        (
            lambda r: replace(
                r,
                accepted={0: D(10), 1: D("4.9995"), 2: D("0.0005")},
            ),
            "not a multiple of quantum",
        ),
        (lambda r: replace(r, met_demand=D(14)), "differs from the sum of accepted"),
        (
            lambda r: replace(
                r, accepted={0: D(10), 1: D(10), 2: D(0)}, met_demand=D(20)
            ),
            "exceeds min(demand, offered_volume)",
        ),
        (lambda r: replace(r, offered_volume=D(31)), "offered_volume"),
        (lambda r: replace(r, ptf=None), "finite Decimal when met_demand > 0"),
        (lambda r: replace(r, ptf=D("Infinity")), "finite Decimal when met_demand > 0"),
        (lambda r: replace(r, ptf=D(250)), "outside the bid bounds"),
        (lambda r: replace(r, marginal_group=(7,)), "not bids"),
        (lambda r: replace(r, marginal_group=[1]), "tuple without repeated keys"),
        (lambda r: replace(r, marginal_group=(1, 1)), "tuple without repeated keys"),
    ],
)
def test_each_broken_rule_raises(broken: object, rule: str) -> None:
    with pytest.raises(ClearingContractError) as info:
        _check(broken(_valid()))  # type: ignore[operator]
    message = str(info.value)
    assert "FAKE" in message and rule in message
    assert isinstance(info.value, ValueError)


def test_ptf_without_accepted_quantity_raises() -> None:
    zero = MeritOrderClearing().clear(BIDS, D(0), Q)
    with pytest.raises(ClearingContractError, match="ptf must be None"):
        validate_clearing_result(
            replace(zero, ptf=D(20)), BIDS, D(0), Q, LOW, HIGH, algorithm="FAKE"
        )


def test_ptf_below_the_lower_bound_raises() -> None:
    with pytest.raises(ClearingContractError, match="outside the bid bounds"):
        _check(_valid(), low=D(31))
