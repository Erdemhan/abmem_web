"""Runs with a selectable clearing algorithm and unknown algorithm keys (PLAN 2.10)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from decimal import Decimal

import pytest

from abmem.models.enums import MarketStrategy, SimulationState
from abmem.models.models import Offer, Period
from abmem.services.market.clearing import Bid, ClearingContractError, ClearingResult
from abmem.services.market.clearing_algorithms import registry as clearing_registry
from abmem.services.market.clearing_algorithms.base import BaseClearingAlgorithm
from abmem.services.market.clearing_algorithms.merit_order import MeritOrderClearing
from abmem.services.simulation import simulation_service
from abmem.services.simulation.context import (
    SimulationContext,
    SimulationValidationError,
)
from tests.runtime.conftest import constant_demand, make_run, resources

pytestmark = pytest.mark.django_db

REVERSE = "TEST_REVERSE_ORDER"
BROKEN = "TEST_BROKEN_CLEARING"


class ReverseOrderClearing(BaseClearingAlgorithm):
    """Test-only clearing: accepts the most expensive bids first.

    The PTF is the lowest accepted price. Defined without a key, the fixture
    registers a keyed subclass for one test only.
    """

    def clear(
        self, bids: Sequence[Bid], demand: Decimal, quantum: Decimal
    ) -> ClearingResult:
        """Accept bids in descending price order until the demand is met."""
        accepted = {bid.key: Decimal(0) for bid in bids}
        remaining = demand
        ptf: Decimal | None = None
        for bid in sorted(bids, key=lambda b: (-b.price, b.key)):
            if remaining <= 0 or bid.quantity <= 0:
                continue
            take = min(bid.quantity, remaining)
            accepted[bid.key] = take
            remaining -= take
            ptf = bid.price
        met = sum(accepted.values(), Decimal(0))
        marginal = tuple(b.key for b in bids if ptf is not None and b.price == ptf)
        return ClearingResult(
            accepted=accepted,
            ptf=ptf,
            met_demand=met,
            offered_volume=sum((b.quantity for b in bids), Decimal(0)),
            marginal_group=marginal,
        )


@pytest.fixture
def reverse_clearing() -> Iterator[str]:
    """Register ``ReverseOrderClearing`` under a test key for one test."""
    type(
        "RegisteredReverseOrder",
        (ReverseOrderClearing,),
        {"key": REVERSE, "label": "Reverse order (test)", "__module__": __name__},
    )
    try:
        yield REVERSE
    finally:
        clearing_registry.CLEARING_ALGORITHMS.unregister(REVERSE)


@pytest.fixture
def broken_clearing() -> Iterator[str]:
    """Register a clearing algorithm whose result reports a wrong met demand."""

    def clear(
        self: BaseClearingAlgorithm,
        bids: Sequence[Bid],
        demand: Decimal,
        quantum: Decimal,
    ) -> ClearingResult:
        result = MeritOrderClearing().clear(bids, demand, quantum)
        return replace(result, met_demand=result.met_demand + 1)

    type(
        "BrokenClearing",
        (BaseClearingAlgorithm,),
        {"key": BROKEN, "clear": clear, "__module__": __name__},
    )
    try:
        yield BROKEN
    finally:
        clearing_registry.CLEARING_ALGORITHMS.unregister(BROKEN)


def _run(clearing: str, strategy: MarketStrategy, periods: int = 3) -> list[Period]:
    """Run three marginal-cost agents with ``clearing`` and return the periods."""
    r = resources(3)
    simulation = make_run(
        [
            {"plants": [(r[0], 600)], "algorithm": "MARGINAL_COST"},
            {"plants": [(r[1], 600)], "algorithm": "MARGINAL_COST"},
            {"plants": [(r[2], 600)], "algorithm": "MARGINAL_COST"},
        ],
        period_number=periods,
        strategy=strategy,
    )
    simulation.market.clearing_algorithm = clearing
    simulation.market.save(update_fields=["clearing_algorithm"])
    ctx = SimulationContext.build(
        simulation, demand=constant_demand(), export_json=False
    )
    assert ctx.clearing.key == clearing
    from abmem.services.market import market_service

    return [market_service.run_period(ctx, t) for t in range(1, periods + 1)]


def _offers(period: Period) -> list[Offer]:
    """Return the offers of ``period`` ordered by ascending price."""
    return sorted(period.offer_set.all(), key=lambda o: (o.offerPrice, o.id))


@pytest.mark.parametrize("strategy", [MarketStrategy.PAYASBID, MarketStrategy.PAYASPTF])
def test_run_uses_the_selected_clearing_algorithm(
    reverse_clearing: str, strategy: MarketStrategy
) -> None:
    for period in _run(reverse_clearing, strategy):
        cheap, middle, expensive = _offers(period)
        assert cheap.offerPrice < middle.offerPrice < expensive.offerPrice
        # Reverse order: the most expensive offer is filled first.
        assert expensive.acceptanceAmount == expensive.amount
        assert Decimal(middle.acceptanceAmount) == period.demand - expensive.amount
        assert cheap.acceptanceAmount == 0 and not cheap.acceptance
        assert period.ptf == middle.offerPrice
        for offer in (middle, expensive):
            expected = (
                offer.offerPrice if strategy == MarketStrategy.PAYASBID else period.ptf
            )
            assert offer.acceptancePrice == expected
        assert cheap.acceptancePrice is None


def test_merit_order_default_fills_the_cheapest_offer_first() -> None:
    for period in _run("MERIT_ORDER", MarketStrategy.PAYASPTF, periods=1):
        cheap, middle, expensive = _offers(period)
        assert cheap.acceptanceAmount == cheap.amount
        assert expensive.acceptanceAmount == 0
        assert period.ptf == middle.offerPrice


def test_contract_violation_fails_the_run(
    broken_clearing: str, fast_demand: object
) -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 600)], "algorithm": "MARGINAL_COST"}])
    simulation.market.clearing_algorithm = broken_clearing
    simulation.market.save(update_fields=["clearing_algorithm"])
    with pytest.raises(ClearingContractError, match=broken_clearing):
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert broken_clearing in simulation.error and "met_demand" in simulation.error
    assert not Period.objects.filter(market=simulation.market).exists()


def test_unknown_clearing_key_stops_the_run_at_validation(fast_demand: object) -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)]}])
    simulation.market.clearing_algorithm = "REMOVED_CLEARING"
    simulation.market.save(update_fields=["clearing_algorithm"])
    with pytest.raises(SimulationValidationError, match="'REMOVED_CLEARING'"):
        simulation_service.run(simulation, export_json=False)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert "clearing algorithm 'REMOVED_CLEARING' is not available" in simulation.error


def test_unknown_agent_algorithm_key_stops_the_run_at_validation(
    fast_demand: object,
) -> None:
    r = resources(2)
    simulation = make_run(
        [
            {"plants": [(r[0], 100)], "algorithm": "RANDOM"},
            {"plants": [(r[1], 100)], "algorithm": "REMOVED_ALGO"},
        ]
    )
    with pytest.raises(SimulationValidationError) as info:
        simulation_service.run(simulation, export_json=False)
    assert "agent 'a1': algorithm 'REMOVED_ALGO' is not available" in str(info.value)
    simulation.refresh_from_db()
    assert simulation.state == SimulationState.FAILED
    assert not Period.objects.filter(market=simulation.market).exists()


def test_unknown_forced_algorithm_is_rejected() -> None:
    r = resources(1)
    simulation = make_run([{"plants": [(r[0], 100)], "algorithm": "RANDOM"}])
    with pytest.raises(SimulationValidationError, match="'NOPE' is not available"):
        SimulationContext.build(
            simulation,
            forced_algorithm="NOPE",
            demand=constant_demand(),
            export_json=False,
        )
