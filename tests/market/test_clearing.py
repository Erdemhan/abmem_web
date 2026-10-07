"""Unit tests for the pure merit-order clearing (PLAN 1.7, B4, B8, B11, OQ-18)."""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from abmem.models.enums import MarketStrategy
from abmem.services.market.clearing import (
    Bid,
    ClearingResult,
    acceptance_prices,
    clear_market,
    validate_clearing_result,
)

D = Decimal


def _bids(specs: list[tuple[int | str, int | str]]) -> list[Bid]:
    """Build bids keyed by position from (quantity, price) pairs."""
    return [
        Bid(key=i, price=D(str(p)), quantity=D(str(q)))
        for i, (q, p) in enumerate(specs)
    ]


def _check_invariants(bids: list[Bid], demand: Decimal, result: ClearingResult) -> None:
    """Assert every invariant of the spec on one clearing result.

    The result must also satisfy the contract every clearing algorithm is
    checked against at runtime (PLAN 2.10).
    """
    validate_clearing_result(
        result,
        bids,
        demand,
        D("0.001"),
        D(0),
        max((b.price for b in bids), default=D(0)),
        algorithm="MERIT_ORDER",
    )
    offered = sum((b.quantity for b in bids), D(0))
    assert result.offered_volume == offered
    assert set(result.accepted) == {b.key for b in bids}
    assert sum(result.accepted.values(), D(0)) == min(demand, offered)
    assert result.met_demand == min(demand, offered)
    for bid in bids:
        assert D(0) <= result.accepted[bid.key] <= bid.quantity
    positive = [b for b in bids if b.quantity > 0]
    if result.ptf is None:
        assert all(v == 0 for v in result.accepted.values())
        return
    for bid in positive:
        if bid.price < result.ptf:
            assert result.accepted[bid.key] == bid.quantity
        elif bid.price > result.ptf:
            assert result.accepted[bid.key] == 0
    assert result.ptf == max(b.price for b in positive if result.accepted[b.key] > 0)


def test_analysis_table_example() -> None:
    bids = _bids([(400, 50), (300, 70), (500, 90), (200, 120)])
    result = clear_market(bids, D(1000))
    assert [result.accepted[i] for i in range(4)] == [D(400), D(300), D(300), D(0)]
    assert result.ptf == D(90)
    assert result.met_demand == D(1000)
    assert result.offered_volume == D(1400)
    assert result.marginal_group == (2,)
    _check_invariants(bids, D(1000), result)


def test_b4_oracle_equal_share_capped() -> None:
    bids = _bids([(100, 60), (10, 60)])
    result = clear_market(bids, D(60))
    assert result.accepted[0] == D(50)
    assert result.accepted[1] == D(10)
    assert result.ptf == D(60)
    assert result.met_demand == D(60)
    assert result.marginal_group == (0, 1)


def test_b4_oracle_with_cheaper_group_first() -> None:
    # Remaining demand after the cheaper bid is 60.
    bids = _bids([(40, 10), (100, 60), (10, 60)])
    result = clear_market(bids, D(100))
    assert [result.accepted[i] for i in range(3)] == [D(40), D(50), D(10)]


def test_three_way_tie_unequal_quantities() -> None:
    bids = _bids([(10, 75), (20, 75), (100, 75)])
    result = clear_market(bids, D(60))
    assert [result.accepted[i] for i in range(3)] == [D(10), D(20), D(30)]
    assert sum(result.accepted.values()) == D(60)


def test_tie_split_quantised_sums_exactly() -> None:
    bids = _bids([(50, 30), (50, 30), (50, 30)])
    result = clear_market(bids, D(100))
    values = [result.accepted[i] for i in range(3)]
    assert sum(values) == D(100)
    # 100/3 = 33.333..., one quantum goes to the smallest key.
    assert values == [D("33.334"), D("33.333"), D("33.333")]
    assert all(v == v.quantize(D("0.001")) for v in values)


def test_supply_shortfall() -> None:
    bids = _bids([(100, 40), (200, 80), (50, 60)])
    result = clear_market(bids, D(1000))
    assert [result.accepted[i] for i in range(3)] == [D(100), D(200), D(50)]
    assert result.ptf == D(80)
    assert result.met_demand == D(350)
    assert result.offered_volume == D(350)


def test_single_bid_larger_than_demand() -> None:
    result = clear_market(_bids([(500, 42)]), D(123))
    assert result.accepted[0] == D(123)
    assert result.ptf == D(42)


def test_zero_demand_accepts_nothing() -> None:
    bids = _bids([(100, 40), (200, 80)])
    result = clear_market(bids, D(0))
    assert all(v == 0 for v in result.accepted.values())
    assert result.ptf is None
    assert result.met_demand == D(0)
    assert result.offered_volume == D(300)
    assert result.marginal_group == ()


def test_empty_bids() -> None:
    result = clear_market([], D(100))
    assert result.accepted == {}
    assert result.ptf is None
    assert result.met_demand == D(0)
    assert result.offered_volume == D(0)


def test_zero_quantity_bid_ignored_for_ptf() -> None:
    bids = _bids([(0, 10), (100, 50)])
    result = clear_market(bids, D(60))
    assert result.accepted[0] == D(0)
    assert result.accepted[1] == D(60)
    assert result.ptf == D(50)
    # A zero-quantity bid priced above everything does not move the PTF either.
    result = clear_market(_bids([(100, 50), (0, 99)]), D(500))
    assert result.ptf == D(50)
    assert result.marginal_group == (0,)


def test_only_zero_quantity_bids() -> None:
    result = clear_market(_bids([(0, 10), (0, 20)]), D(60))
    assert result.ptf is None
    assert result.met_demand == D(0)


@pytest.mark.parametrize(
    "bid",
    [
        Bid(0, D(-1), D(10)),
        Bid(0, D(10), D(-1)),
        Bid(0, D("NaN"), D(10)),
        Bid(0, D(10), D("NaN")),
        Bid(0, D("Infinity"), D(10)),
        Bid(0, D(10), D("-Infinity")),
        Bid(0, 10.0, D(10)),  # type: ignore[arg-type]
        Bid(0, D(10), None),  # type: ignore[arg-type]
        Bid(0, D(10), D("0.0001")),
    ],
)
def test_invalid_bid_raises(bid: Bid) -> None:
    with pytest.raises(ValueError):
        clear_market([bid], D(10))


@pytest.mark.parametrize(
    "demand", [D(-1), D("NaN"), D("Infinity"), None, 5.0, D("1.0005")]
)
def test_invalid_demand_raises(demand: object) -> None:
    with pytest.raises(ValueError):
        clear_market(_bids([(10, 10)]), demand)  # type: ignore[arg-type]


@pytest.mark.parametrize("quantum", [D(0), D(-1), D("NaN")])
def test_invalid_quantum_raises(quantum: Decimal) -> None:
    with pytest.raises(ValueError):
        clear_market(_bids([(10, 10)]), D(5), quantum=quantum)


def test_duplicate_keys_raise() -> None:
    with pytest.raises(ValueError):
        clear_market([Bid(0, D(1), D(1)), Bid(0, D(2), D(1))], D(1))


def test_randomised_invariants_and_permutation_invariance() -> None:
    rng = random.Random(20261006)
    for _ in range(200):
        n = rng.randint(0, 12)
        price_pool = [D(rng.randint(0, 20)) for _ in range(rng.randint(1, 5))]
        bids = [
            Bid(
                key=i,
                price=rng.choice(price_pool),
                quantity=D(rng.randint(0, 300000)) / 1000,
            )
            for i in range(n)
        ]
        demand = D(rng.randint(0, 2000000)) / 1000
        result = clear_market(bids, demand)
        _check_invariants(bids, demand, result)
        shuffled = bids[:]
        rng.shuffle(shuffled)
        other = clear_market(shuffled, demand)
        assert dict(other.accepted) == dict(result.accepted)
        assert other.ptf == result.ptf
        assert other.marginal_group == result.marginal_group


def test_acceptance_prices_pay_as_bid_and_pay_as_ptf() -> None:
    bids = _bids([(400, 50), (300, 70), (500, 90), (200, 120)])
    result = clear_market(bids, D(1000))
    bid_prices = acceptance_prices(result, bids, MarketStrategy.PAYASBID)
    assert bid_prices == {0: D(50), 1: D(70), 2: D(90), 3: None}
    ptf_prices = acceptance_prices(result, bids, MarketStrategy.PAYASPTF)
    assert ptf_prices == {0: D(90), 1: D(90), 2: D(90), 3: None}


def test_acceptance_prices_unknown_strategy() -> None:
    bids = _bids([(10, 10)])
    result = clear_market(bids, D(5))
    with pytest.raises(ValueError):
        acceptance_prices(result, bids, "PAYASWHATEVER")  # type: ignore[arg-type]


def test_acceptance_prices_nothing_accepted() -> None:
    bids = _bids([(10, 10)])
    result = clear_market(bids, D(0))
    assert acceptance_prices(result, bids, MarketStrategy.PAYASPTF) == {0: None}


@pytest.mark.django_db
def test_offer_acceptance_amount_decimal_round_trip() -> None:
    from abmem.models.models import Offer
    from tests.factories import (
        make_agent_with_plants,
        make_market,
        make_offer,
        make_period,
        make_resource,
    )

    market = make_market()
    resource = make_resource()
    agent = make_agent_with_plants(market, plants=[(resource, 1000)])
    period = make_period(market, demand=100)
    period.metDemand = D("99.125")
    period.save()
    offer = make_offer(
        period,
        agent,
        resource,
        7000,
        D(50),
        acceptance=True,
        acceptance_price=D(50),
        acceptance_amount=D("6264.320"),  # type: ignore[arg-type]
        save=True,
    )
    stored = Offer.objects.get(pk=offer.pk)
    assert stored.acceptanceAmount == D("6264.320")
    assert isinstance(stored.acceptanceAmount, Decimal)
    assert stored.to_dict()["acceptanceAmount"] == "6264.320"
    period.refresh_from_db()
    assert period.metDemand == D("99.125")
