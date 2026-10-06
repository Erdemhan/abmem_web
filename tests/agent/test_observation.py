"""Tests for abmem.services.agent.observation."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from abmem.constants.paths import DATA_DIR
from abmem.services.agent.observation import (
    ALLOWED_LAG_SETS,
    DEFAULT_LAGS,
    DEMAND_SCALING_MEAN,
    DEMAND_SCALING_STD,
    Observation,
    ObservationNormalizer,
    build_observation,
)


class _Demand:
    """Synthetic demand source: demand(t) = 30000 + t."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def at(self, period_number: int) -> Decimal:
        self.calls.append(period_number)
        return Decimal(30000 + period_number)


def _ptf(t: int) -> Decimal:
    return Decimal(t) + Decimal("0.25")


@pytest.fixture
def history() -> dict[int, Decimal]:
    return {t: _ptf(t) for t in range(1, 201)}


def test_constants() -> None:
    assert DEFAULT_LAGS == (1, 24)
    assert ALLOWED_LAG_SETS == frozenset({(1, 24), (1, 24, 168)})


def test_scaling_constants_match_market_data() -> None:
    """OQ-2: mean and population std of the 8808 valid DEMAND values."""
    df = pd.read_excel(DATA_DIR / "marketData.xlsx", usecols=["DEMAND"])
    values = pd.to_numeric(df["DEMAND"], errors="coerce").dropna().to_numpy(np.float64)
    assert values.size == 8808
    assert np.all(np.isfinite(values))
    assert DEMAND_SCALING_MEAN == pytest.approx(float(values.mean()), rel=1e-12)
    assert DEMAND_SCALING_STD == pytest.approx(float(values.std(ddof=0)), rel=1e-12)


def test_default_lags_at_t30(history) -> None:
    obs = build_observation(30, _Demand(), history)
    assert obs.complete
    assert obs.lags == (1, 24)
    assert obs.raw is not None and obs.raw.dtype == np.float64
    np.testing.assert_array_equal(obs.raw, [30030.0, 29.25, 6.25])


@pytest.mark.parametrize(("t", "missing"), [(1, (1, 24)), (24, (24,)), (25, ())])
def test_default_lags_missing(history, t, missing) -> None:
    obs = build_observation(t, _Demand(), history)
    assert obs.missing_lags == missing
    assert obs.complete is (missing == ())
    assert (obs.raw is None) is (missing != ())


def test_long_lags_at_t170(history) -> None:
    obs = build_observation(170, _Demand(), history, lags=(1, 24, 168))
    assert obs.complete
    np.testing.assert_array_equal(obs.raw, [30170.0, 169.25, 146.25, 2.25])


@pytest.mark.parametrize(
    ("t", "missing"), [(1, (1, 24, 168)), (25, (168,)), (168, (168,)), (169, ())]
)
def test_long_lags_missing(history, t, missing) -> None:
    obs = build_observation(t, _Demand(), history, lags=[1, 24, 168])
    assert obs.missing_lags == missing


@pytest.mark.parametrize("lags", [(24, 1), (1,), (1, 24, 672), (), (1, 168)])
def test_disallowed_lag_set_raises(history, lags) -> None:
    with pytest.raises(ValueError):
        build_observation(30, _Demand(), history, lags=lags)


def test_none_entry_is_missing_not_zero(history) -> None:
    history[29] = None
    obs = build_observation(30, _Demand(), history)
    assert obs.missing_lags == (1,)
    assert obs.raw is None


def test_absent_entry_is_missing(history) -> None:
    del history[6]
    obs = build_observation(30, _Demand(), history)
    assert obs.missing_lags == (24,)


def test_empty_history() -> None:
    obs = build_observation(30, _Demand(), {})
    assert obs.missing_lags == (1, 24)


def test_no_look_ahead(history) -> None:
    """Only demand(t) and PTF of periods strictly before t are read."""
    demand = _Demand()
    poisoned = dict(history)
    poisoned[30] = Decimal("NaN")
    poisoned[31] = Decimal("NaN")
    obs = build_observation(30, demand, poisoned)
    assert obs.complete
    assert demand.calls == [30]


def test_zero_ptf_is_kept(history) -> None:
    history[29] = Decimal("0")
    obs = build_observation(30, _Demand(), history)
    assert obs.complete and obs.raw[1] == 0.0


@pytest.mark.parametrize("t", [0, -1, 2.5, True])
def test_invalid_t_raises(history, t) -> None:
    with pytest.raises(ValueError):
        build_observation(t, _Demand(), history)


def test_non_finite_ptf_raises(history) -> None:
    history[29] = Decimal("Infinity")
    with pytest.raises(ValueError):
        build_observation(30, _Demand(), history)


def test_non_finite_demand_raises(history) -> None:
    class _NanDemand:
        def at(self, period_number: int) -> Decimal:
            return Decimal("NaN")

    with pytest.raises(ValueError):
        build_observation(30, _NanDemand(), history)


def test_observation_complete_property() -> None:
    assert Observation(t=1, lags=(1, 24), raw=None, missing_lags=(1,)).complete is False


# ---------------------------------------------------------------- normaliser


def test_for_market_maps_bounds_to_plus_minus_one() -> None:
    norm = ObservationNormalizer.for_market((1, 24), 0.0, 3000.0)
    out = norm.transform(np.array([DEMAND_SCALING_MEAN, 0.0, 3000.0]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, -1.0, 1.0], atol=1e-7)
    out = norm.transform(
        np.array([DEMAND_SCALING_MEAN + DEMAND_SCALING_STD, 1500.0, 1500.0])
    )
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0], atol=1e-7)


def test_for_market_long_lags_shape() -> None:
    norm = ObservationNormalizer.for_market((1, 24, 168), -100, 300)
    assert norm.mean.shape == (4,)
    np.testing.assert_array_equal(norm.mean[1:], [100.0] * 3)
    np.testing.assert_array_equal(norm.scale[1:], [200.0] * 3)


def test_transform_batch() -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    batch = np.array(
        [[DEMAND_SCALING_MEAN, 0.0, 200.0], [DEMAND_SCALING_MEAN, 100.0, 50.0]]
    )
    out = norm.transform(batch)
    assert out.shape == (2, 3) and out.dtype == np.float32
    np.testing.assert_allclose(out[1], [0.0, 0.0, -0.5], atol=1e-7)


def test_distinct_states_give_distinct_vectors() -> None:
    """C4 regression: the analiz.md C4 inputs must not collapse to one vector."""
    norm = ObservationNormalizer.for_market((1, 24, 168), 0, 3000)
    raws = np.array(
        [[39000, 100, 0, 0], [25000, 60, 0, 0], [57000, 180, 0, 0]], dtype=np.float64
    )
    out = norm.transform(raws)
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[0], out[2])
    assert not np.allclose(out[1], out[2])
    # Per-row transform equals batch transform (no batch-dependent statistics).
    for i in range(3):
        np.testing.assert_array_equal(norm.transform(raws[i]), out[i])


def test_no_integer_truncation() -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    out = norm.transform(np.array([DEMAND_SCALING_MEAN, 100.5, 100.25]))
    assert out[1] == pytest.approx(0.005) and out[2] == pytest.approx(0.0025)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_transform_rejects_non_finite(bad: float) -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    with pytest.raises(ValueError):
        norm.transform(np.array([40000.0, bad, 10.0]))


@pytest.mark.parametrize("shape", [(2,), (4,), (2, 4), (1, 2, 3), ()])
def test_transform_rejects_wrong_shape(shape) -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    with pytest.raises(ValueError):
        norm.transform(np.zeros(shape))


@pytest.mark.parametrize(
    ("low", "high"), [(100, 100), (200, 100), (0, float("inf")), (float("nan"), 1)]
)
def test_for_market_invalid_bounds(low, high) -> None:
    with pytest.raises(ValueError):
        ObservationNormalizer.for_market(DEFAULT_LAGS, low, high)


def test_for_market_disallowed_lags() -> None:
    with pytest.raises(ValueError):
        ObservationNormalizer.for_market((1, 2), 0, 1)


@pytest.mark.parametrize(
    ("mean", "scale"),
    [([0.0], [0.0]), ([0.0], [-1.0]), ([0.0, 1.0], [1.0]), ([], []), ([np.nan], [1.0])],
)
def test_direct_construction_validation(mean, scale) -> None:
    with pytest.raises(ValueError):
        ObservationNormalizer(mean=np.array(mean), scale=np.array(scale))


def test_parameters_are_read_only() -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    with pytest.raises(ValueError):
        norm.mean[0] = 1.0


def test_to_dict() -> None:
    norm = ObservationNormalizer.for_market(DEFAULT_LAGS, 0, 200)
    d = norm.to_dict()
    assert d == {
        "mean": [DEMAND_SCALING_MEAN, 100.0, 100.0],
        "scale": [DEMAND_SCALING_STD, 100.0, 100.0],
    }
    assert all(type(v) is float for v in d["mean"] + d["scale"])


def test_build_then_normalise_pipeline(history) -> None:
    obs = build_observation(30, _Demand(), history)
    norm = ObservationNormalizer.for_market(obs.lags, 0, 200)
    out = norm.transform(obs.raw)
    assert out.shape == (3,) and np.all(np.isfinite(out))
