"""Tests for abmem.services.hpo.param_space."""

from __future__ import annotations

import math

import numpy as np
import pytest

from abmem.services.hpo.param_space import ParamSpace, ParamSpec


def test_linear_round_trip() -> None:
    spec = ParamSpec("gamma", 0.9, 0.999, "linear")
    for value in (0.9, 0.95, 0.999):
        assert spec.from_unit(spec.to_unit(value)) == pytest.approx(value)
    assert spec.from_unit(0.0) == 0.9
    assert spec.from_unit(1.0) == 0.999


def test_log_scale_maps_geometric_midpoint() -> None:
    spec = ParamSpec("lr", 1e-5, 1e-1, "log")
    assert spec.from_unit(0.5) == pytest.approx(1e-3)
    assert spec.to_unit(1e-3) == pytest.approx(0.5)
    for value in (1e-5, 3e-4, 1e-1):
        assert spec.from_unit(spec.to_unit(value)) == pytest.approx(value)


def test_int_rounding_is_consistent_and_uniform() -> None:
    spec = ParamSpec("batch", 1, 4, "int")
    assert spec.from_unit(0.0) == 1
    assert spec.from_unit(0.2499) == 1
    assert spec.from_unit(0.25) == 2
    assert spec.from_unit(0.99) == 4
    assert spec.from_unit(1.0) == 4
    for k in range(1, 5):
        assert spec.from_unit(spec.to_unit(k)) == k
        assert isinstance(spec.from_unit(spec.to_unit(k)), int)
    # Equal probability mass per integer.
    grid = np.linspace(0.0, 1.0, 40001)[:-1]
    counts = np.bincount([spec.from_unit(u) for u in grid])[1:]
    assert np.allclose(counts / counts.sum(), 0.25, atol=1e-3)


def test_int_accepts_integral_floats() -> None:
    spec = ParamSpec("epochs", 1.0, 10.0, "int")
    assert spec.from_unit(1.0) == 10


def test_from_unit_clips_and_stays_in_bounds() -> None:
    spec = ParamSpec("tau", 0.001, 0.1, "log")
    assert spec.from_unit(-0.5) == pytest.approx(0.001)
    assert spec.from_unit(1.5) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        spec.from_unit(math.nan)


@pytest.mark.parametrize(
    "low, high, scale",
    [
        (1.0, 1.0, "linear"),
        (2.0, 1.0, "linear"),
        (0.0, 1.0, "log"),
        (-1.0, 1.0, "log"),
        (0.5, 3.0, "int"),
        (math.nan, 1.0, "linear"),
        (0.0, math.inf, "linear"),
        (0.0, 1.0, "cubic"),
    ],
)
def test_invalid_bounds_rejected(low: float, high: float, scale: str) -> None:
    with pytest.raises(ValueError):
        ParamSpec("p", low, high, scale)  # type: ignore[arg-type]


def test_to_unit_rejects_out_of_range_values() -> None:
    spec = ParamSpec("p", 0.0, 1.0)
    with pytest.raises(ValueError):
        spec.to_unit(1.5)
    with pytest.raises(ValueError):
        ParamSpec("k", 1, 5, "int").to_unit(2.5)


def test_space_rejects_duplicates_and_empty() -> None:
    with pytest.raises(ValueError):
        ParamSpace((ParamSpec("a", 0, 1), ParamSpec("a", 0, 2)))
    with pytest.raises(ValueError):
        ParamSpace(())


def test_from_bounds_decode_and_sample() -> None:
    space = ParamSpace.from_bounds(
        {
            "lr_actor": {"low": 1e-5, "high": 1e-2, "scale": "log"},
            "batch_size": {"low": 16, "high": 256, "scale": "int"},
            "gamma": {"low": 0.9, "high": 0.999},
            "tau": (0.001, 0.05, "linear"),
        }
    )
    assert space.names == ("lr_actor", "batch_size", "gamma", "tau")
    assert space.specs[2].scale == "linear"
    rng = np.random.default_rng(0)
    for _ in range(100):
        x = space.sample(rng)
        assert x.shape == (4,)
        assert np.all((x >= 0.0) & (x <= 1.0))
        params = space.decode(x)
        for spec in space.specs:
            assert spec.low <= params[spec.name] <= spec.high
        assert isinstance(params["batch_size"], int)
        assert space.decode(space.encode(params))["batch_size"] == params["batch_size"]
    with pytest.raises(ValueError):
        space.decode(np.zeros(3))


@pytest.mark.parametrize(
    "bounds",
    [
        {},
        {"a": {"low": 0.0}},
        {"a": {"low": 0.0, "high": 1.0, "step": 1}},
        {"a": "0..1"},
        {"a": {"low": True, "high": 2}},
    ],
)
def test_from_bounds_rejects_malformed(bounds: dict) -> None:
    with pytest.raises(ValueError):
        ParamSpace.from_bounds(bounds)


def test_spaces_are_independent() -> None:
    a = ParamSpace.from_bounds({"x": {"low": 0, "high": 1}})
    b = ParamSpace.from_bounds({"y": {"low": 5, "high": 6}})
    assert a.names == ("x",)
    assert b.names == ("y",)
