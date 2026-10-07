"""Synthetic demand generation (PLAN 2.8, spec demand_generation)."""

from __future__ import annotations

import copy
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from django.core.exceptions import ValidationError

from abmem.services.market.demand import DemandSeries
from abmem.services.market.demand_generator import (
    DemandProfile,
    generate_demand,
    night_mask,
    normalize_profile,
)

WIDE: dict[str, Any] = {"distribution": "NORMAL", "mean": 1000.0, "std": 100.0, "min": 1.0, "max": 100000.0}


def profile_dict(**overrides: Any) -> dict[str, Any]:
    """Return a valid version-1 profile mapping with ``overrides`` applied."""
    data: dict[str, Any] = {
        "version": 1,
        "seed": 2024,
        "day_start_hour": 8,
        "day_end_hour": 20,
        "autocorrelation": 0.8,
        "day": {"distribution": "NORMAL", "mean": 42000.0, "std": 4000.0, "min": 30000.0, "max": 55000.0},
        "night": {"distribution": "UNIFORM", "min": 25000.0, "max": 30000.0},
    }
    data.update(overrides)
    return data


def make_profile(**overrides: Any) -> DemandProfile:
    return DemandProfile.from_dict(profile_dict(**overrides))


def lag1(values: np.ndarray) -> float:
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


# determinism and prefix consistency ------------------------------------------------


def test_same_profile_gives_identical_series() -> None:
    first = generate_demand(make_profile(), 500)
    second = generate_demand(make_profile(), 500)
    assert first.dtype == np.float64
    assert first.tobytes() == second.tobytes()


def test_different_seeds_give_different_series() -> None:
    assert not np.array_equal(
        generate_demand(make_profile(seed=1), 100), generate_demand(make_profile(seed=2), 100)
    )


@pytest.mark.parametrize("short, long", [(1, 24), (168, 336), (168, 500), (336, 500), (499, 500)])
def test_shorter_horizon_is_a_bitwise_prefix_of_a_longer_one(short: int, long: int) -> None:
    profile = make_profile()
    assert generate_demand(profile, long)[:short].tobytes() == generate_demand(profile, short).tobytes()


def test_extending_the_horizon_keeps_every_earlier_value() -> None:
    profile = make_profile(autocorrelation=0.95)
    series = [generate_demand(profile, n) for n in (168, 336, 500)]
    assert series[1][:168].tobytes() == series[0].tobytes()
    assert series[2][:336].tobytes() == series[1].tobytes()


def test_series_is_identical_in_a_fresh_process() -> None:
    code = (
        "import json, sys\n"
        "from abmem.services.market.demand_generator import DemandProfile, generate_demand\n"
        "p = DemandProfile.from_dict(json.loads(sys.argv[1]))\n"
        "print(generate_demand(p, 300).tobytes().hex())\n"
    )
    import json

    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(profile_dict())],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == generate_demand(make_profile(), 300).tobytes().hex()


def test_zero_periods_and_negative_periods() -> None:
    assert generate_demand(make_profile(), 0).shape == (0,)
    with pytest.raises(ValueError):
        generate_demand(make_profile(), -1)


# bounds, rounding and segments ---------------------------------------------------


@pytest.mark.parametrize("distribution", ["NORMAL", "UNIFORM", "LOGNORMAL"])
def test_values_lie_in_the_segment_bounds_and_are_rounded(distribution: str) -> None:
    day = {"distribution": distribution, "mean": 500.0, "std": 400.0, "min": 300.0, "max": 700.0}
    night = {"distribution": distribution, "mean": 150.0, "std": 200.0, "min": 100.0, "max": 200.0}
    profile = make_profile(day=day, night=night)
    values = generate_demand(profile, 2000)
    mask = night_mask(2000, 8, 20)
    assert np.all(values > 0)
    assert np.all((values[~mask] >= 300.0) & (values[~mask] <= 700.0))
    assert np.all((values[mask] >= 100.0) & (values[mask] <= 200.0))
    assert np.allclose(values, np.round(values, 3), rtol=0, atol=1e-9)


def test_tiny_minimum_stays_positive_after_rounding() -> None:
    segment = {"distribution": "UNIFORM", "min": 0.0001, "max": 0.0009}
    values = generate_demand(make_profile(day=segment, night=segment), 200)
    assert np.all(values > 0)
    assert np.all((values >= 0.0001) & (values <= 0.0009))


@pytest.mark.parametrize(
    "start, end, day_hours",
    [(8, 20, list(range(8, 20))), (0, 6, list(range(0, 6))), (18, 24, list(range(18, 24))), (0, 23, list(range(23)))],
)
def test_segment_assignment_by_hour(start: int, end: int, day_hours: list[int]) -> None:
    day = {"distribution": "UNIFORM", "min": 1000.0, "max": 2000.0}
    night = {"distribution": "UNIFORM", "min": 10.0, "max": 20.0}
    profile = make_profile(day_start_hour=start, day_end_hour=end, day=day, night=night)
    values = generate_demand(profile, 72)
    for index, value in enumerate(values):
        hour = index % 24  # period t = index + 1 has hour (t - 1) mod 24
        assert (value >= 1000.0) == (hour in day_hours), (index, hour, value)
    assert profile.day_hours + profile.night_hours == 24
    assert int(night_mask(24, start, end).sum()) == profile.night_hours


# correlation and marginals -------------------------------------------------------


def test_phi_zero_is_uncorrelated_and_phi_high_is_persistent() -> None:
    def series(phi: float) -> np.ndarray:
        return generate_demand(make_profile(autocorrelation=phi, day=WIDE, night=WIDE), 5000)

    assert abs(lag1(series(0.0))) < 0.05
    assert 0.4 < lag1(series(0.5)) < 0.6
    assert lag1(series(0.9)) > 0.8


@pytest.mark.parametrize("distribution", ["NORMAL", "LOGNORMAL"])
def test_means_and_std_near_requested_with_wide_bounds(distribution: str) -> None:
    segment = dict(WIDE, distribution=distribution)
    values = generate_demand(make_profile(autocorrelation=0.0, day=segment, night=segment), 20000)
    assert values.mean() == pytest.approx(1000.0, rel=0.01)
    assert values.std() == pytest.approx(100.0, rel=0.05)


def test_uniform_covers_the_range() -> None:
    segment = {"distribution": "UNIFORM", "min": 100.0, "max": 200.0}
    values = generate_demand(make_profile(autocorrelation=0.0, day=segment, night=segment), 20000)
    assert values.min() < 101.0 and values.max() > 199.0
    assert values.mean() == pytest.approx(150.0, rel=0.01)
    counts, _ = np.histogram(values, bins=10, range=(100.0, 200.0))
    assert counts.min() > 1600


def test_from_profile_builds_a_generated_series() -> None:
    profile = make_profile()
    series = DemandSeries.from_profile(profile, 48)
    assert series.source == "generated"
    assert series.values.tobytes() == generate_demand(profile, 48).tobytes()
    series.validate_horizon(48)


# validation ----------------------------------------------------------------------


def test_round_trip_of_the_stored_form() -> None:
    profile = make_profile()
    stored = profile.to_dict()
    assert stored["night"] == {"distribution": "UNIFORM", "min": 25000.0, "max": 30000.0}
    assert DemandProfile.from_dict(stored) == profile
    without_version = dict(stored)
    del without_version["version"]
    assert DemandProfile.from_dict(without_version) == profile


def test_numeric_strings_are_accepted() -> None:
    data = profile_dict(seed="5", autocorrelation="0.5")
    data["day"] = {"distribution": "normal", "mean": "1000", "std": "10", "min": "900", "max": "1100"}
    profile = DemandProfile.from_dict(data)
    assert profile.seed == 5 and profile.autocorrelation == 0.5
    assert profile.day.mean == 1000.0


def _errors(data: Any) -> dict[str, list[str]]:
    with pytest.raises(ValidationError) as info:
        DemandProfile.from_dict(data)
    return info.value.message_dict


def _with_segment(name: str, **fields: Any) -> dict[str, Any]:
    data = profile_dict()
    segment = copy.deepcopy(data["day"])
    segment.update(fields)
    for key in [key for key, value in fields.items() if value is _DROP]:
        del segment[key]
    data[name] = segment
    return data


_DROP = object()


@pytest.mark.parametrize(
    "data, key",
    [
        (_with_segment("day", std=0.0), "day.std"),
        (_with_segment("night", std=-5.0), "night.std"),
        (_with_segment("day", min=60000.0), "day.max"),
        (_with_segment("day", min=0.0), "day.min"),
        (_with_segment("day", mean=56000.0), "day.mean"),
        (_with_segment("day", mean=30000.0), "day.mean"),
        (_with_segment("day", distribution="GAMMA"), "day.distribution"),
        (_with_segment("day", mean=_DROP), "day.mean"),
        (_with_segment("day", max=_DROP), "day.max"),
        (_with_segment("day", mean="abc"), "day.mean"),
        (_with_segment("day", extra=1), "day"),
        (profile_dict(day_start_hour=8, day_end_hour=8), "day_end_hour"),
        (profile_dict(day_start_hour=0, day_end_hour=24), "day_end_hour"),
        (profile_dict(day_start_hour=10, day_end_hour=5), "day_end_hour"),
        (profile_dict(day_start_hour=24), "day_start_hour"),
        (profile_dict(day_end_hour=25), "day_end_hour"),
        (profile_dict(autocorrelation=1.0), "autocorrelation"),
        (profile_dict(autocorrelation=-0.1), "autocorrelation"),
        (profile_dict(autocorrelation=None), "autocorrelation"),
        (profile_dict(seed=-1), "seed"),
        (profile_dict(seed=2**31), "seed"),
        (profile_dict(seed=1.5), "seed"),
        (profile_dict(seed=True), "seed"),
        (profile_dict(version=2), "version"),
        (profile_dict(unknown=1), "demand_profile"),
        (profile_dict(night=None), "night"),
    ],
)
def test_validation_errors_are_keyed_by_field(data: dict[str, Any], key: str) -> None:
    errors = _errors(data)
    assert key in errors, errors


def test_missing_top_level_fields_are_all_reported() -> None:
    errors = _errors({})
    for key in ("seed", "day_start_hour", "day_end_hour", "autocorrelation", "day", "night"):
        assert key in errors, errors


def test_not_a_mapping() -> None:
    assert "demand_profile" in _errors([1, 2])


def test_normalize_profile_prefixes_keys_and_accepts_none() -> None:
    assert normalize_profile(None) is None
    assert normalize_profile(profile_dict())["version"] == 1
    with pytest.raises(ValidationError) as info:
        normalize_profile(_with_segment("day", std=0.0))
    assert "demand_profile.day.std" in info.value.message_dict
    with pytest.raises(ValidationError) as info:
        normalize_profile("text")
    assert "demand_profile" in info.value.message_dict
