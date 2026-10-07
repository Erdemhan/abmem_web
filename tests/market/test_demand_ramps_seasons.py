"""Day/night transition ramps and the summer/winter split of a demand profile (EQ-02c)."""

from __future__ import annotations

import copy
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pytest
from django.core.exceptions import ValidationError

from abmem.services.market.demand_generator import (
    DEMAND_DECIMALS,
    U_EPSILON,
    DemandProfile,
    Distribution,
    Segment,
    _quantile_cross_moment,
    day_weights,
    generate_demand,
    night_mask,
)

DAY = {
    "distribution": "NORMAL",
    "mean": 42000.0,
    "std": 4000.0,
    "min": 30000.0,
    "max": 55000.0,
}
NIGHT = {
    "distribution": "LOGNORMAL",
    "mean": 33000.0,
    "std": 3000.0,
    "min": 24000.0,
    "max": 45000.0,
}
WINTER_DAY = {"distribution": "UNIFORM", "min": 45000.0, "max": 52000.0}
WINTER_NIGHT = {
    "distribution": "NORMAL",
    "mean": 37000.0,
    "std": 2500.0,
    "min": 30000.0,
    "max": 44000.0,
}


def _raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "seed": 11,
        "autocorrelation": 0.9,
        "day_start_hour": 8,
        "day_end_hour": 20,
        "day": dict(DAY),
        "night": dict(NIGHT),
    }
    raw.update(overrides)
    return raw


def _seasons(**overrides: Any) -> dict[str, Any]:
    seasons: dict[str, Any] = {
        "summer_end_period": 24 * 10,
        "transition_days": 4,
        "winter": {
            "day_start_hour": 9,
            "day_end_hour": 18,
            "morning_ramp_hours": 2,
            "evening_ramp_hours": 3,
            "day": dict(WINTER_DAY),
            "night": dict(WINTER_NIGHT),
        },
    }
    seasons.update(overrides)
    return seasons


def _reference_without_ramps(profile: DemandProfile, periods: int) -> np.ndarray:
    """The PLAN 2.8 algorithm: one segment per hour chosen by the night mask."""
    rng = np.random.default_rng(profile.seed)
    phi = profile.autocorrelation
    scale = math.sqrt(1.0 - phi * phi)
    night = night_mask(periods, profile.day_start_hour, profile.day_end_hour)
    values = np.empty(periods)
    z = 0.0
    for index in range(periods):
        eps = float(rng.standard_normal())
        z = eps if index == 0 else phi * z + scale * eps
        u = min(max(NormalDist().cdf(z), U_EPSILON), 1.0 - U_EPSILON)
        segment = profile.night if night[index] else profile.day
        demand = round(segment.quantile(u), DEMAND_DECIMALS)
        values[index] = min(max(demand, segment.min), segment.max)
    return values


# ---------------------------------------------------------------- day weights


def test_zero_ramps_give_the_exact_day_mask() -> None:
    for start, end in ((8, 20), (0, 12), (6, 24), (21, 23)):
        weights = day_weights(start, end)
        assert set(weights.tolist()) <= {0.0, 1.0}
        assert (weights == 0.0).tolist() == night_mask(24, start, end).tolist()


def test_ramp_weights_are_symmetric_and_bounded() -> None:
    weights = day_weights(8, 20, 4, 4)
    assert np.all((weights >= 0.0) & (weights <= 1.0))
    # Morning: hours 6..9 rise, symmetric around the 8:00 boundary.
    assert weights[6] == pytest.approx(1.0 - weights[9])
    assert weights[7] == pytest.approx(1.0 - weights[8])
    assert weights[5] == 0.0 and weights[10] == 1.0
    # Evening mirror image around 20:00.
    assert weights[18] == pytest.approx(weights[9]) and weights[21] == pytest.approx(
        weights[6]
    )
    assert np.all(np.diff(weights[4:12]) >= 0.0) and np.all(
        np.diff(weights[16:24]) <= 0.0
    )


def test_ramp_wraps_around_midnight() -> None:
    weights = day_weights(2, 22, 4, 4)
    # Night is 22..2. Each ramp reaches across midnight, so 23:00 and 00:00
    # sit 1.5 h from the evening and the morning boundary respectively.
    assert weights[0] == pytest.approx(weights[23])
    assert 0.0 < weights[1] < 1.0 and 0.0 < weights[23] < 1.0
    assert 0.0 < weights[0] < weights[1]


# ---------------------------------------------------------------- generation


def test_profile_without_ramps_and_seasons_matches_the_plan_2_8_algorithm() -> None:
    profile = DemandProfile.from_dict(_raw())
    assert (
        generate_demand(profile, 500).tobytes()
        == _reference_without_ramps(profile, 500).tobytes()
    )


def test_ramps_soften_the_boundary_step() -> None:
    hard = DemandProfile.from_dict(_raw(autocorrelation=0.95))
    soft = DemandProfile.from_dict(
        _raw(autocorrelation=0.95, morning_ramp_hours=4, evening_ramp_hours=4)
    )
    periods = 24 * 300
    hours = np.arange(1, periods) % 24  # hour of the later period of each step

    def boundary_step(series: np.ndarray) -> float:
        steps = np.abs(np.diff(series))
        return float(steps[(hours == 8) | (hours == 20)].mean())

    assert boundary_step(generate_demand(soft, periods)) < 0.5 * boundary_step(
        generate_demand(hard, periods)
    )


def test_transition_values_stay_inside_the_weighted_bounds() -> None:
    profile = DemandProfile.from_dict(_raw(morning_ramp_hours=6, evening_ramp_hours=6))
    weights = profile.weights(24 * 50)
    values = generate_demand(profile, 24 * 50)
    mins = np.array([s.min for s in profile.segments()])
    maxs = np.array([s.max for s in profile.segments()])
    assert np.all(values >= weights @ mins - 1e-9) and np.all(
        values <= weights @ maxs + 1e-9
    )


def test_prefix_consistency_with_ramps_and_seasons() -> None:
    profile = DemandProfile.from_dict(
        _raw(morning_ramp_hours=3, evening_ramp_hours=4, seasons=_seasons())
    )
    long = generate_demand(profile, 24 * 30)
    assert generate_demand(profile, 24 * 12).tobytes() == long[: 24 * 12].tobytes()


def test_abrupt_season_switch_after_the_summer_end_period() -> None:
    profile = DemandProfile.from_dict(
        _raw(seasons=_seasons(summer_end_period=100, transition_days=0))
    )
    v = profile.seasons.winter_weight(200)
    assert np.all(v[:100] == 0.0) and np.all(v[100:] == 1.0)
    weights = profile.weights(200)
    assert np.all(weights[:100, 2:] == 0.0) and np.all(weights[100:, :2] == 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_season_transition_is_centred_on_the_summer_end() -> None:
    profile = DemandProfile.from_dict(
        _raw(seasons=_seasons(summer_end_period=240, transition_days=4))
    )
    v = profile.seasons.winter_weight(480)
    # Width 96 h centred on t = 240: from period 193 to 288.
    assert v[191] == 0.0 and v[288] == 1.0
    assert v[239] == pytest.approx(1.0 - v[240])
    assert np.all(np.diff(v) >= 0.0)


def test_winter_equal_to_summer_reproduces_the_plain_series() -> None:
    plain = _raw(morning_ramp_hours=2, evening_ramp_hours=2)
    winter = {
        key: copy.deepcopy(plain[key])
        for key in ("day_start_hour", "day_end_hour", "day", "night")
    }
    winter.update(morning_ramp_hours=2, evening_ramp_hours=2)
    seasonal = dict(plain, seasons=_seasons(winter=winter))
    a = generate_demand(DemandProfile.from_dict(plain), 24 * 20)
    b = generate_demand(DemandProfile.from_dict(seasonal), 24 * 20)
    assert np.allclose(a, b, atol=2e-3)


# ---------------------------------------------------------------- scaling


def test_cross_moment_matches_midpoint_integration() -> None:
    first = Segment(
        Distribution.NORMAL, min=30000.0, max=55000.0, mean=42000.0, std=4000.0
    )
    second = Segment(
        Distribution.LOGNORMAL, min=24000.0, max=45000.0, mean=33000.0, std=3000.0
    )
    n = 200_000
    u = (np.arange(n) + 0.5) / n
    numeric = float(
        np.mean([first.quantile(float(p)) * second.quantile(float(p)) for p in u])
    )
    assert _quantile_cross_moment(first, second) == pytest.approx(numeric, rel=1e-6)


def test_ramp_scaling_matches_a_long_series() -> None:
    profile = DemandProfile.from_dict(
        _raw(autocorrelation=0.0, morning_ramp_hours=4, evening_ramp_hours=6)
    )
    series = generate_demand(profile, 24 * 4000)
    mean, std = profile.scaling()
    assert mean == pytest.approx(float(series.mean()), rel=2e-3)
    assert std == pytest.approx(float(series.std()), rel=1e-2)


def test_seasonal_scaling_matches_a_long_series() -> None:
    periods = 24 * 4000
    raw = _raw(
        autocorrelation=0.0,
        morning_ramp_hours=3,
        seasons=_seasons(summer_end_period=periods // 3, transition_days=20),
    )
    profile = DemandProfile.from_dict(raw)
    series = generate_demand(profile, periods)
    mean, std = profile.scaling(periods)
    assert mean == pytest.approx(float(series.mean()), rel=2e-3)
    assert std == pytest.approx(float(series.std()), rel=1e-2)


def test_seasonal_scaling_needs_the_number_of_periods() -> None:
    profile = DemandProfile.from_dict(_raw(seasons=_seasons()))
    with pytest.raises(ValueError):
        profile.scaling()


# ---------------------------------------------------------------- schema


def test_defaults_and_round_trip() -> None:
    profile = DemandProfile.from_dict(_raw())
    stored = profile.to_dict()
    assert stored["morning_ramp_hours"] == 0 and stored["evening_ramp_hours"] == 0
    assert stored["seasons"] is None
    seasonal = DemandProfile.from_dict(_raw(seasons=_seasons()))
    assert DemandProfile.from_dict(seasonal.to_dict()) == seasonal


@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"morning_ramp_hours": 13}, "morning_ramp_hours"),
        ({"evening_ramp_hours": -1}, "evening_ramp_hours"),
        ({"morning_ramp_hours": 1.5}, "morning_ramp_hours"),
        (
            {
                "day_start_hour": 8,
                "day_end_hour": 10,
                "morning_ramp_hours": 3,
                "evening_ramp_hours": 2,
            },
            "evening_ramp_hours",
        ),
        ({"seasons": _seasons(summer_end_period=0)}, "seasons.summer_end_period"),
        (
            {"seasons": _seasons(summer_end_period=24, transition_days=3)},
            "seasons.transition_days",
        ),
        ({"seasons": _seasons(winter=None)}, "seasons.winter"),
        ({"seasons": dict(_seasons(), extra=1)}, "seasons"),
        ({"seasons": "yes"}, "seasons"),
    ],
)
def test_invalid_ramps_and_seasons(overrides: dict[str, Any], key: str) -> None:
    with pytest.raises(ValidationError) as info:
        DemandProfile.from_dict(_raw(**overrides))
    assert key in info.value.message_dict


def test_winter_segment_errors_are_keyed_and_labelled() -> None:
    seasons = _seasons()
    seasons["winter"]["night"]["std"] = -5
    with pytest.raises(ValidationError) as info:
        DemandProfile.from_dict(_raw(seasons=seasons))
    messages = info.value.message_dict["seasons.winter.night.std"]
    assert messages[0].startswith("Winter night")


def test_winter_ramps_overlap_error() -> None:
    seasons = _seasons()
    seasons["winter"].update(
        day_start_hour=10, day_end_hour=12, morning_ramp_hours=3, evening_ramp_hours=3
    )
    with pytest.raises(ValidationError) as info:
        DemandProfile.from_dict(_raw(seasons=seasons))
    assert "seasons.winter.evening_ramp_hours" in info.value.message_dict
