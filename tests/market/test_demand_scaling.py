"""Observation scaling of generated demand (FORMULATION EQ-02b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from abmem.services.agent.observation import (
    DEMAND_SCALING_MEAN,
    DEMAND_SCALING_STD,
    ObservationNormalizer,
)
from abmem.services.market.demand_generator import (
    DemandProfile,
    Distribution,
    Segment,
    generate_demand,
)


def _numeric_moments(segment: Segment, n: int = 400_000) -> tuple[float, float]:
    """Mean and variance of the truncated marginal by midpoint quantile integration."""
    u = (np.arange(n) + 0.5) / n
    x = np.array([segment.quantile(float(p)) for p in u])
    return float(x.mean()), float(x.var())


@pytest.mark.parametrize(
    "segment",
    [
        Segment(Distribution.UNIFORM, min=100.0, max=300.0),
        Segment(
            Distribution.NORMAL, min=30000.0, max=55000.0, mean=42000.0, std=4000.0
        ),
        Segment(Distribution.NORMAL, min=900.0, max=1300.0, mean=1000.0, std=250.0),
        Segment(
            Distribution.LOGNORMAL, min=22000.0, max=40000.0, mean=30000.0, std=3000.0
        ),
        Segment(Distribution.LOGNORMAL, min=200.0, max=900.0, mean=300.0, std=200.0),
    ],
    ids=[
        "uniform",
        "normal-wide",
        "normal-tight",
        "lognormal-wide",
        "lognormal-skewed",
    ],
)
def test_truncated_moments_match_numerical_integration(segment: Segment) -> None:
    mean, variance = segment.moments()
    num_mean, num_var = _numeric_moments(segment)
    assert mean == pytest.approx(num_mean, rel=1e-4)
    assert variance == pytest.approx(num_var, rel=2e-3)


def test_uniform_moments_are_exact() -> None:
    mean, variance = Segment(Distribution.UNIFORM, min=10.0, max=40.0).moments()
    assert mean == 25.0
    assert variance == pytest.approx(900.0 / 12.0)


def _profile(
    day: Segment, night: Segment, start: int = 8, end: int = 20
) -> DemandProfile:
    return DemandProfile.from_dict(
        {
            "seed": 7,
            "day_start_hour": start,
            "day_end_hour": end,
            "autocorrelation": 0.0,
            "day": day.to_dict(),
            "night": night.to_dict(),
        }
    )


def test_profile_scaling_is_the_hour_weighted_mixture() -> None:
    day = Segment(Distribution.UNIFORM, min=100.0, max=300.0)
    night = Segment(Distribution.UNIFORM, min=40.0, max=60.0)
    profile = _profile(day, night, 6, 24)  # 18 day hours, 6 night hours
    mean, std = profile.scaling()
    w_day, w_night = 18 / 24, 6 / 24
    m_d, v_d = 200.0, 200.0**2 / 12
    m_n, v_n = 50.0, 20.0**2 / 12
    expected_mean = w_day * m_d + w_night * m_n
    expected_var = w_day * (v_d + m_d**2) + w_night * (v_n + m_n**2) - expected_mean**2
    assert mean == pytest.approx(expected_mean)
    assert std == pytest.approx(math.sqrt(expected_var))


def test_profile_scaling_matches_a_long_generated_series() -> None:
    day = Segment(
        Distribution.NORMAL, min=30000.0, max=55000.0, mean=42000.0, std=4000.0
    )
    night = Segment(
        Distribution.LOGNORMAL, min=22000.0, max=40000.0, mean=30000.0, std=3000.0
    )
    profile = _profile(day, night)
    series = generate_demand(profile, 24 * 4000)
    mean, std = profile.scaling()
    assert mean == pytest.approx(float(series.mean()), rel=2e-3)
    assert std == pytest.approx(float(series.std()), rel=1e-2)


def test_normaliser_default_keeps_file_constants() -> None:
    normaliser = ObservationNormalizer.for_market((1, 24), 0.0, 200.0)
    assert normaliser.mean[0] == DEMAND_SCALING_MEAN
    assert normaliser.scale[0] == DEMAND_SCALING_STD


def test_normaliser_uses_given_demand_scaling() -> None:
    normaliser = ObservationNormalizer.for_market(
        (1, 24), 0.0, 200.0, demand_mean=500.0, demand_std=50.0
    )
    assert normaliser.mean[0] == 500.0
    assert normaliser.scale[0] == 50.0
    assert normaliser.transform(np.array([550.0, 100.0, 100.0]))[0] == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_normaliser_rejects_invalid_demand_std(bad: float) -> None:
    with pytest.raises(ValueError):
        ObservationNormalizer.for_market(
            (1, 24), 0.0, 200.0, demand_mean=500.0, demand_std=bad
        )
