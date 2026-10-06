"""Unit tests for the validated demand series (PLAN 1.7, A8, OQ-12)."""

from __future__ import annotations

import math
import pathlib
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from abmem.services.market.demand import DemandDataError, DemandSeries


def _write_xlsx(
    path: pathlib.Path, demand: list[float], other: list[float] | None = None
) -> pathlib.Path:
    """Write a two-column workbook with a DEMAND column."""
    other = other if other is not None else list(range(len(demand)))
    pd.DataFrame({"OTHER": other, "DEMAND": demand}).to_excel(path, index=False)
    return path


@pytest.fixture
def xlsx_with_trailing_nan(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_xlsx(
        tmp_path / "demand.xlsx",
        [28929.45, 27713.39, 26342.91, math.nan, math.nan, math.nan],
    )


def test_from_excel_drops_trailing_nan_rows(
    xlsx_with_trailing_nan: pathlib.Path,
) -> None:
    series = DemandSeries.from_excel(xlsx_with_trailing_nan)
    assert len(series.values) == 3
    assert "demand.xlsx" in series.source and "DEMAND" in series.source
    series.validate_horizon(3)


def test_at_one_is_first_row(xlsx_with_trailing_nan: pathlib.Path) -> None:
    series = DemandSeries.from_excel(xlsx_with_trailing_nan)
    assert series.at(1) == Decimal("28929.45")
    assert series.at(3) == Decimal("26342.91")
    assert isinstance(series.at(2), Decimal)


@pytest.mark.parametrize("period", [0, -1, 4])
def test_at_out_of_range_raises(
    xlsx_with_trailing_nan: pathlib.Path, period: int
) -> None:
    series = DemandSeries.from_excel(xlsx_with_trailing_nan)
    with pytest.raises(IndexError, match="outside the demand series"):
        series.at(period)


def test_horizon_too_long(xlsx_with_trailing_nan: pathlib.Path) -> None:
    series = DemandSeries.from_excel(xlsx_with_trailing_nan)
    with pytest.raises(DemandDataError) as info:
        series.validate_horizon(4)
    message = str(info.value)
    assert "demand.xlsx" in message and "DEMAND" in message
    assert "4 periods" in message and "period 4" in message


def test_nan_inside_horizon(tmp_path: pathlib.Path) -> None:
    series = DemandSeries.from_excel(
        _write_xlsx(tmp_path / "d.xlsx", [10.0, math.nan, 30.0])
    )
    assert len(series.values) == 3
    series.validate_horizon(1)
    with pytest.raises(DemandDataError, match="period 2"):
        series.validate_horizon(3)


@pytest.mark.parametrize("bad", [-5.0, 0.0, math.inf])
def test_negative_zero_or_infinite_value_rejected(bad: float) -> None:
    series = DemandSeries(values=np.array([10.0, 20.0, bad, 40.0]), source="mem")
    series.validate_horizon(2)
    with pytest.raises(DemandDataError, match="period 3"):
        series.validate_horizon(4)


def test_validate_horizon_zero_and_negative() -> None:
    series = DemandSeries(values=np.array([10.0]), source="mem")
    series.validate_horizon(0)
    with pytest.raises(ValueError):
        series.validate_horizon(-1)


def test_empty_series() -> None:
    series = DemandSeries(values=np.array([]), source="mem")
    with pytest.raises(DemandDataError):
        series.validate_horizon(1)
    with pytest.raises(IndexError):
        series.at(1)


def test_all_empty_column(tmp_path: pathlib.Path) -> None:
    series = DemandSeries.from_excel(
        _write_xlsx(tmp_path / "e.xlsx", [math.nan, math.nan])
    )
    assert len(series.values) == 0


def test_missing_column_and_missing_file(tmp_path: pathlib.Path) -> None:
    path = _write_xlsx(tmp_path / "d.xlsx", [1.0, 2.0])
    with pytest.raises(DemandDataError, match="no column"):
        DemandSeries.from_excel(path, column="NOPE")
    with pytest.raises(DemandDataError):
        DemandSeries.from_excel(tmp_path / "missing.xlsx")


def test_at_non_finite_value_raises() -> None:
    series = DemandSeries(values=np.array([math.nan]), source="mem")
    with pytest.raises(DemandDataError):
        series.at(1)


def test_values_are_read_only_copy() -> None:
    raw = np.array([1.0, 2.0])
    series = DemandSeries(values=raw, source="mem")
    raw[0] = 99.0
    assert series.values[0] == 1.0
    with pytest.raises(ValueError):
        series.values[0] = 5.0
    with pytest.raises(ValueError):
        DemandSeries(values=np.ones((2, 2)), source="mem")


def test_window_uses_period_numbers() -> None:
    series = DemandSeries(values=np.array([1.0, 2.0, 3.0, 4.0]), source="mem")
    assert series.window(1, 3).tolist() == [1.0, 2.0]
    assert series.window(2, 5).tolist() == [2.0, 3.0, 4.0]
    assert series.window(3, 3).tolist() == []
    for start, stop in [(0, 2), (3, 2), (1, 6)]:
        with pytest.raises(IndexError):
            series.window(start, stop)


def test_real_market_data_has_8808_valid_rows() -> None:
    from abmem.constants.paths import DATA_DIR

    series = DemandSeries.from_excel(DATA_DIR / "marketData.xlsx")
    assert len(series.values) == 8808
    series.validate_horizon(8808)
    with pytest.raises(DemandDataError):
        series.validate_horizon(8809)


def test_get_demand_requires_loaded_series(monkeypatch: pytest.MonkeyPatch) -> None:
    from abmem.services.market import market_service

    monkeypatch.setattr(market_service, "demandSeries", None)
    with pytest.raises(RuntimeError):
        market_service.getDemand(1)
    monkeypatch.setattr(
        market_service,
        "demandSeries",
        DemandSeries(values=np.array([5.0, 6.0]), source="mem"),
    )
    assert market_service.getDemand(2) == Decimal("6.0")


@pytest.mark.django_db
def test_demand_error_leaves_simulation_created(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from abmem.models.enums import MarketState, SimulationState
    from abmem.services.market import market_service
    from abmem.services.simulation import simulation_service
    from tests.factories import make_market, make_simulation

    simulation = make_simulation(period_number=10**6)
    market = make_market(simulation=simulation)
    simulation.refresh_from_db()

    def _fail_run(_market: object) -> None:
        raise AssertionError("market run must not start")

    monkeypatch.setattr(market_service, "run", _fail_run)

    result = simulation_service.run(simulation)

    assert result == []
    simulation.refresh_from_db()
    market.refresh_from_db()
    assert simulation.state == SimulationState.CREATED
    assert market.state == MarketState.CREATED
