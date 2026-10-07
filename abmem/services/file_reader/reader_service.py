"""Excel reader for data files in ``abmem/data``."""

from __future__ import annotations

import pandas as pd

from ...constants.paths import DATA_DIR


def readExcel(path: str, columns: list[str], map: list[str] | None = None) -> dict:
    """Read columns of an Excel file in ``DATA_DIR``.

    Args:
        path: File name relative to ``DATA_DIR``.
        columns: Column names to read.
        map: Keys of the returned dict, one per column. Defaults to the
            column names.

    Returns:
        Mapping of key to the column values as a list.

    Raises:
        OSError: If the file cannot be read.
        KeyError: If a column is missing.
    """
    keys = list(map) if map else list(columns)
    frame = pd.read_excel(DATA_DIR / path)
    return {key: frame[column].tolist() for column, key in zip(columns, keys)}
