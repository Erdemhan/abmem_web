"""Registers the ``slow`` marker. Slow tests run only with ABMEM_RUN_SLOW=1."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: long-running benchmark, enabled with ABMEM_RUN_SLOW=1"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get("ABMEM_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow benchmark, set ABMEM_RUN_SLOW=1 to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
