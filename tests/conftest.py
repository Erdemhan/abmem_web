"""Fixtures shared by the whole test suite."""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

import pytest
from django.test import override_settings


@pytest.fixture(autouse=True)
def checkpoint_dir(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """Write policy checkpoints of every test into its own temporary directory.

    ``override_settings`` is used instead of the ``settings`` fixture because
    some tests parametrize an argument named ``settings``.

    Yields:
        The checkpoint directory of the test (``settings.ABMEM_CHECKPOINT_DIR``).
    """
    directory = tmp_path / "checkpoints"
    with override_settings(ABMEM_CHECKPOINT_DIR=directory):
        yield directory
