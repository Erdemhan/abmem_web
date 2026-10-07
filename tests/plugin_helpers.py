"""Temporary agent and clearing algorithm classes for tests (PLAN 2.10).

A class with a ``key`` registers itself when it is created. These context
managers create such a class for the duration of one test and remove its key
from the registry afterwards, so the registries stay unchanged between tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from abmem.services.algorithms import registry as algorithm_registry
from abmem.services.algorithms.algorithm_config import RandomConfig
from abmem.services.algorithms.base import ActionResult, BaseAgentAlgorithm
from abmem.services.market.clearing_algorithms import registry as clearing_registry
from abmem.services.market.clearing_algorithms.config import ClearingConfig
from abmem.services.market.clearing_algorithms.merit_order import MeritOrderClearing


@dataclass(frozen=True)
class ParamClearingConfig(ClearingConfig):
    """Parameters of the test clearing algorithm (none changes the clearing).

    Attributes:
        markup: Test value, >= 0.
        rounds: Test count, >= 1 and at most 10 times markup + 5.
        strict: Test switch.
        weights: Test integers.
        cap: Optional test bound.
    """

    markup: float = 0.5
    rounds: int = 3
    strict: bool = False
    weights: tuple[int, ...] = (1, 2)
    cap: float | None = None

    def _validate(self) -> None:
        """Validate the test ranges."""
        super()._validate()
        if self.markup < 0:
            raise ValueError(f"markup must be >= 0, got {self.markup}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {self.rounds}")
        if self.rounds > 10 * self.markup + 5:
            raise ValueError(
                f"rounds ({self.rounds}) must be <= 10 * markup + 5 ({self.markup})"
            )


@contextmanager
def temporary_param_clearing_algorithm(
    key: str = "TEST_PARAM_CLEARING", label: str = "Test clearing with parameters"
) -> Iterator[type]:
    """Register a merit order copy with :class:`ParamClearingConfig` under ``key``.

    Args:
        key: Registry key.
        label: Display label.

    Yields:
        The registered class.
    """
    cls = type(
        "TemporaryParamClearing",
        (MeritOrderClearing,),
        {
            "key": key,
            "label": label,
            "description": "Merit order with test parameters.",
            "config_class": ParamClearingConfig,
            "__module__": __name__,
        },
    )
    try:
        yield cls
    finally:
        clearing_registry.CLEARING_ALGORITHMS.unregister(key)


@contextmanager
def temporary_clearing_algorithm(
    key: str = "TEST_CLEARING", label: str = "Test clearing"
) -> Iterator[type]:
    """Register a copy of the merit order under ``key`` while the block runs.

    Args:
        key: Registry key.
        label: Display label.

    Yields:
        The registered class.
    """
    cls = type(
        "TemporaryClearing",
        (MeritOrderClearing,),
        {
            "key": key,
            "label": label,
            "description": "Merit order registered by a test.",
            "__module__": __name__,
        },
    )
    try:
        yield cls
    finally:
        clearing_registry.CLEARING_ALGORITHMS.unregister(key)


@contextmanager
def temporary_agent_algorithm(
    key: str = "TEST_AGENT_ALGO", label: str = "Test algorithm"
) -> Iterator[type]:
    """Register a non-learning algorithm bidding the middle price under ``key``.

    Args:
        key: Registry key.
        label: Display label.

    Yields:
        The registered class.
    """

    def select_action(
        self: BaseAgentAlgorithm, obs: np.ndarray, explore: bool
    ) -> ActionResult:
        return ActionResult(u=np.zeros(self.spec.action_dim, dtype=np.float32))

    cls = type(
        "TemporaryAgentAlgorithm",
        (BaseAgentAlgorithm,),
        {
            "key": key,
            "label": label,
            "description": "Bids the middle of the bid bounds (test).",
            "is_learning": False,
            "is_on_policy": False,
            "config_class": RandomConfig,
            "select_action": select_action,
            "__module__": __name__,
        },
    )
    try:
        yield cls
    finally:
        algorithm_registry.AGENT_ALGORITHMS.unregister(key)
