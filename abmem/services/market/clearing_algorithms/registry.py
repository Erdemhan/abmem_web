"""Registry of market clearing algorithms discovered from BaseClearingAlgorithm subclasses.

The first query imports every public module of
``abmem.services.market.clearing_algorithms``, so a new clearing algorithm
module dropped into that package becomes selectable without editing this file
(PLAN 2.10, ``docs/extending_algorithms.md``). Keys are the strings stored in
``Market.clearing_algorithm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from abmem.services.plugin_registry import PluginRegistry

if TYPE_CHECKING:
    from .base import BaseClearingAlgorithm

__all__ = [
    "CLEARING_ALGORITHMS",
    "DEFAULT_CLEARING_ALGORITHM",
    "ClearingAlgorithmInfo",
    "available_clearing_algorithms",
    "clearing_label_for",
    "get_clearing_algorithm",
    "is_known_clearing",
]

DEFAULT_CLEARING_ALGORITHM: Final[str] = "MERIT_ORDER"

CLEARING_ALGORITHMS: Final[PluginRegistry[type[BaseClearingAlgorithm]]] = (
    PluginRegistry(
        "clearing algorithm",
        packages=("abmem.services.market.clearing_algorithms",),
    )
)


@dataclass(frozen=True)
class ClearingAlgorithmInfo:
    """Description of one registered clearing algorithm.

    Attributes:
        key: Stored key, for example ``"MERIT_ORDER"``.
        label: Display name.
        description: One-sentence description shown as help text.
    """

    key: str
    label: str
    description: str


def available_clearing_algorithms() -> tuple[ClearingAlgorithmInfo, ...]:
    """Return every registered clearing algorithm.

    Returns:
        The default algorithm first, then the others ordered by label (then
        key).
    """
    infos = [
        ClearingAlgorithmInfo(
            key=key,
            label=str(cls.label or key),
            description=str(cls.description or ""),
        )
        for key, cls in CLEARING_ALGORITHMS.classes().items()
    ]
    infos.sort(
        key=lambda info: (info.key != DEFAULT_CLEARING_ALGORITHM, info.label, info.key)
    )
    return tuple(infos)


def get_clearing_algorithm(key: str) -> BaseClearingAlgorithm:
    """Return a new instance of the clearing algorithm ``key``.

    Args:
        key: Clearing algorithm key.

    Returns:
        An instance of the registered class.

    Raises:
        KeyError: If ``key`` is unknown. The message lists the known keys.
    """
    return CLEARING_ALGORITHMS.get(key)()


def is_known_clearing(key: object) -> bool:
    """Return True if ``key`` is a registered clearing algorithm key."""
    return CLEARING_ALGORITHMS.is_known(key)


def clearing_label_for(key: object) -> str:
    """Return the display label of ``key``, the key itself when unknown.

    Args:
        key: Clearing algorithm key.
    """
    if not is_known_clearing(key):
        return "" if key is None else str(key)
    cls = CLEARING_ALGORITHMS.get(str(key))
    return str(cls.label or key)
