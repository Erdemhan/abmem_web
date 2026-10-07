"""Registry of agent algorithms discovered from the subclasses of BaseAgentAlgorithm.

Every concrete subclass of :class:`~abmem.services.algorithms.base.BaseAgentAlgorithm`
that declares a ``key`` registers itself when its module is imported. The
first registry query imports every public module of this package (except the
``optimization`` subpackage), so a new algorithm module dropped into
``abmem/services/algorithms/`` becomes selectable without editing this file
(PLAN 2.10, ``docs/extending_algorithms.md``).

This module imports neither torch nor the algorithm modules at import time.
Keys are the strings stored in ``Agent.algorithm_type``,
``HPOStudy.algorithm_type``, ``AlgorithmDefault.algorithm_type`` and
``PolicyCheckpoint.algorithm_type``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from abmem.services.plugin_registry import PluginRegistry

if TYPE_CHECKING:
    from .algorithm_config import AlgorithmConfig
    from .base import BaseAgentAlgorithm, PolicySpec

__all__ = [
    "AGENT_ALGORITHMS",
    "DEFAULT_ALGORITHM",
    "AlgorithmInfo",
    "available_algorithms",
    "build_policy",
    "default_config",
    "get_algorithm_class",
    "get_config_class",
    "info_for_class",
    "is_baseline",
    "is_known",
    "label_for",
    "sorted_infos",
]

DEFAULT_ALGORITHM: Final[str] = "DDPG"

AGENT_ALGORITHMS: Final[PluginRegistry[type[BaseAgentAlgorithm]]] = PluginRegistry(
    "agent algorithm",
    packages=("abmem.services.algorithms",),
    exclude=("abmem.services.algorithms.optimization",),
)


@dataclass(frozen=True)
class AlgorithmInfo:
    """Description of one registered agent algorithm.

    Attributes:
        key: Stored key, for example ``"DDPG"``.
        label: Display name.
        description: One-sentence description shown as help text.
        is_learning: True for learning algorithms.
        is_on_policy: True for on-policy learners.
    """

    key: str
    label: str
    description: str
    is_learning: bool
    is_on_policy: bool

    @property
    def is_baseline(self) -> bool:
        """Return True for a non-learning algorithm."""
        return not self.is_learning


def available_algorithms() -> tuple[AlgorithmInfo, ...]:
    """Return every registered agent algorithm.

    Returns:
        Learning algorithms first, then baselines, each group ordered by
        label (then key).
    """
    return sorted_infos(AGENT_ALGORITHMS.classes())


def info_for_class(key: str, cls: type[BaseAgentAlgorithm]) -> AlgorithmInfo:
    """Return the :class:`AlgorithmInfo` of a registered class.

    Args:
        key: Registry key.
        cls: The registered class.
    """
    return AlgorithmInfo(
        key=key,
        label=str(getattr(cls, "label", "") or key),
        description=str(getattr(cls, "description", "") or ""),
        is_learning=bool(cls.is_learning),
        is_on_policy=bool(getattr(cls, "is_on_policy", False)),
    )


def sorted_infos(
    classes: Mapping[str, type[BaseAgentAlgorithm]],
) -> tuple[AlgorithmInfo, ...]:
    """Return the infos of ``classes``, learning algorithms first, then by label.

    Args:
        classes: Registered classes by key.
    """
    infos = [info_for_class(key, cls) for key, cls in classes.items()]
    infos.sort(key=lambda info: (not info.is_learning, info.label, info.key))
    return tuple(infos)


def get_algorithm_class(key: str) -> type[BaseAgentAlgorithm]:
    """Return the algorithm class registered under ``key``.

    Args:
        key: Algorithm key.

    Returns:
        The class.

    Raises:
        KeyError: If ``key`` is unknown. The message lists the known keys.
    """
    return AGENT_ALGORITHMS.get(key)


def get_config_class(key: str) -> type[AlgorithmConfig]:
    """Return the config class of the algorithm ``key``.

    Raises:
        KeyError: If ``key`` is unknown.
    """
    return get_algorithm_class(key).config_class


def default_config(key: str) -> AlgorithmConfig:
    """Return the default config instance of the algorithm ``key``.

    Raises:
        KeyError: If ``key`` is unknown.
    """
    return get_config_class(key)()


def is_known(key: object) -> bool:
    """Return True if ``key`` is a registered algorithm key."""
    return AGENT_ALGORITHMS.is_known(key)


def is_baseline(key: object) -> bool:
    """Return True if ``key`` is a registered non-learning algorithm.

    Args:
        key: Algorithm key. Unknown keys are not baselines.
    """
    if not is_known(key):
        return False
    return not get_algorithm_class(str(key)).is_learning


def label_for(key: object) -> str:
    """Return the display label of ``key``, the key itself when unknown.

    Args:
        key: Algorithm key.
    """
    if not is_known(key):
        return "" if key is None else str(key)
    cls = get_algorithm_class(str(key))
    return str(getattr(cls, "label", "") or key)


def build_policy(
    key: str,
    spec: PolicySpec,
    params: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> BaseAgentAlgorithm:
    """Build a policy from its key, problem spec and parameters.

    Args:
        key: Algorithm key.
        spec: Problem description of the agent.
        params: Hyperparameter overrides, None for the defaults (A2).
        rng: Random generator owned by the policy.

    Returns:
        ``cls(spec, cls.config_class.from_dict(params), rng)``.

    Raises:
        KeyError: If ``key`` is unknown.
        ValueError: If ``params`` contains unknown keys or invalid values.
    """
    cls = get_algorithm_class(key)
    config = cls.config_class.from_dict(params)
    return cls(spec, config, rng)
