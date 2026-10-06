"""Mapping from AlgorithmType to policy classes and config classes.

All eight class paths are declared here as ``'module:Class'`` strings and are
imported lazily with :mod:`importlib`, so the algorithm modules can be
rewritten independently without editing this file.
"""

from __future__ import annotations

import importlib
import types
from collections.abc import Mapping
from typing import Any, Final

import numpy as np

from abmem.models.enums import AlgorithmType

from .algorithm_config import CONFIG_CLASSES, AlgorithmConfig
from .base import BaseAgentAlgorithm, PolicySpec

ALGORITHM_CLASS_PATHS: Final[Mapping[AlgorithmType, str]] = types.MappingProxyType(
    {
        AlgorithmType.DDPG: "abmem.services.algorithms.DDPG:DDPGAlgorithm",
        AlgorithmType.TD3: "abmem.services.algorithms.TD3:TD3Algorithm",
        AlgorithmType.SAC: "abmem.services.algorithms.SAC:SACAlgorithm",
        AlgorithmType.PPO: "abmem.services.algorithms.ppo:PPOAlgorithm",
        AlgorithmType.RDPG: "abmem.services.algorithms.RDPG:RDPGAlgorithm",
        AlgorithmType.RANDOM: "abmem.services.algorithms.baselines:RandomPolicy",
        AlgorithmType.MARGINAL_COST: "abmem.services.algorithms.baselines:MarginalCostPolicy",
        AlgorithmType.FIXED_MARKUP: "abmem.services.algorithms.baselines:FixedMarkupPolicy",
    }
)

BASELINE_TYPES: Final[frozenset[AlgorithmType]] = frozenset(
    {AlgorithmType.RANDOM, AlgorithmType.MARGINAL_COST, AlgorithmType.FIXED_MARKUP}
)


def _lookup(table: Mapping[AlgorithmType, Any], algorithm_type: Any) -> Any:
    """Return ``table[algorithm_type]`` or raise a KeyError naming the value.

    Raises:
        KeyError: If ``algorithm_type`` is not a known AlgorithmType.
    """
    try:
        return table[algorithm_type]
    except (KeyError, TypeError):
        known = [t.name for t in AlgorithmType]
        raise KeyError(
            f"unknown algorithm type {algorithm_type!r}; known types: {known}"
        ) from None


def get_algorithm_class(algorithm_type: AlgorithmType) -> type[BaseAgentAlgorithm]:
    """Import and return the policy class of ``algorithm_type``.

    Args:
        algorithm_type: The algorithm type.

    Returns:
        The policy class.

    Raises:
        KeyError: If ``algorithm_type`` is unknown.
        ImportError: If the module or the class cannot be imported.
        TypeError: If the resolved object is not a BaseAgentAlgorithm subclass
            declaring the same ``algorithm_type``.
    """
    path: str = _lookup(ALGORITHM_CLASS_PATHS, algorithm_type)
    module_name, class_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ImportError(f"{module_name} does not define {class_name}") from None
    if not (isinstance(cls, type) and issubclass(cls, BaseAgentAlgorithm)):
        raise TypeError(f"{path} is not a BaseAgentAlgorithm subclass")
    if getattr(cls, "algorithm_type", None) != algorithm_type:
        raise TypeError(
            f"{path} declares algorithm_type {getattr(cls, 'algorithm_type', None)!r}, "
            f"expected {algorithm_type!r}"
        )
    return cls


def config_class(algorithm_type: AlgorithmType) -> type[AlgorithmConfig]:
    """Return the config class of ``algorithm_type``.

    Raises:
        KeyError: If ``algorithm_type`` is unknown.
    """
    return _lookup(CONFIG_CLASSES, algorithm_type)


def default_config(algorithm_type: AlgorithmType) -> AlgorithmConfig:
    """Return the default config instance of ``algorithm_type``.

    Raises:
        KeyError: If ``algorithm_type`` is unknown.
    """
    return config_class(algorithm_type)()


def build_policy(
    algorithm_type: AlgorithmType,
    spec: PolicySpec,
    params: Mapping[str, Any] | None,
    rng: np.random.Generator,
) -> BaseAgentAlgorithm:
    """Build a policy from its type, problem spec and parameters.

    Args:
        algorithm_type: The algorithm type.
        spec: Problem description of the agent.
        params: Hyperparameter overrides, None for the defaults (A2).
        rng: Random generator owned by the policy.

    Returns:
        ``cls(spec, config_class(algorithm_type).from_dict(params), rng)``.

    Raises:
        KeyError: If ``algorithm_type`` is unknown.
        ValueError: If ``params`` contains unknown keys or invalid values.
    """
    config = config_class(algorithm_type).from_dict(params)
    cls = get_algorithm_class(algorithm_type)
    return cls(spec, config, rng)
