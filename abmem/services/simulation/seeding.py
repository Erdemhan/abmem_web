"""Seeding of every random number generator used by a run (PLAN 1.10, C9).

``seed_everything`` seeds the global generators of ``random``, NumPy and
PyTorch. ``spawn_agent_rngs`` derives one independent NumPy generator per
agent from the run seed with ``numpy.random.SeedSequence.spawn``, so the
streams of different agents do not overlap and do not depend on the order in
which other code draws random numbers.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"


def _validate_seed(seed: int) -> int:
    """Return ``seed`` as int if it lies in ``[0, 2**32)``.

    Args:
        seed: Candidate seed.

    Returns:
        The seed as a Python int.

    Raises:
        ValueError: If ``seed`` is not an integer in ``[0, 2**32)``.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    value = int(seed)
    if not 0 <= value < 2**32:
        raise ValueError(f"seed must lie in [0, 2**32), got {value}")
    return value


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, the NumPy legacy global generator and PyTorch.

    Args:
        seed: Run seed in ``[0, 2**32)``.
        deterministic: If True, enable ``torch.use_deterministic_algorithms``.
            On CUDA, ``CUBLAS_WORKSPACE_CONFIG`` is set to ``':4096:8'`` first
            if it is not already set (required for deterministic cuBLAS).

    Raises:
        ValueError: If ``seed`` is out of range.
    """
    value = _validate_seed(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    if deterministic:
        if torch.cuda.is_available():
            os.environ.setdefault(
                "CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG_VALUE
            )
        torch.use_deterministic_algorithms(True)


def spawn_agent_rngs(seed: int, n_agents: int) -> list[np.random.Generator]:
    """Create one independent generator per agent.

    Args:
        seed: Run seed in ``[0, 2**32)``.
        n_agents: Number of agents, ``>= 0``. Generator ``i`` belongs to the
            ``i``-th agent in ``Agent.id`` order.

    Returns:
        ``[np.random.default_rng(child) for child in SeedSequence(seed).spawn(n_agents)]``.

    Raises:
        ValueError: If ``seed`` is out of range or ``n_agents`` is negative.
    """
    value = _validate_seed(seed)
    if isinstance(n_agents, bool) or not isinstance(n_agents, (int, np.integer)):
        raise ValueError(f"n_agents must be an integer, got {n_agents!r}")
    if n_agents < 0:
        raise ValueError(f"n_agents must be >= 0, got {n_agents}")
    children = np.random.SeedSequence(value).spawn(int(n_agents))
    return [np.random.default_rng(child) for child in children]
