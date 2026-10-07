"""Shared experience storage: off-policy ring buffer and on-policy rollout buffer.

This module is the single replay buffer implementation of the project (A10).
All arrays are float32. Observations and actions are stored in normalised
units and rewards are stored without rounding (C1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .base import Transition


@dataclass(frozen=True)
class Batch:
    """Minibatch of independent transitions as float32 tensors.

    Attributes:
        obs: Shape ``(B, obs_dim)``.
        actions: Shape ``(B, action_dim)``.
        rewards: Shape ``(B, 1)``.
        next_obs: Shape ``(B, obs_dim)``.
        terminated: Shape ``(B, 1)``, 1.0 for terminal steps.
        intervened: Shape ``(B, 1)``, 1.0 if the fail-stack rule intervened.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    intervened: torch.Tensor


@dataclass(frozen=True)
class SequenceBatch:
    """Minibatch of windows of consecutive transitions as float32 tensors.

    Attributes:
        obs: Shape ``(B, L, obs_dim)``.
        actions: Shape ``(B, L, action_dim)``.
        rewards: Shape ``(B, L, 1)``.
        next_obs: Shape ``(B, L, obs_dim)``.
        terminated: Shape ``(B, L, 1)``.
        intervened: Shape ``(B, L, 1)``.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    terminated: torch.Tensor
    intervened: torch.Tensor


@dataclass(frozen=True)
class RolloutBatch:
    """Content of a rollout buffer as float32 tensors (length T).

    Attributes:
        obs: Shape ``(T, obs_dim)``.
        actions: Shape ``(T, action_dim)``, applied normalised actions.
        pre_squash: Shape ``(T, action_dim)``, pre-tanh Gaussian samples.
        log_probs: Shape ``(T,)``, behaviour log-probabilities.
        values: Shape ``(T,)``, behaviour value estimates.
        rewards: Shape ``(T,)``.
        terminated: Shape ``(T,)``.
        intervened_mask: Shape ``(T,)``, 1.0 where the fail-stack rule intervened.
    """

    obs: torch.Tensor
    actions: torch.Tensor
    pre_squash: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    intervened_mask: torch.Tensor


def _check_dim(name: str, value: int) -> int:
    """Return ``value`` as int if it is an integer >= 1, else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _check_shapes(t: Transition, obs_dim: int, action_dim: int) -> None:
    """Validate the shapes and the action range of a transition.

    Raises:
        TypeError: If ``t`` is not a Transition.
        ValueError: If a shape does not match or the action leaves ``[-1, 1]``.
    """
    if not isinstance(t, Transition):
        raise TypeError(f"expected Transition, got {type(t).__name__}")
    if t.obs.shape != (obs_dim,):
        raise ValueError(f"obs must have shape ({obs_dim},), got {t.obs.shape}")
    if t.action.shape != (action_dim,):
        raise ValueError(
            f"action must have shape ({action_dim},), got {t.action.shape}"
        )
    if np.any(np.abs(t.action) > 1.0):
        raise ValueError(f"action must be normalised to [-1, 1], got {t.action}")


class ReplayBuffer:
    """Fixed-capacity ring buffer of transitions for off-policy algorithms.

    When full, the oldest transition is overwritten. Sampling uses the
    generator passed at construction, so it is reproducible per seed.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, capacity: int, rng: np.random.Generator
    ) -> None:
        """Allocate the storage arrays.

        Args:
            obs_dim: Observation length, >= 1.
            action_dim: Action length, >= 1.
            capacity: Maximum number of stored transitions, >= 1.
            rng: Generator used for sampling.

        Raises:
            ValueError: If a dimension or the capacity is < 1.
            TypeError: If ``rng`` is not a numpy Generator.
        """
        self.obs_dim = _check_dim("obs_dim", obs_dim)
        self.action_dim = _check_dim("action_dim", action_dim)
        self.capacity = _check_dim("capacity", capacity)
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
        self._rng = rng
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=np.float32)
        self._intervened = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def add(self, t: Transition) -> None:
        """Store a transition, overwriting the oldest one when full.

        Args:
            t: Transition with ``obs``/``next_obs`` of shape ``(obs_dim,)`` and a
                normalised ``action`` of shape ``(action_dim,)``.

        Raises:
            TypeError: If ``t`` is not a Transition.
            ValueError: If a shape does not match or the action leaves ``[-1, 1]``.
        """
        _check_shapes(t, self.obs_dim, self.action_dim)
        i = self._pos
        self._obs[i] = t.obs
        self._actions[i] = t.action
        self._rewards[i] = t.reward
        self._next_obs[i] = t.next_obs
        self._terminated[i] = float(t.terminated)
        self._intervened[i] = float(t.intervened)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._size

    def _chronological(self) -> np.ndarray:
        """Return physical indices of the stored transitions, oldest first."""
        start = self._pos if self._size == self.capacity else 0
        return (start + np.arange(self._size)) % self.capacity

    @property
    def rewards(self) -> np.ndarray:
        """Stored rewards in insertion order (oldest first), as a float32 copy."""
        return self._rewards[self._chronological()].copy()

    def sample(self, batch_size: int, device: torch.device) -> Batch:
        """Sample independent transitions uniformly without replacement.

        Args:
            batch_size: Number of transitions, 1 <= batch_size <= len(self).
            device: Target device of the tensors.

        Returns:
            A :class:`Batch` with tensors of leading dimension ``batch_size``.

        Raises:
            ValueError: If ``batch_size`` < 1 or more than the stored count.
        """
        batch_size = _check_dim("batch_size", batch_size)
        if self._size < batch_size:
            raise ValueError(
                f"cannot sample {batch_size} transitions from a buffer holding "
                f"{self._size}; check ready_to_learn first"
            )
        idx = self._rng.choice(self._size, size=batch_size, replace=False)

        def to(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        return Batch(
            obs=to(self._obs[idx]),
            actions=to(self._actions[idx]),
            rewards=to(self._rewards[idx]).unsqueeze(-1),
            next_obs=to(self._next_obs[idx]),
            terminated=to(self._terminated[idx]).unsqueeze(-1),
            intervened=to(self._intervened[idx]).unsqueeze(-1),
        )

    def sample_sequences(
        self, batch_size: int, seq_len: int, device: torch.device
    ) -> SequenceBatch:
        """Sample windows of ``seq_len`` consecutive stored transitions.

        Windows are taken in insertion order and never span the ring-buffer
        write position, so each window is a contiguous piece of the trajectory.
        Window start points are drawn without replacement when at least
        ``batch_size`` distinct windows exist, otherwise with replacement.

        Args:
            batch_size: Number of windows, >= 1.
            seq_len: Window length, >= 1 and <= len(self).
            device: Target device of the tensors.

        Returns:
            A :class:`SequenceBatch` with tensors of shape ``(B, L, dim)``.

        Raises:
            ValueError: If ``batch_size`` or ``seq_len`` < 1, or fewer than
                ``seq_len`` transitions are stored.
        """
        batch_size = _check_dim("batch_size", batch_size)
        seq_len = _check_dim("seq_len", seq_len)
        if self._size < seq_len:
            raise ValueError(
                f"cannot sample sequences of length {seq_len} from a buffer "
                f"holding {self._size}"
            )
        n_windows = self._size - seq_len + 1
        starts = self._rng.choice(
            n_windows, size=batch_size, replace=n_windows < batch_size
        )
        logical = starts[:, None] + np.arange(seq_len)[None, :]
        idx = self._chronological()[logical]

        def to(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=torch.float32, device=device)

        return SequenceBatch(
            obs=to(self._obs[idx]),
            actions=to(self._actions[idx]),
            rewards=to(self._rewards[idx]).unsqueeze(-1),
            next_obs=to(self._next_obs[idx]),
            terminated=to(self._terminated[idx]).unsqueeze(-1),
            intervened=to(self._intervened[idx]).unsqueeze(-1),
        )


class RolloutBuffer:
    """Fixed-length storage of one on-policy rollout (PPO)."""

    def __init__(self, obs_dim: int, action_dim: int, length: int) -> None:
        """Allocate the storage arrays.

        Args:
            obs_dim: Observation length, >= 1.
            action_dim: Action length, >= 1.
            length: Number of transitions per rollout, >= 1.

        Raises:
            ValueError: If a dimension or the length is < 1.
        """
        self.obs_dim = _check_dim("obs_dim", obs_dim)
        self.action_dim = _check_dim("action_dim", action_dim)
        self.length = _check_dim("length", length)
        self._obs = np.zeros((self.length, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.length, self.action_dim), dtype=np.float32)
        self._pre_squash = np.zeros((self.length, self.action_dim), dtype=np.float32)
        self._log_probs = np.zeros(self.length, dtype=np.float32)
        self._values = np.zeros(self.length, dtype=np.float32)
        self._rewards = np.zeros(self.length, dtype=np.float32)
        self._next_obs = np.zeros((self.length, self.obs_dim), dtype=np.float32)
        self._terminated = np.zeros(self.length, dtype=np.float32)
        self._intervened = np.zeros(self.length, dtype=np.float32)
        self._size = 0

    def add(self, t: Transition) -> None:
        """Append a transition with its behaviour statistics.

        Args:
            t: Transition with ``behaviour_log_prob``, ``behaviour_value`` and
                ``pre_squash`` set.

        Raises:
            TypeError: If ``t`` is not a Transition.
            ValueError: If the buffer is full, a behaviour statistic is
                missing, or a shape does not match.
        """
        _check_shapes(t, self.obs_dim, self.action_dim)
        if self._size >= self.length:
            raise ValueError(
                f"rollout buffer is full ({self.length}); update and clear it first"
            )
        if (
            t.behaviour_log_prob is None
            or t.behaviour_value is None
            or t.pre_squash is None
        ):
            raise ValueError(
                "RolloutBuffer requires behaviour_log_prob, behaviour_value and pre_squash"
            )
        if t.pre_squash.shape != (self.action_dim,):
            raise ValueError(
                f"pre_squash must have shape ({self.action_dim},), got {t.pre_squash.shape}"
            )
        i = self._size
        self._obs[i] = t.obs
        self._actions[i] = t.action
        self._pre_squash[i] = t.pre_squash
        self._log_probs[i] = t.behaviour_log_prob
        self._values[i] = t.behaviour_value
        self._rewards[i] = t.reward
        self._next_obs[i] = t.next_obs
        self._terminated[i] = float(t.terminated)
        self._intervened[i] = float(t.intervened)
        self._size += 1

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._size

    def full(self) -> bool:
        """Return True if ``length`` transitions are stored."""
        return self._size == self.length

    def clear(self) -> None:
        """Discard all stored transitions."""
        self._size = 0

    @property
    def rewards(self) -> np.ndarray:
        """Stored rewards in insertion order, as a float32 copy."""
        return self._rewards[: self._size].copy()

    @property
    def intervened_mask(self) -> np.ndarray:
        """Boolean mask of the stored steps where the fail-stack rule intervened."""
        return self._intervened[: self._size] > 0.5

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute generalised advantage estimates (Schulman et al., 2016, eq. 16).

        ``delta_t = r_t + gamma * (1 - terminated_t) * V(s_{t+1}) - V(s_t)`` and
        ``A_t = sum_l (gamma * lambda)^l * delta_{t+l}``, computed backwards
        with the recursion ``A_t = delta_t + gamma * lambda * (1 - terminated_t)
        * A_{t+1}``. ``V(s_{t+1})`` is the stored value of step ``t + 1`` and
        ``last_value`` for the last stored step.

        Args:
            last_value: V(s_T) of the observation following the last step.
            gamma: Discount factor, 0 < gamma < 1.
            gae_lambda: GAE lambda, 0 <= lambda <= 1.

        Returns:
            ``(advantages, returns)``, float32 arrays of shape ``(len(self),)``
            with ``returns = advantages + values``.

        Raises:
            ValueError: If the buffer is empty or an argument is out of range
                or not finite.
        """
        if self._size == 0:
            raise ValueError("cannot compute GAE on an empty rollout buffer")
        if not math.isfinite(last_value):
            raise ValueError(f"last_value must be finite, got {last_value}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
        n = self._size
        values = self._values[:n].astype(np.float64)
        rewards = self._rewards[:n].astype(np.float64)
        not_done = 1.0 - self._terminated[:n].astype(np.float64)
        next_values = np.append(values[1:], float(last_value))
        deltas = rewards + gamma * not_done * next_values - values
        advantages = np.zeros(n, dtype=np.float64)
        running = 0.0
        for t in range(n - 1, -1, -1):
            running = deltas[t] + gamma * gae_lambda * not_done[t] * running
            advantages[t] = running
        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def to_tensors(self, device: torch.device) -> RolloutBatch:
        """Return the stored rollout as float32 tensors.

        Args:
            device: Target device.

        Returns:
            A :class:`RolloutBatch` of length ``len(self)``.
        """
        n = self._size

        def to(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array[:n], dtype=torch.float32, device=device)

        return RolloutBatch(
            obs=to(self._obs),
            actions=to(self._actions),
            pre_squash=to(self._pre_squash),
            log_probs=to(self._log_probs),
            values=to(self._values),
            rewards=to(self._rewards),
            terminated=to(self._terminated),
            intervened_mask=to(self._intervened),
        )
