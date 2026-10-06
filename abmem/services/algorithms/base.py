"""Policy interface shared by all learning algorithms and baselines.

All policies operate in normalised spaces (ADR-003): observations are
normalised float32 vectors and actions are float32 vectors in ``[-1, 1]``, one
entry per plant. Mapping actions to prices, the fail-stack rule and reward
computation live outside the algorithms.
"""

from __future__ import annotations

import abc
import functools
import math
import numbers
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch import nn

from abmem.models.enums import AlgorithmType

from .algorithm_config import AlgorithmConfig
from .device import get_device

if TYPE_CHECKING:
    from .replay_buffer import ReplayBuffer, RolloutBuffer


def _finite_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite float or raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return as_float


def _finite_vector(name: str, value: Any) -> np.ndarray:
    """Return ``value`` as a finite 1-D float32 array (copied) or raise ValueError."""
    array = np.array(value, dtype=np.float32, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values, got {array}")
    return array


@dataclass(frozen=True)
class PolicySpec:
    """Static description of the decision problem of one agent.

    Attributes:
        obs_dim: Length of the normalised observation vector, >= 1.
        action_dim: Number of plants (one normalised action per plant), >= 1.
        price_low: Lower bid bound of the market.
        price_high: Upper bid bound of the market, > price_low.
        plant_variable_costs: fuelCost per plant (TL/MWh), length action_dim.
        plant_static_costs: omCost + investmentCost + co2Cost per plant
            (TL/MWh), length action_dim.
    """

    obs_dim: int
    action_dim: int
    price_low: float
    price_high: float
    plant_variable_costs: tuple[float, ...]
    plant_static_costs: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate dimensions, bounds and costs.

        Raises:
            ValueError: If a dimension is < 1, the bounds are not finite and
                increasing, or a cost tuple has the wrong length or a
                non-finite value.
        """
        for name in ("obs_dim", "action_dim"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
            object.__setattr__(self, name, int(value))
        low = _finite_float("price_low", self.price_low)
        high = _finite_float("price_high", self.price_high)
        if not high > low:
            raise ValueError(
                f"price_high ({high}) must be greater than price_low ({low})"
            )
        object.__setattr__(self, "price_low", low)
        object.__setattr__(self, "price_high", high)
        for name in ("plant_variable_costs", "plant_static_costs"):
            costs = tuple(
                _finite_float(f"{name}[{i}]", c)
                for i, c in enumerate(getattr(self, name))
            )
            if len(costs) != self.action_dim:
                raise ValueError(
                    f"{name} must have length action_dim={self.action_dim}, "
                    f"got {len(costs)}"
                )
            object.__setattr__(self, name, costs)


@dataclass(frozen=True)
class ActionResult:
    """Output of :meth:`BaseAgentAlgorithm.select_action`.

    Attributes:
        u: Normalised action, float32, shape ``(action_dim,)``, values in
            ``[-1, 1]``.
        log_prob: Behaviour log-probability of ``u`` (PPO).
        value: State value estimate V(o) (PPO).
        pre_squash: Gaussian sample before the tanh squashing (PPO).
    """

    u: np.ndarray
    log_prob: float | None = None
    value: float | None = None
    pre_squash: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check dtype, shape, finiteness and range of the action.

        Raises:
            ValueError: If ``u`` is not a 1-D float32 array of finite values in
                ``[-1, 1]``, or an optional field is not finite.
        """
        if not isinstance(self.u, np.ndarray) or self.u.dtype != np.float32:
            raise ValueError(
                f"u must be a float32 numpy array, got {type(self.u).__name__} "
                f"{getattr(self.u, 'dtype', '')}"
            )
        if self.u.ndim != 1 or self.u.size < 1:
            raise ValueError(
                f"u must be a non-empty 1-D array, got shape {self.u.shape}"
            )
        if not np.all(np.isfinite(self.u)):
            raise ValueError(f"u must contain only finite values, got {self.u}")
        if np.any(np.abs(self.u) > 1.0):
            raise ValueError(f"u must lie in [-1, 1], got {self.u}")
        if self.log_prob is not None:
            object.__setattr__(
                self, "log_prob", _finite_float("log_prob", self.log_prob)
            )
        if self.value is not None:
            object.__setattr__(self, "value", _finite_float("value", self.value))
        if self.pre_squash is not None:
            pre = _finite_vector("pre_squash", self.pre_squash)
            if pre.shape != self.u.shape:
                raise ValueError(
                    f"pre_squash shape {pre.shape} differs from u shape {self.u.shape}"
                )
            object.__setattr__(self, "pre_squash", pre)


@dataclass(frozen=True)
class Transition:
    """One environment step of one agent in normalised spaces.

    Attributes:
        obs: Normalised observation o_t, float32, shape ``(obs_dim,)``.
        action: Applied normalised action u_t (after the fail-stack rule).
        reward: Reward r_t as float, not rounded and not clipped (C1).
        next_obs: Normalised observation o_{t+1}, float32, shape ``(obs_dim,)``.
        terminated: Terminal flag (always False in the single infinite
            trajectory of a simulation).
        behaviour_log_prob: Behaviour log-probability (on-policy algorithms).
        behaviour_value: Behaviour value estimate (on-policy algorithms).
        pre_squash: Pre-tanh Gaussian sample (on-policy algorithms).
        intervened: True if the fail-stack rule replaced any plant price.
    """

    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    terminated: bool = False
    behaviour_log_prob: float | None = None
    behaviour_value: float | None = None
    pre_squash: np.ndarray | None = None
    intervened: bool = False

    def __post_init__(self) -> None:
        """Convert arrays to float32 copies and reject non-finite values.

        Raises:
            ValueError: If an array is not 1-D, obs and next_obs differ in
                shape, or any value is NaN or infinite.
        """
        obs = _finite_vector("obs", self.obs)
        next_obs = _finite_vector("next_obs", self.next_obs)
        if obs.shape != next_obs.shape:
            raise ValueError(
                f"obs shape {obs.shape} differs from next_obs shape {next_obs.shape}"
            )
        object.__setattr__(self, "obs", obs)
        object.__setattr__(self, "next_obs", next_obs)
        object.__setattr__(self, "action", _finite_vector("action", self.action))
        object.__setattr__(self, "reward", _finite_float("reward", self.reward))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "intervened", bool(self.intervened))
        if self.behaviour_log_prob is not None:
            object.__setattr__(
                self,
                "behaviour_log_prob",
                _finite_float("behaviour_log_prob", self.behaviour_log_prob),
            )
        if self.behaviour_value is not None:
            object.__setattr__(
                self,
                "behaviour_value",
                _finite_float("behaviour_value", self.behaviour_value),
            )
        if self.pre_squash is not None:
            object.__setattr__(
                self, "pre_squash", _finite_vector("pre_squash", self.pre_squash)
            )


@dataclass(frozen=True)
class UpdateStats:
    """Diagnostics returned by :meth:`BaseAgentAlgorithm.update`.

    Attributes:
        critic_loss: Mean critic (or value) loss over the gradient steps.
        actor_loss: Mean actor (or policy) loss, None if no actor step was taken.
        mean_q: Mean Q-value (or value estimate) of the last batch.
        extra: Additional algorithm-specific scalars.
    """

    critic_loss: float | None
    actor_loss: float | None
    mean_q: float | None
    extra: Mapping[str, float] = field(default_factory=dict)


def _wrap_select_action(
    func: Callable[[BaseAgentAlgorithm, np.ndarray, bool], ActionResult],
) -> Callable[[BaseAgentAlgorithm, Any, bool], ActionResult]:
    """Wrap a concrete select_action with input checks, output checks and no_grad.

    Args:
        func: The subclass implementation.

    Returns:
        A wrapper that validates the observation, runs ``func`` under
        ``torch.no_grad()`` and checks the action dimension of the result.
    """

    @functools.wraps(func)
    def wrapper(self: BaseAgentAlgorithm, obs: Any, explore: bool) -> ActionResult:
        array = _finite_vector("obs", obs)
        if array.shape != (self.spec.obs_dim,):
            raise ValueError(
                f"obs must have shape ({self.spec.obs_dim},), got {array.shape}"
            )
        with torch.no_grad():
            result = func(self, array, bool(explore))
        if not isinstance(result, ActionResult):
            raise TypeError(
                f"select_action must return ActionResult, got {type(result).__name__}"
            )
        if result.u.shape != (self.spec.action_dim,):
            raise ValueError(
                f"select_action returned shape {result.u.shape}, expected "
                f"({self.spec.action_dim},)"
            )
        return result

    wrapper._abmem_checked = True  # type: ignore[attr-defined]
    return wrapper


class BaseAgentAlgorithm(abc.ABC):
    """Template base class of every policy (DRL algorithms and baselines).

    Subclasses set the class variables, implement :meth:`select_action` and,
    for learning policies, :meth:`_update`. The template guarantees:

    * ``select_action`` receives a validated float32 observation of shape
      ``(obs_dim,)``, runs under ``torch.no_grad()`` and must return exactly
      ``action_dim`` values (checked). Exploration noise is added only when
      ``explore`` is True (C3).
    * ``observe`` and ``update`` are no-ops while frozen.
    * ``update`` calls ``_update`` only when ``ready_to_learn`` holds (B7).
    * ``end_period`` decays the exploration sigma once per period.

    Attributes:
        spec: Problem description.
        config: Validated hyperparameters, instance of ``config_class``.
        rng: Generator used for exploration noise and buffer sampling.
        device: Torch device of all networks (``device.get_device()``).
        buffer: Replay or rollout buffer, None for non-learning policies.
    """

    algorithm_type: ClassVar[AlgorithmType]
    is_learning: ClassVar[bool]
    is_on_policy: ClassVar[bool]
    config_class: ClassVar[type[AlgorithmConfig]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap a concrete ``select_action`` defined on the subclass."""
        super().__init_subclass__(**kwargs)
        func = cls.__dict__.get("select_action")
        if (
            func is not None
            and callable(func)
            and not getattr(func, "__isabstractmethod__", False)
            and not getattr(func, "_abmem_checked", False)
        ):
            cls.select_action = _wrap_select_action(func)  # type: ignore[method-assign]

    def __init__(
        self, spec: PolicySpec, config: AlgorithmConfig, rng: np.random.Generator
    ) -> None:
        """Store the spec, config and generator and build the default buffer.

        Args:
            spec: Problem description.
            config: Hyperparameters, must be an instance of ``config_class``.
            rng: Random generator owned by this policy.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        if not isinstance(spec, PolicySpec):
            raise TypeError(f"spec must be a PolicySpec, got {type(spec).__name__}")
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} requires {self.config_class.__name__}, "
                f"got {type(config).__name__}"
            )
        if not isinstance(rng, np.random.Generator):
            raise TypeError(f"rng must be a numpy Generator, got {type(rng).__name__}")
        self.spec = spec
        self.config = config
        self.rng = rng
        self.device: torch.device = get_device()
        self._frozen = False
        self._exploration_sigma = float(config.exploration_sigma)
        self.buffer: ReplayBuffer | RolloutBuffer | None = self._build_buffer()

    def _build_buffer(self) -> ReplayBuffer | RolloutBuffer | None:
        """Create the default buffer. Subclasses may override.

        Returns:
            A :class:`RolloutBuffer` of ``config.rollout_length`` for on-policy
            learners, a :class:`ReplayBuffer` of ``config.buffer_size`` for
            off-policy learners, None for non-learning policies.
        """
        from .replay_buffer import ReplayBuffer, RolloutBuffer

        if not self.is_learning:
            return None
        if self.is_on_policy:
            length = getattr(self.config, "rollout_length", self.config.buffer_size)
            return RolloutBuffer(self.spec.obs_dim, self.spec.action_dim, int(length))
        return ReplayBuffer(
            self.spec.obs_dim, self.spec.action_dim, self.config.buffer_size, self.rng
        )

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return the normalised action for a normalised observation.

        Args:
            obs: Normalised observation, float32, shape ``(obs_dim,)``.
            explore: Add exploration noise (or sample stochastically) if True.
                With False the output is a deterministic function of ``obs``
                and the parameters.

        Returns:
            The action and, for on-policy algorithms, behaviour statistics.
        """

    def observe(self, transition: Transition) -> None:
        """Store a transition. No-op when frozen or without a buffer.

        Args:
            transition: The step to store.
        """
        if self._frozen or self.buffer is None:
            return
        self.buffer.add(transition)

    def update(self) -> UpdateStats | None:
        """Run ``config.gradient_steps`` gradient steps if ready to learn.

        Returns:
            Statistics if at least one gradient step was taken, else None.
        """
        if self._frozen or not self.is_learning or not self.ready_to_learn:
            return None
        return self._update()

    def _update(self) -> UpdateStats | None:
        """Algorithm-specific update, called only when ``ready_to_learn``.

        Returns:
            Statistics of the gradient steps taken, None if none was taken.

        Raises:
            NotImplementedError: If a learning subclass does not override it.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _update()")

    def end_period(self) -> None:
        """Apply the per-period exploration decay ``sigma = max(sigma_min, sigma * decay)``."""
        self._exploration_sigma = max(
            self.config.exploration_sigma_min,
            self._exploration_sigma * self.config.exploration_sigma_decay,
        )

    def set_frozen(self, frozen: bool) -> None:
        """Enable or disable learning (evaluation phase: no observe/update).

        Args:
            frozen: True to freeze the policy.
        """
        self._frozen = bool(frozen)

    def exploration_scale(self) -> float | None:
        """Return the current exploration scale.

        Returns:
            The Gaussian sigma for learning policies (DDPG, TD3, RDPG), None for
            non-learning policies. SAC and PPO override this (alpha, mean std).
        """
        return self._exploration_sigma if self.is_learning else None

    @property
    def ready_to_learn(self) -> bool:
        """True if the buffer holds at least ``config.learning_starts`` transitions."""
        return (
            self.buffer is not None and len(self.buffer) >= self.config.learning_starts
        )

    @property
    def frozen(self) -> bool:
        """True while the policy is frozen."""
        return self._frozen

    # Helpers for subclasses -------------------------------------------------

    def _add_exploration_noise(self, u: np.ndarray) -> np.ndarray:
        """Return ``clip(u + N(0, sigma^2 I), -1, 1)`` as float32.

        Args:
            u: Deterministic normalised action, shape ``(action_dim,)``.

        Returns:
            The perturbed action, clipped to ``[-1, 1]``.
        """
        noise = self.rng.normal(0.0, self._exploration_sigma, size=np.shape(u))
        return np.clip(np.asarray(u, dtype=np.float64) + noise, -1.0, 1.0).astype(
            np.float32
        )

    def _as_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Return ``array`` as a float32 tensor on ``self.device``."""
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _clip_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
        """Clip gradients to ``config.max_grad_norm`` if it is set.

        Args:
            parameters: Parameters whose gradients are clipped in place.
        """
        if self.config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(list(parameters), self.config.max_grad_norm)
