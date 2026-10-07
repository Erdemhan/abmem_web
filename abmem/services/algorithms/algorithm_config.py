"""Frozen hyperparameter configurations of all agent policies.

One config instance exists per agent per run. These dataclasses replace the
module-level hyperparameter globals of the legacy algorithm code (B10, PLAN 1.8).
Defaults follow the published defaults of the reference papers:

* DDPG: Lillicrap et al. (2016), "Continuous control with deep reinforcement learning".
* TD3: Fujimoto et al. (2018), "Addressing function approximation error in actor-critic methods".
* SAC: Haarnoja et al. (2018), "Soft actor-critic algorithms and applications".
* PPO: Schulman et al. (2017), "Proximal policy optimization algorithms", with GAE
  from Schulman et al. (2016), "High-dimensional continuous control using GAE".
* RDPG: Heess et al. (2015), "Memory-based control with recurrent neural networks".

Project-specific defaults resolved by the user (ARCHITECTURE.md section 11):
``gradient_steps = 1`` and ``learning_starts = 64`` (OQ-3), PPO
``rollout_length = 64`` (OQ-4), fixed-markup ``markup = 0.10`` on
(fuelCost + staticCost) (OQ-6).
"""

from __future__ import annotations

import dataclasses
import math
import numbers
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Self

from abmem.models.enums import AlgorithmType

SearchScale = Literal["linear", "log", "int"]
SearchBounds = dict[str, tuple[float, float, SearchScale]]


def _is_optional_float(hint: Any) -> bool:
    """Return True if ``hint`` is ``float | None``."""
    return typing.get_origin(hint) in (typing.Union, types.UnionType) and set(
        typing.get_args(hint)
    ) == {float, type(None)}


def _coerce_int(name: str, value: Any) -> int:
    """Coerce an integer field value, accepting integral floats from an optimizer.

    Args:
        name: Field name, used in error messages.
        value: Raw value.

    Returns:
        The value as ``int``.

    Raises:
        ValueError: If the value is a bool, non-numeric, non-finite or a
            non-integral float.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        return int(value)
    as_float = float(value)
    if not math.isfinite(as_float) or not as_float.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(as_float)


def _coerce_float(name: str, value: Any) -> float:
    """Coerce a float field value and require it to be finite.

    Args:
        name: Field name, used in error messages.
        value: Raw value.

    Returns:
        The value as a finite ``float``.

    Raises:
        ValueError: If the value is a bool, non-numeric, NaN or infinite.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return as_float


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` if ``condition`` is False."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class AlgorithmConfig:
    """Hyperparameters shared by all policies.

    Attributes:
        gamma: Discount factor, 0 < gamma < 1.
        tau: Polyak coefficient of the soft target update, 0 < tau <= 1.
        lr_actor: Adam learning rate of the actor, > 0.
        lr_critic: Adam learning rate of the critic, > 0.
        batch_size: Minibatch size of off-policy updates, >= 1.
        buffer_size: Replay buffer capacity, >= batch_size.
        learning_starts: Stored transitions required before the first gradient
            step (B7, PLAN 1.5), >= batch_size.
        gradient_steps: Gradient steps per period (OQ-3), >= 1.
        hidden_sizes: Widths of the hidden layers of the MLPs, each >= 1.
        exploration_sigma: Std of the Gaussian exploration noise in normalised
            action units, >= 0.
        exploration_sigma_decay: Multiplicative decay applied once per period,
            0 < decay <= 1.
        exploration_sigma_min: Lower bound of the decayed sigma, >= 0.
        max_grad_norm: Gradient clipping norm, ``None`` disables clipping, > 0.
    """

    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    learning_starts: int = 64
    gradient_steps: int = 1
    hidden_sizes: tuple[int, ...] = (256, 256)
    exploration_sigma: float = 0.1
    exploration_sigma_decay: float = 1.0
    exploration_sigma_min: float = 0.0
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        """Coerce field types and validate value ranges.

        Raises:
            ValueError: If a field has a wrong type, a non-finite value or a
                value outside its valid range.
        """
        self._coerce_fields()
        self._validate()

    def _coerce_fields(self) -> None:
        """Coerce every field to its annotated type (frozen-safe)."""
        hints = typing.get_type_hints(type(self))
        for f in dataclasses.fields(self):
            hint = hints[f.name]
            value = getattr(self, f.name)
            if hint is int:
                coerced: Any = _coerce_int(f.name, value)
            elif hint is float:
                coerced = _coerce_float(f.name, value)
            elif _is_optional_float(hint):
                coerced = None if value is None else _coerce_float(f.name, value)
            elif typing.get_origin(hint) is tuple:
                if isinstance(value, (str, bytes)) or not isinstance(
                    value, (tuple, list)
                ):
                    raise ValueError(
                        f"{f.name} must be a sequence of integers, got {value!r}"
                    )
                coerced = tuple(_coerce_int(f.name, v) for v in value)
            else:  # pragma: no cover - guards future field types
                raise TypeError(f"unsupported field type for {f.name}: {hint!r}")
            object.__setattr__(self, f.name, coerced)

    def _validate(self) -> None:
        """Validate value ranges. Subclasses extend this and call super()."""
        _require(0.0 < self.gamma < 1.0, f"gamma must be in (0, 1), got {self.gamma}")
        _require(0.0 < self.tau <= 1.0, f"tau must be in (0, 1], got {self.tau}")
        _require(self.lr_actor > 0.0, f"lr_actor must be > 0, got {self.lr_actor}")
        _require(self.lr_critic > 0.0, f"lr_critic must be > 0, got {self.lr_critic}")
        _require(
            self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}"
        )
        _require(
            self.buffer_size >= self.batch_size,
            f"buffer_size ({self.buffer_size}) must be >= batch_size ({self.batch_size})",
        )
        _require(
            self.learning_starts >= self.batch_size,
            f"learning_starts ({self.learning_starts}) must be >= batch_size "
            f"({self.batch_size})",
        )
        _require(
            self.gradient_steps >= 1,
            f"gradient_steps must be >= 1, got {self.gradient_steps}",
        )
        _require(
            len(self.hidden_sizes) >= 1 and all(h >= 1 for h in self.hidden_sizes),
            f"hidden_sizes must be a non-empty sequence of integers >= 1, "
            f"got {self.hidden_sizes}",
        )
        _require(
            self.exploration_sigma >= 0.0,
            f"exploration_sigma must be >= 0, got {self.exploration_sigma}",
        )
        _require(
            0.0 < self.exploration_sigma_decay <= 1.0,
            f"exploration_sigma_decay must be in (0, 1], got "
            f"{self.exploration_sigma_decay}",
        )
        _require(
            self.exploration_sigma_min >= 0.0,
            f"exploration_sigma_min must be >= 0, got {self.exploration_sigma_min}",
        )
        _require(
            self.max_grad_norm is None or self.max_grad_norm > 0.0,
            f"max_grad_norm must be None or > 0, got {self.max_grad_norm}",
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the names of all config fields in declaration order."""
        return tuple(f.name for f in dataclasses.fields(cls))

    @classmethod
    def from_dict(cls, params: Mapping[str, Any] | None) -> Self:
        """Build a config from a parameter mapping.

        Missing keys take the dataclass defaults. ``None`` yields the default
        config (A2, PLAN 1.9).

        Args:
            params: Mapping of field name to value, or ``None``.

        Returns:
            A validated config instance.

        Raises:
            TypeError: If ``params`` is neither a mapping nor ``None``.
            ValueError: If a key is not a field of this class (for example a
                legacy uppercase key such as ``'NOISE_STD'``) or a value is
                invalid.
        """
        if params is None:
            return cls()
        if not isinstance(params, Mapping):
            raise TypeError(
                f"params must be a mapping or None, got {type(params).__name__}"
            )
        accepted = cls.field_names()
        unknown = sorted(str(k) for k in params if k not in accepted)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {unknown} for {cls.__name__}; "
                f"accepted keys: {list(accepted)}"
            )
        return cls(**dict(params))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict that round-trips through from_dict.

        Returns:
            Mapping of field name to value. Tuples are returned as lists.
        """
        result: dict[str, Any] = {}
        for name in self.field_names():
            value = getattr(self, name)
            result[name] = list(value) if isinstance(value, tuple) else value
        return result

    def replace(self, **changes: Any) -> Self:
        """Return a copy with the given fields changed (validated again).

        Args:
            **changes: Field values to change.

        Returns:
            A new validated config instance.

        Raises:
            ValueError: If a key is not a field or a value is invalid.
        """
        accepted = self.field_names()
        unknown = sorted(k for k in changes if k not in accepted)
        if unknown:
            raise ValueError(
                f"unknown parameter(s) {unknown} for {type(self).__name__}; "
                f"accepted keys: {list(accepted)}"
            )
        return dataclasses.replace(self, **changes)

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return the default HPO search space of this config class.

        Every bound lies inside the validation range of its field, and every
        combination of values inside the bounds validates together with the
        defaults of the remaining fields.

        Returns:
            Mapping of field name to ``(low, high, scale)``.
        """
        return {
            "gamma": (0.9, 0.999, "linear"),
            "tau": (1e-3, 5e-2, "log"),
            "lr_actor": (1e-5, 1e-3, "log"),
            "lr_critic": (1e-5, 1e-3, "log"),
            "gradient_steps": (1, 10, "int"),
            "exploration_sigma": (0.01, 0.5, "linear"),
            "exploration_sigma_decay": (0.95, 1.0, "linear"),
        }


@dataclass(frozen=True)
class DDPGConfig(AlgorithmConfig):
    """DDPG hyperparameters (Lillicrap et al., 2016). No additional fields."""


@dataclass(frozen=True)
class TD3Config(AlgorithmConfig):
    """TD3 hyperparameters (Fujimoto et al., 2018).

    Attributes:
        policy_noise: Std of the target policy smoothing noise, >= 0.
        noise_clip: Clip range of the target policy smoothing noise, >= 0.
        policy_delay: Critic updates per actor update, >= 1.
    """

    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2

    def _validate(self) -> None:
        """Validate TD3 ranges in addition to the common ranges."""
        super()._validate()
        _require(
            self.policy_noise >= 0.0,
            f"policy_noise must be >= 0, got {self.policy_noise}",
        )
        _require(
            self.noise_clip >= 0.0, f"noise_clip must be >= 0, got {self.noise_clip}"
        )
        _require(
            self.policy_delay >= 1,
            f"policy_delay must be >= 1, got {self.policy_delay}",
        )

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return the common bounds plus the TD3-specific bounds."""
        bounds = super().default_search_bounds()
        bounds.update(
            {
                "policy_noise": (0.05, 0.5, "linear"),
                "noise_clip": (0.1, 1.0, "linear"),
                "policy_delay": (1, 4, "int"),
            }
        )
        return bounds


@dataclass(frozen=True)
class SACConfig(AlgorithmConfig):
    """SAC hyperparameters (Haarnoja et al., 2018).

    Attributes:
        lr_alpha: Adam learning rate of the temperature, > 0.
        initial_alpha: Initial entropy temperature, > 0.
        target_entropy: Target entropy, ``None`` means ``-action_dim``.
    """

    lr_alpha: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float | None = None

    def _validate(self) -> None:
        """Validate SAC ranges in addition to the common ranges."""
        super()._validate()
        _require(self.lr_alpha > 0.0, f"lr_alpha must be > 0, got {self.lr_alpha}")
        _require(
            self.initial_alpha > 0.0,
            f"initial_alpha must be > 0, got {self.initial_alpha}",
        )

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return the SAC bounds (SAC has no Gaussian exploration sigma)."""
        bounds = super().default_search_bounds()
        for name in ("exploration_sigma", "exploration_sigma_decay"):
            del bounds[name]
        bounds.update(
            {
                "lr_alpha": (1e-5, 1e-3, "log"),
                "initial_alpha": (0.01, 1.0, "log"),
            }
        )
        return bounds


@dataclass(frozen=True)
class PPOConfig(AlgorithmConfig):
    """PPO hyperparameters (Schulman et al., 2017; GAE: Schulman et al., 2016).

    Attributes:
        gae_lambda: GAE lambda, 0 <= lambda <= 1.
        clip_eps: Clipping range of the probability ratio, > 0.
        epochs: Optimisation epochs per rollout, >= 1.
        rollout_length: Transitions per rollout (OQ-4), >= minibatch_size.
        minibatch_size: Minibatch size inside an epoch, >= 1.
        entropy_coef: Entropy bonus coefficient, >= 0.
        value_coef: Value loss coefficient, >= 0.
        initial_log_std: Initial state-independent log std of the policy.
    """

    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    epochs: int = 10
    rollout_length: int = 64
    minibatch_size: int = 32
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    initial_log_std: float = 0.0

    def _validate(self) -> None:
        """Validate PPO ranges in addition to the common ranges."""
        super()._validate()
        _require(
            0.0 <= self.gae_lambda <= 1.0,
            f"gae_lambda must be in [0, 1], got {self.gae_lambda}",
        )
        _require(self.clip_eps > 0.0, f"clip_eps must be > 0, got {self.clip_eps}")
        _require(self.epochs >= 1, f"epochs must be >= 1, got {self.epochs}")
        _require(
            self.minibatch_size >= 1,
            f"minibatch_size must be >= 1, got {self.minibatch_size}",
        )
        _require(
            self.rollout_length >= self.minibatch_size,
            f"rollout_length ({self.rollout_length}) must be >= minibatch_size "
            f"({self.minibatch_size})",
        )
        _require(
            self.entropy_coef >= 0.0,
            f"entropy_coef must be >= 0, got {self.entropy_coef}",
        )
        _require(
            self.value_coef >= 0.0, f"value_coef must be >= 0, got {self.value_coef}"
        )

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return the PPO bounds (no target network, no Gaussian sigma)."""
        return {
            "gamma": (0.9, 0.999, "linear"),
            "lr_actor": (1e-5, 1e-3, "log"),
            "lr_critic": (1e-5, 1e-3, "log"),
            "gae_lambda": (0.8, 1.0, "linear"),
            "clip_eps": (0.05, 0.4, "linear"),
            "epochs": (1, 20, "int"),
            "entropy_coef": (0.0, 0.05, "linear"),
            "value_coef": (0.25, 1.0, "linear"),
        }


@dataclass(frozen=True)
class RDPGConfig(AlgorithmConfig):
    """RDPG hyperparameters (Heess et al., 2015).

    Attributes:
        seq_len: Length of the training sequences, >= 2 and <= buffer_size.
        lstm_hidden: Hidden size of the LSTM layers, >= 1.
    """

    seq_len: int = 24
    lstm_hidden: int = 64

    def _validate(self) -> None:
        """Validate RDPG ranges in addition to the common ranges."""
        super()._validate()
        _require(self.seq_len >= 2, f"seq_len must be >= 2, got {self.seq_len}")
        _require(
            self.seq_len <= self.buffer_size,
            f"seq_len ({self.seq_len}) must be <= buffer_size ({self.buffer_size})",
        )
        _require(
            self.lstm_hidden >= 1, f"lstm_hidden must be >= 1, got {self.lstm_hidden}"
        )


@dataclass(frozen=True)
class RandomConfig(AlgorithmConfig):
    """Uniform random baseline. No additional fields, nothing to search."""

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return an empty search space."""
        return {}


@dataclass(frozen=True)
class MarginalCostConfig(AlgorithmConfig):
    """Marginal-cost bidding baseline. No additional fields, nothing to search."""

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return an empty search space."""
        return {}


@dataclass(frozen=True)
class FixedMarkupConfig(AlgorithmConfig):
    """Fixed-markup baseline (OQ-6).

    The bid price of a plant is ``(fuelCost + staticCost) * (1 + markup)``
    per MWh.

    Attributes:
        markup: Relative markup on the per-MWh cost, > -1.
    """

    markup: float = 0.10

    def _validate(self) -> None:
        """Validate the markup in addition to the common ranges."""
        super()._validate()
        _require(self.markup > -1.0, f"markup must be > -1, got {self.markup}")

    @classmethod
    def default_search_bounds(cls) -> SearchBounds:
        """Return the markup bound only."""
        return {"markup": (0.0, 0.5, "linear")}


CONFIG_CLASSES: Final[Mapping[AlgorithmType, type[AlgorithmConfig]]] = (
    types.MappingProxyType(
        {
            AlgorithmType.DDPG: DDPGConfig,
            AlgorithmType.RDPG: RDPGConfig,
            AlgorithmType.TD3: TD3Config,
            AlgorithmType.SAC: SACConfig,
            AlgorithmType.PPO: PPOConfig,
            AlgorithmType.RANDOM: RandomConfig,
            AlgorithmType.MARGINAL_COST: MarginalCostConfig,
            AlgorithmType.FIXED_MARKUP: FixedMarkupConfig,
        }
    )
)
