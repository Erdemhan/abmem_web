"""Shared actor and critic networks of the deterministic policy-gradient family.

Forward passes contain no exploration noise, no clamping and no per-sample
normalisation (C3, C4). Inputs are already normalised by the observation
builder. The final layers of actors and critics are initialised from
U(-3e-3, 3e-3) as in Lillicrap et al. (2016), section 7.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

FINAL_LAYER_INIT: float = 3e-3


def _init_final_layer(layer: nn.Linear, bound: float = FINAL_LAYER_INIT) -> None:
    """Initialise weights and bias of ``layer`` from U(-bound, bound) in place."""
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: Sequence[int],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a fully connected network ``Linear -> act -> ... -> Linear``.

    No activation follows the output layer.

    Args:
        in_dim: Input width, >= 1.
        out_dim: Output width, >= 1.
        hidden: Widths of the hidden layers, each >= 1 (may be empty).
        activation: Activation module class used between layers.

    Returns:
        The network as ``nn.Sequential``. Its last module is the output
        ``nn.Linear``.

    Raises:
        ValueError: If a width is < 1.
    """
    widths = [in_dim, *hidden, out_dim]
    if any(int(w) < 1 for w in widths):
        raise ValueError(f"all layer widths must be >= 1, got {widths}")
    layers: list[nn.Module] = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(int(widths[i]), int(widths[i + 1])))
        if i < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Deterministic policy mu(s) with tanh output in ``[-1, 1]``."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]
    ) -> None:
        """Build the actor.

        Args:
            obs_dim: Observation length.
            action_dim: Action length.
            hidden_sizes: Hidden layer widths.
        """
        super().__init__()
        self.net = mlp(obs_dim, action_dim, hidden_sizes)
        _init_final_layer(self.net[-1])

    def forward(self, obs: Tensor) -> Tensor:
        """Return ``tanh(net(obs))``.

        Args:
            obs: Shape ``(B, obs_dim)``.

        Returns:
            Actions of shape ``(B, action_dim)`` in ``[-1, 1]``.
        """
        return torch.tanh(self.net(obs))


class QCritic(nn.Module):
    """Action-value function Q(s, a) on concatenated observation and action."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]
    ) -> None:
        """Build the critic.

        Args:
            obs_dim: Observation length.
            action_dim: Action length.
            hidden_sizes: Hidden layer widths.
        """
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_sizes)
        _init_final_layer(self.net[-1])

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        """Return Q(obs, action).

        Args:
            obs: Shape ``(B, obs_dim)``.
            action: Normalised actions, shape ``(B, action_dim)``.

        Returns:
            Q-values of shape ``(B, 1)``.
        """
        return self.net(torch.cat([obs, action], dim=-1))


class RecurrentDeterministicActor(nn.Module):
    """Recurrent deterministic policy mu(h_t) for RDPG (Heess et al., 2015)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lstm_hidden: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """Build an LSTM over observations followed by an MLP head with tanh.

        Args:
            obs_dim: Observation length.
            action_dim: Action length.
            lstm_hidden: LSTM hidden size.
            hidden_sizes: Hidden widths of the MLP head.
        """
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, lstm_hidden, batch_first=True)
        self.head = mlp(lstm_hidden, action_dim, hidden_sizes)
        _init_final_layer(self.head[-1])

    def forward(
        self, obs_seq: Tensor, hidden: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Return per-step actions and the final LSTM state.

        Args:
            obs_seq: Shape ``(B, L, obs_dim)``.
            hidden: Initial ``(h, c)``, each ``(1, B, lstm_hidden)``. None means
                a zero initial state.

        Returns:
            Actions of shape ``(B, L, action_dim)`` in ``[-1, 1]`` and the
            final ``(h, c)``.
        """
        features, state = self.lstm(obs_seq, hidden)
        return torch.tanh(self.head(features)), state


class RecurrentQCritic(nn.Module):
    """Recurrent action-value function for RDPG consuming the per-step actions (B3).

    The LSTM input at step t is the concatenation of o_t and a_t, so the output
    at step t depends on o_1..o_t and a_1..a_t only.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lstm_hidden: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        """Build an LSTM over (observation, action) pairs followed by an MLP head.

        Args:
            obs_dim: Observation length.
            action_dim: Action length.
            lstm_hidden: LSTM hidden size.
            hidden_sizes: Hidden widths of the MLP head.
        """
        super().__init__()
        self.lstm = nn.LSTM(obs_dim + action_dim, lstm_hidden, batch_first=True)
        self.head = mlp(lstm_hidden, 1, hidden_sizes)
        _init_final_layer(self.head[-1])

    def forward(
        self,
        obs_seq: Tensor,
        action_seq: Tensor,
        hidden: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Return per-step Q-values.

        Args:
            obs_seq: Shape ``(B, L, obs_dim)``.
            action_seq: Shape ``(B, L, action_dim)``.
            hidden: Initial ``(h, c)``, None means a zero initial state.

        Returns:
            Q-values of shape ``(B, L, 1)``.
        """
        features, _ = self.lstm(torch.cat([obs_seq, action_seq], dim=-1), hidden)
        return self.head(features)


def _check_same_structure(target: nn.Module, source: nn.Module) -> None:
    """Raise ValueError if the parameter names or shapes of the modules differ."""
    t_shapes = [(n, p.shape) for n, p in target.named_parameters()]
    s_shapes = [(n, p.shape) for n, p in source.named_parameters()]
    if t_shapes != s_shapes:
        raise ValueError("target and source modules have different parameters")


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak update ``theta_target <- tau * theta_source + (1 - tau) * theta_target``.

    Args:
        target: Target network, updated in place.
        source: Online network.
        tau: Interpolation coefficient, 0 < tau <= 1.

    Raises:
        ValueError: If ``tau`` is out of range or the networks differ in structure.
    """
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")
    _check_same_structure(target, source)
    with torch.no_grad():
        for t_param, s_param in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            t_param.lerp_(s_param, tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy all parameters and buffers of ``source`` into ``target``.

    Args:
        target: Target network, updated in place.
        source: Online network.

    Raises:
        ValueError: If the networks differ in structure.
    """
    _check_same_structure(target, source)
    target.load_state_dict(source.state_dict())
