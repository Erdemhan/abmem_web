"""Networks of Soft Actor-Critic (Haarnoja et al., 2018).

The policy is a diagonal Gaussian over a pre-squash variable ``z`` followed by
``u = tanh(z)``, so every action lies in the normalised action space
``[-1, 1]`` (ADR-003). Mapping ``u`` to prices happens outside the algorithm.
The soft Q-functions are the shared :class:`~.networks.QCritic`.

References:
    Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018). Soft Actor-Critic:
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic
    Actor. ICML. Appendix C (enforcing action bounds).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .networks import mlp

LOG_STD_MIN: float = -20.0
LOG_STD_MAX: float = 2.0
_LOG_2: float = math.log(2.0)
_HALF_LOG_2PI: float = 0.5 * math.log(2.0 * math.pi)


def tanh_log_abs_det_jacobian(z: Tensor) -> Tensor:
    """Return ``log(1 - tanh(z)^2)`` elementwise in a numerically stable form.

    Uses the identity ``log(1 - tanh(z)^2) = 2 * (log 2 - z - softplus(-2 z))``,
    which stays finite for large ``|z|`` where ``tanh(z)`` rounds to +-1 in
    float32.

    Args:
        z: Pre-squash values of any shape.

    Returns:
        Tensor of the same shape as ``z``.
    """
    return 2.0 * (_LOG_2 - z - F.softplus(-2.0 * z))


def squashed_gaussian_log_prob(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    """Log-density of ``u = tanh(z)`` with ``z ~ N(mean, diag(exp(log_std)^2))``.

    Haarnoja et al. (2018), appendix C, eq. 21:
    ``log pi(u|s) = log N(z; mean, sigma) - sum_i log(1 - tanh(z_i)^2)``.

    Args:
        z: Pre-squash sample, shape ``(B, action_dim)``.
        mean: Gaussian mean, same shape as ``z``.
        log_std: Gaussian log standard deviation, same shape as ``z``.

    Returns:
        Log-probabilities of shape ``(B, 1)``.
    """
    gaussian = -0.5 * ((z - mean) * torch.exp(-log_std)) ** 2 - log_std - _HALF_LOG_2PI
    log_prob = gaussian - tanh_log_abs_det_jacobian(z)
    return log_prob.sum(dim=-1, keepdim=True)


class SquashedGaussianActor(nn.Module):
    """Stochastic policy ``u = tanh(z)``, ``z ~ N(mu(s), sigma(s)^2)``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        """Build a shared MLP trunk with a mean head and a log-std head.

        Args:
            obs_dim: Observation length.
            action_dim: Action length.
            hidden_sizes: Hidden layer widths of the trunk (at least one).
            log_std_min: Lower clamp of the log standard deviation.
            log_std_max: Upper clamp of the log standard deviation.

        Raises:
            ValueError: If ``hidden_sizes`` is empty or the clamp is not
                increasing.
        """
        super().__init__()
        if len(hidden_sizes) < 1:
            raise ValueError("hidden_sizes must contain at least one layer")
        if not log_std_min < log_std_max:
            raise ValueError(
                f"log_std_min ({log_std_min}) must be < log_std_max ({log_std_max})"
            )
        widths = [int(w) for w in hidden_sizes]
        self.trunk = nn.Sequential(mlp(obs_dim, widths[-1], widths[:-1]), nn.ReLU())
        self.mean_head = nn.Linear(widths[-1], action_dim)
        self.log_std_head = nn.Linear(widths[-1], action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the Gaussian parameters of the pre-squash variable.

        Args:
            obs: Shape ``(B, obs_dim)``.

        Returns:
            ``(mean, log_std)``, each of shape ``(B, action_dim)``, with
            ``log_std`` clamped to ``[log_std_min, log_std_max]``.
        """
        features = self.trunk(obs)
        mean = self.mean_head(features)
        log_std = torch.clamp(
            self.log_std_head(features), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    def sample(
        self, obs: Tensor, generator: torch.Generator | None = None
    ) -> tuple[Tensor, Tensor]:
        """Draw a reparameterised action and its log-probability.

        ``z = mean + sigma * eps`` with ``eps ~ N(0, I)`` (Haarnoja et al.,
        2018, eq. 11), ``u = tanh(z)``, log-probability per appendix C.

        Args:
            obs: Shape ``(B, obs_dim)``.
            generator: Optional torch generator for ``eps``.

        Returns:
            ``(u, log_prob)`` with shapes ``(B, action_dim)`` and ``(B, 1)``.
        """
        mean, log_std = self(obs)
        eps = torch.randn(
            mean.shape, generator=generator, dtype=mean.dtype, device=mean.device
        )
        z = mean + torch.exp(log_std) * eps
        return torch.tanh(z), squashed_gaussian_log_prob(z, mean, log_std)

    def deterministic(self, obs: Tensor) -> Tensor:
        """Return the deterministic action ``tanh(mean(obs))``.

        Args:
            obs: Shape ``(B, obs_dim)``.

        Returns:
            Actions of shape ``(B, action_dim)`` in ``[-1, 1]``.
        """
        mean, _ = self(obs)
        return torch.tanh(mean)
