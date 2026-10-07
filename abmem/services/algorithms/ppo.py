"""Proximal Policy Optimization with a tanh-squashed Gaussian policy.

Implements the clipped surrogate objective of Schulman et al. (2017), eq. 7,
with generalised advantage estimation (GAE) of Schulman et al. (2016), eq. 16,
in the normalised action space ``[-1, 1]`` of ADR-003.

The policy samples ``z ~ N(mu(s), diag(sigma^2))`` with a state-independent
log standard deviation and applies ``u = tanh(z)``. The behaviour
log-probability ``log pi_old(u|s)`` is computed and returned at sampling time
(C6) and :meth:`PPOAlgorithm.evaluate` recomputes the log-probability of the
stored pre-squash sample with exactly the same distribution. Updates run over
fixed-length rollouts (``rollout_length``, OQ-4). Steps where the fail-stack
rule replaced the price (``Transition.intervened``, K3) are excluded from the
policy loss because the applied action was not drawn from the policy, and are
kept in the value loss and in GAE.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from abmem.models.enums import AlgorithmType

from .algorithm_config import PPOConfig
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec, Transition, UpdateStats
from .networks import mlp
from .replay_buffer import RolloutBatch, RolloutBuffer

#: Scale of the initial weights of the policy mean output layer. A small
#: output layer keeps the initial mean close to zero so that the initial policy
#: is governed by ``initial_log_std`` (Andrychowicz et al., 2021, section 3.2).
POLICY_OUTPUT_SCALE: float = 0.01

#: Added to the advantage standard deviation in the normalisation.
ADVANTAGE_EPS: float = 1e-8

_LOG_2: float = math.log(2.0)
_LOG_SQRT_2PI: float = 0.5 * math.log(2.0 * math.pi)


def tanh_gaussian_log_prob(mean: Tensor, log_std: Tensor, z: Tensor) -> Tensor:
    """Log-density of ``u = tanh(z)`` with ``z ~ N(mean, diag(exp(log_std)^2))``.

    ``log pi(u|s) = sum_i [log N(z_i; mean_i, sigma_i) - log(1 - tanh(z_i)^2)]``
    (Haarnoja et al., 2018, appendix C, eq. 21). The Jacobian term uses the
    numerically stable identity ``log(1 - tanh(z)^2) = 2 * (log 2 - z -
    softplus(-2 z))``, which stays finite for large ``|z|``.

    Args:
        mean: Gaussian mean, shape ``(..., action_dim)``.
        log_std: Log standard deviation, broadcastable to ``mean``.
        z: Pre-squash sample, same shape as ``mean``.

    Returns:
        Log-probabilities summed over the action dimension, shape ``(...)``.
    """
    log_std = log_std.expand_as(mean)
    normalised = (z - mean) * torch.exp(-log_std)
    gaussian = -0.5 * normalised.pow(2) - log_std - _LOG_SQRT_2PI
    log_det_jacobian = 2.0 * (_LOG_2 - z - F.softplus(-2.0 * z))
    return (gaussian - log_det_jacobian).sum(dim=-1)


class TanhGaussianPolicy(nn.Module):
    """Diagonal Gaussian policy with state-independent log std, squashed by tanh.

    Attributes:
        mean_net: MLP ``obs -> mu(obs)`` with tanh hidden activations.
        log_std: Learnable log standard deviation, shape ``(action_dim,)``.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        initial_log_std: float,
    ) -> None:
        """Build the mean network and the log std parameter.

        Args:
            obs_dim: Observation length, >= 1.
            action_dim: Action length, >= 1.
            hidden_sizes: Hidden layer widths of the mean network.
            initial_log_std: Initial value of every log std entry.
        """
        super().__init__()
        self.mean_net = mlp(obs_dim, action_dim, hidden_sizes, activation=nn.Tanh)
        output = self.mean_net[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight.mul_(POLICY_OUTPUT_SCALE)
            output.bias.zero_()
        self.log_std = nn.Parameter(
            torch.full((action_dim,), float(initial_log_std), dtype=torch.float32)
        )

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return the Gaussian mean and the broadcast log std.

        Args:
            obs: Observations, shape ``(batch, obs_dim)``.

        Returns:
            ``(mean, log_std)``, both of shape ``(batch, action_dim)``.
        """
        mean = self.mean_net(obs)
        return mean, self.log_std.expand_as(mean)


class PPOAlgorithm(BaseAgentAlgorithm):
    """Clipped PPO (Schulman et al., 2017) with GAE (Schulman et al., 2016).

    Attributes:
        policy: Tanh-squashed Gaussian policy.
        value_net: State value network ``V(s)``.
        policy_opt: Adam optimiser of the policy (``lr_actor``).
        value_opt: Adam optimiser of the value network (``lr_critic``).
    """

    algorithm_type = AlgorithmType.PPO
    is_learning = True
    is_on_policy = True
    config_class = PPOConfig

    config: PPOConfig
    buffer: RolloutBuffer

    def __init__(
        self, spec: PolicySpec, config: PPOConfig, rng: np.random.Generator
    ) -> None:
        """Build the networks, optimisers and the rollout buffer.

        Args:
            spec: Problem description.
            config: PPO hyperparameters.
            rng: Generator used for action sampling and minibatch shuffling.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If ``learning_starts`` exceeds ``rollout_length``. The
                rollout buffer could then never reach ``learning_starts``.
        """
        super().__init__(spec, config, rng)
        if config.learning_starts > config.rollout_length:
            raise ValueError(
                f"learning_starts ({config.learning_starts}) must be <= "
                f"rollout_length ({config.rollout_length}) for PPO"
            )
        self.policy = TanhGaussianPolicy(
            spec.obs_dim, spec.action_dim, config.hidden_sizes, config.initial_log_std
        ).to(self.device)
        self.value_net = mlp(spec.obs_dim, 1, config.hidden_sizes, nn.Tanh).to(
            self.device
        )
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=config.lr_actor)
        self.value_opt = torch.optim.Adam(
            self.value_net.parameters(), lr=config.lr_critic
        )
        self._last_next_obs: np.ndarray | None = None

    def _state_modules(self) -> dict[str, nn.Module]:
        """Return the policy and the value network by name."""
        return {"policy": self.policy, "value_net": self.value_net}

    def _state_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Return the policy and value optimisers by name."""
        return {"policy": self.policy_opt, "value": self.value_opt}

    def _load_extra_state(
        self, extra: Mapping[str, Any], *, restore_runtime: bool
    ) -> None:
        """Forget the bootstrap observation, it belongs to the unsaved rollout."""
        self._last_next_obs = None

    # Acting -----------------------------------------------------------------

    def _value(self, obs: Tensor) -> Tensor:
        """Return ``V(obs)`` with shape ``(batch,)``."""
        return self.value_net(obs).squeeze(-1)

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Sample (or take the mean of) the squashed Gaussian policy.

        With ``explore=True``, ``z = mu(s) + sigma * eps`` with
        ``eps ~ N(0, I)`` drawn from ``self.rng`` and ``u = tanh(z)``. With
        ``explore=False``, ``z = mu(s)``. In both cases the returned
        ``log_prob`` is ``log pi(tanh(z)|s)`` under the current parameters,
        computed at sampling time (C6), together with ``V(s)`` and ``z``.

        Args:
            obs: Normalised observation, shape ``(obs_dim,)``.
            explore: Sample stochastically if True.

        Returns:
            ActionResult with ``u``, ``log_prob``, ``value`` and ``pre_squash``.
        """
        obs_t = self._as_tensor(obs).unsqueeze(0)
        mean, log_std = self.policy(obs_t)
        if explore:
            eps = self._as_tensor(
                self.rng.standard_normal(self.spec.action_dim).astype(np.float32)
            ).unsqueeze(0)
            z = mean + torch.exp(log_std) * eps
        else:
            z = mean
        log_prob = tanh_gaussian_log_prob(mean, log_std, z)
        value = self._value(obs_t)
        u = torch.tanh(z).squeeze(0).cpu().numpy().astype(np.float32)
        return ActionResult(
            u=np.clip(u, -1.0, 1.0),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            pre_squash=z.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def evaluate(self, obs: Tensor, pre_squash: Tensor) -> Tensor:
        """Log-probability of stored pre-squash samples under the current policy.

        Uses the same tanh-squashed Gaussian as :meth:`select_action` (C6).

        Args:
            obs: Observations, shape ``(batch, obs_dim)``.
            pre_squash: Stored samples ``z``, shape ``(batch, action_dim)``.

        Returns:
            ``log pi(tanh(z)|s)``, shape ``(batch,)``, differentiable with
            respect to the policy parameters.
        """
        mean, log_std = self.policy(obs)
        return tanh_gaussian_log_prob(mean, log_std, pre_squash)

    def exploration_scale(self) -> float:
        """Return the mean policy standard deviation ``mean(exp(log_std))``."""
        with torch.no_grad():
            return float(torch.exp(self.policy.log_std).mean().item())

    # Learning ---------------------------------------------------------------

    def observe(self, transition: Transition) -> None:
        """Store a transition and remember its ``next_obs`` for bootstrapping.

        Args:
            transition: Step with behaviour statistics from :meth:`select_action`.
        """
        if self._frozen:
            return
        super().observe(transition)
        self._last_next_obs = transition.next_obs.copy()

    @property
    def ready_to_learn(self) -> bool:
        """True when the rollout is complete and holds ``learning_starts`` steps."""
        return self.buffer.full() and len(self.buffer) >= self.config.learning_starts

    def compute_advantages(self) -> tuple[np.ndarray, np.ndarray]:
        """GAE advantages and returns of the stored rollout (Schulman et al., 2016).

        Bootstraps with ``V(next_obs)`` of the last stored transition under the
        current value network and applies eq. 16 through
        :meth:`RolloutBuffer.compute_gae`.

        Returns:
            ``(advantages, returns)``, float32 arrays of shape ``(len(buffer),)``.

        Raises:
            ValueError: If the buffer is empty.
        """
        if len(self.buffer) == 0 or self._last_next_obs is None:
            raise ValueError("cannot compute advantages of an empty rollout")
        with torch.no_grad():
            last_value = float(
                self._value(self._as_tensor(self._last_next_obs).unsqueeze(0)).item()
            )
        return self.buffer.compute_gae(
            last_value, self.config.gamma, self.config.gae_lambda
        )

    def minibatch_losses(
        self,
        batch: RolloutBatch,
        advantages: Tensor,
        returns: Tensor,
        idx: Tensor,
    ) -> dict[str, Tensor]:
        """Losses of one minibatch (Schulman et al., 2017, eq. 7 and eq. 9).

        ``ratio = exp(log pi_theta(u|s) - log pi_old(u|s))``. The clipped
        surrogate ``min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)`` and
        the entropy estimate ``-log pi_theta`` are averaged over the
        non-intervened steps only. Over an empty mask both are 0. The value
        loss ``(V(s) - R)^2`` is averaged over all steps.

        Args:
            batch: Full rollout as tensors.
            advantages: Normalised advantages, shape ``(T,)``.
            returns: GAE returns, shape ``(T,)``.
            idx: Long tensor of the minibatch indices.

        Returns:
            Mapping with ``policy_loss``, ``value_loss``, ``entropy``,
            ``ratio`` (detached), ``values`` (detached) and ``approx_kl``.
        """
        obs = batch.obs[idx]
        log_prob_new = self.evaluate(obs, batch.pre_squash[idx])
        log_ratio = log_prob_new - batch.log_probs[idx]
        ratio = torch.exp(log_ratio)
        adv = advantages[idx]
        eps = self.config.clip_eps
        surrogate = torch.min(
            ratio * adv, torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        )
        mask = 1.0 - batch.intervened_mask[idx]
        count = mask.sum().clamp_min(1.0)
        policy_loss = -(surrogate * mask).sum() / count
        entropy = -(log_prob_new * mask).sum() / count
        values = self._value(obs)
        value_loss = (values - returns[idx]).pow(2).mean()
        with torch.no_grad():
            approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / count
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "ratio": ratio.detach(),
            "values": values.detach(),
            "approx_kl": approx_kl,
        }

    def _minibatches(self, n: int) -> Iterator[Tensor]:
        """Yield shuffled index tensors of size ``minibatch_size`` covering ``n``."""
        perm = self.rng.permutation(n)
        size = self.config.minibatch_size
        for start in range(0, n, size):
            yield torch.as_tensor(
                perm[start : start + size], dtype=torch.long, device=self.device
            )

    def _update(self) -> UpdateStats:
        """Run ``epochs`` passes of minibatch updates over the full rollout.

        Computes GAE, normalises the advantages over the rollout, optimises
        ``L = L_clip - entropy_coef * H + value_coef * L_V`` and clears the
        rollout buffer afterwards, so every rollout is used by exactly one
        update.

        Returns:
            Mean value loss (``critic_loss``), mean policy loss
            (``actor_loss``), mean ``V(s)`` of the last minibatch (``mean_q``)
            and diagnostics in ``extra``: ``first_minibatch_max_ratio_dev``
            (``max |ratio - 1|`` of the first minibatch of the first epoch),
            ``approx_kl``, ``clip_fraction``, ``entropy`` and
            ``intervened_fraction``.
        """
        advantages_np, returns_np = self.compute_advantages()
        batch = self.buffer.to_tensors(self.device)
        advantages = self._as_tensor(advantages_np)
        returns = self._as_tensor(returns_np)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + ADVANTAGE_EPS
        )
        n = len(self.buffer)
        intervened_fraction = float(batch.intervened_mask.mean().item())
        mask = 1.0 - batch.intervened_mask

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        first_ratio_dev: float | None = None
        last_values = torch.zeros(1, device=self.device)
        for _ in range(self.config.epochs):
            for idx in self._minibatches(n):
                losses = self.minibatch_losses(batch, advantages, returns, idx)
                loss = (
                    losses["policy_loss"]
                    - self.config.entropy_coef * losses["entropy"]
                    + self.config.value_coef * losses["value_loss"]
                )
                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()
                loss.backward()
                self._clip_gradients(self.policy.parameters())
                self._clip_gradients(self.value_net.parameters())
                self.policy_opt.step()
                self.value_opt.step()

                ratio = losses["ratio"]
                if first_ratio_dev is None:
                    first_ratio_dev = float((ratio - 1.0).abs().max().item())
                mb_mask = mask[idx]
                clipped = ((ratio - 1.0).abs() > self.config.clip_eps).float()
                clip_fractions.append(
                    float(
                        (
                            (clipped * mb_mask).sum() / mb_mask.sum().clamp_min(1.0)
                        ).item()
                    )
                )
                policy_losses.append(float(losses["policy_loss"].item()))
                value_losses.append(float(losses["value_loss"].item()))
                entropies.append(float(losses["entropy"].item()))
                kls.append(float(losses["approx_kl"].item()))
                last_values = losses["values"]

        self.buffer.clear()
        self._last_next_obs = None
        return UpdateStats(
            critic_loss=float(np.mean(value_losses)),
            actor_loss=float(np.mean(policy_losses)),
            mean_q=float(last_values.mean().item()),
            extra={
                "first_minibatch_max_ratio_dev": float(first_ratio_dev or 0.0),
                "approx_kl": float(np.mean(kls)),
                "clip_fraction": float(np.mean(clip_fractions)),
                "entropy": float(np.mean(entropies)),
                "intervened_fraction": intervened_fraction,
            },
        )
