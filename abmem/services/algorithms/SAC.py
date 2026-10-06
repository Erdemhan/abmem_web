"""Soft Actor-Critic in the normalised action space.

Implements SAC with twin soft Q-functions, Polyak-averaged target networks, a
tanh-squashed Gaussian policy and automatic entropy temperature tuning.
Policy, replay buffer and critics all work with actions ``u in [-1, 1]``
(ADR-003, C5). Mapping ``u`` to prices happens outside the algorithm.

References:
    Haarnoja, T., Zhou, A., Abbeel, P., Levine, S. (2018a). Soft Actor-Critic:
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic
    Actor. ICML.
    Haarnoja, T., Zhou, A., Hartikainen, K., et al. (2018b). Soft Actor-Critic
    Algorithms and Applications. arXiv:1812.05905.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import torch
from torch.nn import functional as F

from abmem.models.enums import AlgorithmType

from .algorithm_config import SACConfig
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec, UpdateStats
from .networks import QCritic, hard_update, soft_update
from .replay_buffer import Batch, ReplayBuffer
from .sac_networks import SquashedGaussianActor


class SACAlgorithm(BaseAgentAlgorithm):
    """Soft Actor-Critic with automatic temperature (Haarnoja et al., 2018a, b).

    Attributes:
        actor: Squashed Gaussian policy.
        q1: First soft Q-function.
        q2: Second soft Q-function.
        q1_target: Polyak-averaged copy of ``q1``.
        q2_target: Polyak-averaged copy of ``q2``.
        log_alpha: Log of the entropy temperature (trainable scalar tensor).
        target_entropy: Target entropy of the temperature loss.
    """

    algorithm_type: ClassVar[AlgorithmType] = AlgorithmType.SAC
    is_learning: ClassVar[bool] = True
    is_on_policy: ClassVar[bool] = False
    config_class: ClassVar[type[SACConfig]] = SACConfig

    config: SACConfig
    buffer: ReplayBuffer

    def __init__(
        self, spec: PolicySpec, config: SACConfig, rng: np.random.Generator
    ) -> None:
        """Build networks, targets, optimisers and the temperature.

        Args:
            spec: Problem description.
            config: SAC hyperparameters.
            rng: Random generator of this policy. It also seeds the torch
                generator used for policy sampling, so one seed fixes all
                randomness of the policy.
        """
        super().__init__(spec, config, rng)
        obs_dim, action_dim = spec.obs_dim, spec.action_dim
        hidden = config.hidden_sizes
        self.actor = SquashedGaussianActor(obs_dim, action_dim, hidden).to(self.device)
        self.q1 = QCritic(obs_dim, action_dim, hidden).to(self.device)
        self.q2 = QCritic(obs_dim, action_dim, hidden).to(self.device)
        self.q1_target = QCritic(obs_dim, action_dim, hidden).to(self.device)
        self.q2_target = QCritic(obs_dim, action_dim, hidden).to(self.device)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)
        for param in (*self.q1_target.parameters(), *self.q2_target.parameters()):
            param.requires_grad_(False)

        self.log_alpha = torch.tensor(
            math.log(config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.target_entropy: float = (
            -float(action_dim)
            if config.target_entropy is None
            else float(config.target_entropy)
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(
            [*self.q1.parameters(), *self.q2.parameters()], lr=config.lr_critic
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)

        self._torch_gen = torch.Generator(device=self.device)
        self._torch_gen.manual_seed(int(rng.integers(0, 2**63 - 1)))

    @property
    def alpha(self) -> float:
        """Current entropy temperature ``exp(log_alpha)``."""
        return float(self.log_alpha.detach().exp().item())

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return ``u ~ pi(.|o)`` when exploring, else ``tanh(mu(o))``.

        Args:
            obs: Normalised observation, shape ``(obs_dim,)``.
            explore: Sample from the stochastic policy if True.

        Returns:
            The normalised action in ``[-1, 1]``.
        """
        obs_t = self._as_tensor(obs).unsqueeze(0)
        if explore:
            u, _ = self.actor.sample(obs_t, generator=self._torch_gen)
        else:
            u = self.actor.deterministic(obs_t)
        return ActionResult(u=u.squeeze(0).cpu().numpy().astype(np.float32))

    def exploration_scale(self) -> float:
        """Return the entropy temperature alpha."""
        return self.alpha

    def critic_target(self, batch: Batch) -> torch.Tensor:
        """Soft Bellman target with clipped double Q.

        ``y = r + gamma * (1 - d) * (min_j Q'_j(s', u') - alpha * log pi(u'|s'))``,
        ``u' ~ pi(.|s')`` (Haarnoja et al., 2018a, eq. 3, 5-6, with the two
        Q-functions of section 4.2 and target networks).

        Args:
            batch: Minibatch of transitions.

        Returns:
            Targets of shape ``(B, 1)``, without gradient.
        """
        with torch.no_grad():
            next_u, next_log_prob = self.actor.sample(
                batch.next_obs, generator=self._torch_gen
            )
            next_q = torch.min(
                self.q1_target(batch.next_obs, next_u),
                self.q2_target(batch.next_obs, next_u),
            )
            soft_value = next_q - self.log_alpha.exp() * next_log_prob
            return (
                batch.rewards
                + self.config.gamma * (1.0 - batch.terminated) * soft_value
            )

    def _update(self) -> UpdateStats:
        """Run ``config.gradient_steps`` SAC iterations.

        Each iteration (Haarnoja et al., 2018b, Algorithm 1) performs:

        1. Critic step on ``J_Q = E[(Q_j(s, u) - y)^2] / 2`` for both Q-functions
           (eq. 5), ``u`` taken from the buffer.
        2. Policy step on ``J_pi = E[alpha * log pi(f(eps; s)|s) - min_j Q_j(s,
           f(eps; s))]`` with the reparameterisation ``f`` (eq. 12-13 of 2018a).
        3. Temperature step on ``J(alpha) = E[-alpha (log pi(u|s) +
           target_entropy)]`` (2018b, eq. 18), optimised in ``log alpha``.
        4. Polyak update of both target Q-functions with ``tau``.

        Returns:
            Mean losses over the iterations and the mean Q of the last batch.
        """
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        alpha_losses: list[float] = []
        entropies: list[float] = []
        mean_q = 0.0
        for _ in range(self.config.gradient_steps):
            batch = self.buffer.sample(self.config.batch_size, self.device)

            # 1. Soft Q-function regression.
            target = self.critic_target(batch)
            q1 = self.q1(batch.obs, batch.actions)
            q2 = self.q2(batch.obs, batch.actions)
            critic_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self._clip_gradients([*self.q1.parameters(), *self.q2.parameters()])
            self.critic_opt.step()

            # 2. Policy improvement with the reparameterisation trick.
            alpha = self.log_alpha.detach().exp()
            u, log_prob = self.actor.sample(batch.obs, generator=self._torch_gen)
            q_pi = torch.min(self.q1(batch.obs, u), self.q2(batch.obs, u))
            actor_loss = (alpha * log_prob - q_pi).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self._clip_gradients(self.actor.parameters())
            self.actor_opt.step()

            # 3. Automatic temperature adjustment.
            alpha_loss = -(
                self.log_alpha * (log_prob.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # 4. Target networks.
            soft_update(self.q1_target, self.q1, self.config.tau)
            soft_update(self.q2_target, self.q2, self.config.tau)

            critic_losses.append(float(critic_loss.item()))
            actor_losses.append(float(actor_loss.item()))
            alpha_losses.append(float(alpha_loss.item()))
            entropies.append(float(-log_prob.detach().mean().item()))
            mean_q = float(torch.min(q1, q2).detach().mean().item())

        # Gradients of the critics accumulated by the policy loss are cleared
        # so that no stale gradient remains on the online networks.
        self.critic_opt.zero_grad()
        return UpdateStats(
            critic_loss=float(np.mean(critic_losses)),
            actor_loss=float(np.mean(actor_losses)),
            mean_q=mean_q,
            extra={
                "alpha": self.alpha,
                "alpha_loss": float(np.mean(alpha_losses)),
                "entropy": float(np.mean(entropies)),
            },
        )
