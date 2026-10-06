"""Twin Delayed Deep Deterministic policy gradient (TD3).

Reference: Fujimoto, van Hoof and Meger (2018), "Addressing function
approximation error in actor-critic methods", ICML, Algorithm 1.

The implementation operates only in the normalised spaces of ADR-003:
observations are normalised float32 vectors and actions are float32 vectors in
``[-1, 1]``, one entry per plant. Three mechanisms distinguish TD3 from DDPG:

* Clipped double Q-learning (Section 4.2): two critics, the TD target uses the
  minimum of the two target critics.
* Target policy smoothing (Section 5.3, eq. 14-15): clipped Gaussian noise is
  added to the target action, which is then clamped to the action bounds
  ``[-1, 1]`` (C5).
* Delayed policy updates (Section 5.2): the actor and all target networks are
  updated once every ``policy_delay`` critic updates.

The fail-stack rule, action shrinking and reward multipliers live outside the
algorithm (ADR-004, C7). Exploration noise is added only in
:meth:`TD3Algorithm.select_action` with ``explore=True`` (C3).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from abmem.models.enums import AlgorithmType

from .algorithm_config import TD3Config
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec, UpdateStats
from .networks import DeterministicActor, QCritic, soft_update
from .replay_buffer import Batch, ReplayBuffer


class TD3Algorithm(BaseAgentAlgorithm):
    """TD3 policy with twin critics, target policy smoothing and delayed updates.

    Attributes:
        actor: Online deterministic actor mu_phi.
        actor_target: Target actor mu_phi'.
        critic1: First online critic Q_theta1.
        critic2: Second online critic Q_theta2.
        critic1_target: Target of ``critic1``.
        critic2_target: Target of ``critic2``.
        actor_optimizer: Adam optimiser of the actor.
        critic_optimizer: Adam optimiser of both critics.
        critic_updates: Number of critic gradient steps taken so far.
        actor_updates: Number of actor gradient steps taken so far.
    """

    algorithm_type = AlgorithmType.TD3
    is_learning = True
    is_on_policy = False
    config_class = TD3Config

    config: TD3Config

    def __init__(
        self, spec: PolicySpec, config: TD3Config, rng: np.random.Generator
    ) -> None:
        """Build the networks, target networks, optimisers and replay buffer.

        Target networks start as exact copies of the online networks
        (Algorithm 1, initialisation step).

        Args:
            spec: Problem description of the agent.
            config: TD3 hyperparameters.
            rng: Random generator owned by this policy. It drives exploration
                noise, minibatch sampling and (through a derived seed) the
                target policy smoothing noise.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        super().__init__(spec, config, rng)
        hidden = config.hidden_sizes
        self.actor = DeterministicActor(spec.obs_dim, spec.action_dim, hidden).to(
            self.device
        )
        self.critic1 = QCritic(spec.obs_dim, spec.action_dim, hidden).to(self.device)
        self.critic2 = QCritic(spec.obs_dim, spec.action_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        for target in (self.actor_target, self.critic1_target, self.critic2_target):
            target.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            [*self.critic1.parameters(), *self.critic2.parameters()],
            lr=config.lr_critic,
        )

        # Smoothing noise is drawn from a torch generator seeded from ``rng`` so
        # that the whole policy is reproducible from its numpy generator.
        self._noise_generator = torch.Generator(device=self.device)
        self._noise_generator.manual_seed(int(rng.integers(0, 2**63 - 1)))

        self.critic_updates = 0
        self.actor_updates = 0

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return ``mu(s)``, perturbed by clipped Gaussian noise if exploring.

        With ``explore=True`` the action is ``clip(mu(s) + N(0, sigma^2 I), -1, 1)``
        (Algorithm 1, action selection) with the current exploration sigma.

        Args:
            obs: Normalised observation, float32, shape ``(obs_dim,)``.
            explore: Add exploration noise if True.

        Returns:
            The normalised action, float32, shape ``(action_dim,)``.
        """
        obs_t = self._as_tensor(obs).unsqueeze(0)
        u = self.actor(obs_t).squeeze(0).cpu().numpy().astype(np.float32)
        if explore:
            u = self._add_exploration_noise(u)
        return ActionResult(u=u)

    def target_actions(self, next_obs: Tensor) -> Tensor:
        """Return smoothed target actions (Fujimoto et al., 2018, eq. 14-15).

        ``a' = clip(mu'(s') + clip(eps, -c, c), -1, 1)`` with
        ``eps ~ N(0, policy_noise^2 I)`` and ``c = noise_clip``, computed in the
        normalised action space (C5).

        Args:
            next_obs: Next observations, shape ``(B, obs_dim)``, on ``self.device``.

        Returns:
            Target actions of shape ``(B, action_dim)`` in ``[-1, 1]``.
        """
        with torch.no_grad():
            mu = self.actor_target(next_obs)
            eps = torch.randn(
                mu.shape,
                generator=self._noise_generator,
                device=mu.device,
                dtype=mu.dtype,
            )
            eps = (eps * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            return (mu + eps).clamp(-1.0, 1.0)

    def td_target(
        self, rewards: Tensor, next_obs: Tensor, terminated: Tensor
    ) -> Tensor:
        """Return the clipped double-Q target (Section 4.2 and Algorithm 1).

        ``y = r + gamma * (1 - terminated) * min(Q1'(s', a'), Q2'(s', a'))`` with
        ``a'`` from :meth:`target_actions`. The reward enters unchanged.

        Args:
            rewards: Shape ``(B, 1)``.
            next_obs: Shape ``(B, obs_dim)``.
            terminated: Shape ``(B, 1)``, 1.0 for terminal transitions.

        Returns:
            TD targets of shape ``(B, 1)``, detached from the graph.
        """
        with torch.no_grad():
            next_actions = self.target_actions(next_obs)
            q1 = self.critic1_target(next_obs, next_actions)
            q2 = self.critic2_target(next_obs, next_actions)
            return rewards + self.config.gamma * (1.0 - terminated) * torch.min(q1, q2)

    def _critic_step(self, batch: Batch) -> tuple[float, float]:
        """Regress both critics to the shared TD target.

        Args:
            batch: Minibatch of transitions.

        Returns:
            The optimised loss ``MSE(Q1, y) + MSE(Q2, y)`` and the mean of
            ``Q1(s, a)`` over the batch.
        """
        y = self.td_target(batch.rewards, batch.next_obs, batch.terminated)
        q1 = self.critic1(batch.obs, batch.actions)
        q2 = self.critic2(batch.obs, batch.actions)
        loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self._clip_gradients([*self.critic1.parameters(), *self.critic2.parameters()])
        self.critic_optimizer.step()
        self.critic_updates += 1
        return float(loss.item()), float(q1.detach().mean().item())

    def _actor_step(self, batch: Batch) -> float:
        """Deterministic policy gradient step on ``-mean Q1(s, mu(s))``, then
        Polyak update of all target networks (Algorithm 1, delayed branch).

        Args:
            batch: Minibatch of transitions.

        Returns:
            The actor loss.
        """
        actor_loss = -self.critic1(batch.obs, self.actor(batch.obs)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()
        self.actor_updates += 1

        tau = self.config.tau
        soft_update(self.critic1_target, self.critic1, tau)
        soft_update(self.critic2_target, self.critic2, tau)
        soft_update(self.actor_target, self.actor, tau)
        return float(actor_loss.item())

    def _update(self) -> UpdateStats:
        """Run ``config.gradient_steps`` TD3 iterations (Algorithm 1).

        Each iteration samples a minibatch and updates both critics. Every
        ``policy_delay``-th critic update (counted over the lifetime of the
        policy) is followed by an actor update and the target updates.

        Returns:
            Mean critic loss, mean actor loss (None if no actor step was taken
            in this call), mean Q1 of the last batch, and the cumulative update
            counts in ``extra``.
        """
        buffer = self.buffer
        assert isinstance(buffer, ReplayBuffer)
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        mean_q = 0.0
        for _ in range(self.config.gradient_steps):
            batch = buffer.sample(self.config.batch_size, self.device)
            critic_loss, mean_q = self._critic_step(batch)
            critic_losses.append(critic_loss)
            if self.critic_updates % self.config.policy_delay == 0:
                actor_losses.append(self._actor_step(batch))
        return UpdateStats(
            critic_loss=float(np.mean(critic_losses)),
            actor_loss=float(np.mean(actor_losses)) if actor_losses else None,
            mean_q=mean_q,
            extra={
                "critic_updates": float(self.critic_updates),
                "actor_updates": float(self.actor_updates),
            },
        )
