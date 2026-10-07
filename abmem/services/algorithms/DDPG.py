"""Deep Deterministic Policy Gradient (Lillicrap et al., 2016, Algorithm 1).

The policy operates in the normalised spaces of ADR-003: observations are
normalised float32 vectors and actions lie in ``[-1, 1]``, one entry per plant.
The actor ends in ``tanh`` and contains no exploration noise (C3). Gaussian
exploration noise is added only in :meth:`DDPGAlgorithm.select_action` with
``explore=True``. The critic and the replay buffer use the same normalised
actions (C5). Rewards are stored and used as unmodified floats (C1, C7).

References:
    Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y.,
    Silver, D., and Wierstra, D. (2016). Continuous control with deep
    reinforcement learning. ICLR 2016. arXiv:1509.02971.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


from .algorithm_config import DDPGConfig
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec, UpdateStats
from .networks import DeterministicActor, QCritic, soft_update
from .replay_buffer import Batch, ReplayBuffer


class DDPGAlgorithm(BaseAgentAlgorithm):
    """DDPG with target networks and Polyak averaging.

    Notation follows Lillicrap et al. (2016), Algorithm 1: ``mu`` is the actor
    with parameters ``theta^mu``, ``Q`` the critic with parameters
    ``theta^Q``, and ``mu'``, ``Q'`` their target copies.

    Attributes:
        actor: Online deterministic actor mu(s), tanh output in ``[-1, 1]``.
        actor_target: Target actor mu'(s), never perturbed by noise.
        critic: Online critic Q(s, a).
        critic_target: Target critic Q'(s, a).
        actor_optimizer: Adam optimiser of the actor.
        critic_optimizer: Adam optimiser of the critic.
    """

    key = "DDPG"
    label = "DDPG"
    description = "Deep deterministic policy gradient, off-policy actor-critic (Lillicrap et al., 2016)."
    is_learning = True
    is_on_policy = False
    config_class = DDPGConfig

    config: DDPGConfig
    buffer: ReplayBuffer

    def __init__(
        self, spec: PolicySpec, config: DDPGConfig, rng: np.random.Generator
    ) -> None:
        """Build actor, critic, their target copies and the Adam optimisers.

        The target networks start as exact copies of the online networks
        (Algorithm 1, "Initialize target network Q' and mu'").

        Args:
            spec: Problem description of the agent.
            config: DDPG hyperparameters.
            rng: Random generator for exploration noise and buffer sampling.

        Raises:
            TypeError: If an argument has the wrong type.
        """
        super().__init__(spec, config, rng)
        hidden = config.hidden_sizes
        self.actor: DeterministicActor = DeterministicActor(
            spec.obs_dim, spec.action_dim, hidden
        ).to(self.device)
        self.critic: QCritic = QCritic(spec.obs_dim, spec.action_dim, hidden).to(
            self.device
        )
        self.actor_target: DeterministicActor = copy.deepcopy(self.actor)
        self.critic_target: QCritic = copy.deepcopy(self.critic)
        for target in (self.actor_target, self.critic_target):
            target.requires_grad_(False)
            target.eval()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )

    def _state_modules(self) -> dict[str, nn.Module]:
        """Return the online and target networks by name."""
        return {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
        }

    def _state_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Return the actor and critic optimisers by name."""
        return {"actor": self.actor_optimizer, "critic": self.critic_optimizer}

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return ``mu(o)`` or, when exploring, ``clip(mu(o) + N(0, sigma^2 I), -1, 1)``.

        This is the exploration policy of Lillicrap et al. (2016), section 3,
        with uncorrelated Gaussian noise instead of an Ornstein-Uhlenbeck
        process. Sigma decays once per period in :meth:`end_period`.

        Args:
            obs: Normalised observation, float32, shape ``(obs_dim,)``
                (validated by the base class).
            explore: Add Gaussian exploration noise if True. With False the
                action is a deterministic function of ``obs``.

        Returns:
            The normalised action of shape ``(action_dim,)``.
        """
        u = (
            self.actor(self._as_tensor(obs).unsqueeze(0))
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        if explore:
            u = self._add_exploration_noise(u)
        return ActionResult(u=u)

    def compute_target(self, batch: Batch) -> torch.Tensor:
        """Return the critic target ``y = r + gamma * (1 - d) * Q'(s', mu'(s'))``.

        Algorithm 1, "Set y_i". The target actor is evaluated without noise
        (C3) and no gradient flows into the target networks.

        Args:
            batch: Minibatch of transitions.

        Returns:
            Targets of shape ``(B, 1)``.
        """
        with torch.no_grad():
            next_actions = self.actor_target(batch.next_obs)
            next_q = self.critic_target(batch.next_obs, next_actions)
            return batch.rewards + self.config.gamma * (1.0 - batch.terminated) * next_q

    def _gradient_step(self, batch: Batch) -> tuple[float, float, float]:
        """Run one critic step, one actor step and the soft target update.

        Critic: minimise ``L = mean((y - Q(s, a))^2)`` with ``a`` the stored
        normalised action (Algorithm 1, "Update critic"). Actor: minimise
        ``-mean(Q(s, mu(s)))``, whose gradient is the sampled deterministic
        policy gradient (Algorithm 1, "Update the actor policy"). Targets:
        ``theta' <- tau * theta + (1 - tau) * theta'`` (Algorithm 1, "Update
        the target networks").

        Args:
            batch: Minibatch of transitions.

        Returns:
            Critic loss, actor loss and mean Q(s, a) of the batch.
        """
        y = self.compute_target(batch)
        q = self.critic(batch.obs, batch.actions)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip_gradients(self.critic.parameters())
        self.critic_optimizer.step()

        # The critic is held fixed while differentiating through it.
        self.critic.requires_grad_(False)
        try:
            actor_loss = -self.critic(batch.obs, self.actor(batch.obs)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
        finally:
            self.critic.requires_grad_(True)
        self._clip_gradients(self.actor.parameters())
        self.actor_optimizer.step()

        soft_update(self.critic_target, self.critic, self.config.tau)
        soft_update(self.actor_target, self.actor, self.config.tau)
        return (
            float(critic_loss.item()),
            float(actor_loss.item()),
            float(q.detach().mean().item()),
        )

    def _update(self) -> UpdateStats:
        """Run ``config.gradient_steps`` DDPG gradient steps.

        Called by the base class only when the buffer holds at least
        ``config.learning_starts`` transitions (B7). Since the config enforces
        ``learning_starts >= batch_size``, every step can sample a fresh
        minibatch of ``batch_size`` transitions.

        Returns:
            Mean critic and actor losses over the steps and the mean Q-value of
            the last batch.
        """
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        mean_q = 0.0
        for _ in range(self.config.gradient_steps):
            batch = self.buffer.sample(self.config.batch_size, self.device)
            critic_loss, actor_loss, mean_q = self._gradient_step(batch)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)
        return UpdateStats(
            critic_loss=float(np.mean(critic_losses)),
            actor_loss=float(np.mean(actor_losses)),
            mean_q=mean_q,
        )
