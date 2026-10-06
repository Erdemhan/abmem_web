"""Shared fixtures of the algorithm-layer tests.

The tests in this package are pure (no database). Two minimal policies are
provided to exercise the BaseAgentAlgorithm template and the contract helper
before the real algorithms land in wave 2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from abmem.models.enums import AlgorithmType
from abmem.services.algorithms.algorithm_config import DDPGConfig, RandomConfig
from abmem.services.algorithms.base import (
    ActionResult,
    BaseAgentAlgorithm,
    PolicySpec,
    UpdateStats,
)
from abmem.services.algorithms.networks import DeterministicActor, QCritic


def make_spec(action_dim: int = 1, obs_dim: int = 3) -> PolicySpec:
    """Return a valid PolicySpec with ``action_dim`` plants."""
    return PolicySpec(
        obs_dim=obs_dim,
        action_dim=action_dim,
        price_low=0.0,
        price_high=200.0,
        plant_variable_costs=tuple(10.0 + i for i in range(action_dim)),
        plant_static_costs=tuple(5.0 + i for i in range(action_dim)),
    )


class TinyActorCriticPolicy(BaseAgentAlgorithm):
    """Minimal off-policy learner used to test the template and the contract."""

    algorithm_type = AlgorithmType.DDPG
    is_learning = True
    is_on_policy = False
    config_class = DDPGConfig

    def __init__(
        self, spec: PolicySpec, config: DDPGConfig, rng: np.random.Generator
    ) -> None:
        super().__init__(spec, config, rng)
        self.actor = DeterministicActor(
            spec.obs_dim, spec.action_dim, config.hidden_sizes
        ).to(self.device)
        self.critic = QCritic(spec.obs_dim, spec.action_dim, config.hidden_sizes).to(
            self.device
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )
        self.select_calls_with_grad = 0

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        if torch.is_grad_enabled():
            self.select_calls_with_grad += 1
        u = self.actor(self._as_tensor(obs).unsqueeze(0)).squeeze(0).cpu().numpy()
        if explore:
            u = self._add_exploration_noise(u)
        return ActionResult(u=u.astype(np.float32))

    def _update(self) -> UpdateStats:
        assert self.buffer is not None
        critic_losses, actor_losses = [], []
        q = torch.zeros(1)
        for _ in range(self.config.gradient_steps):
            batch = self.buffer.sample(self.config.batch_size, self.device)
            q = self.critic(batch.obs, batch.actions)
            critic_loss = nn.functional.mse_loss(q, batch.rewards)
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self._clip_gradients(self.critic.parameters())
            self.critic_opt.step()
            actor_loss = -self.critic(batch.obs, self.actor(batch.obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            critic_losses.append(float(critic_loss.item()))
            actor_losses.append(float(actor_loss.item()))
        return UpdateStats(
            critic_loss=float(np.mean(critic_losses)),
            actor_loss=float(np.mean(actor_losses)),
            mean_q=float(q.mean().item()),
        )


class ConstantBaseline(BaseAgentAlgorithm):
    """Non-learning policy that always bids the middle of the action range."""

    algorithm_type = AlgorithmType.RANDOM
    is_learning = False
    is_on_policy = False
    config_class = RandomConfig

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        return ActionResult(u=np.zeros(self.spec.action_dim, dtype=np.float32))


def small_ddpg_config(**changes: object) -> DDPGConfig:
    """Return a small, fast DDPGConfig for tests."""
    base = DDPGConfig(
        batch_size=8, buffer_size=256, learning_starts=16, hidden_sizes=(16,)
    )
    return base.replace(**changes) if changes else base


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded numpy generator."""
    return np.random.default_rng(0)


@pytest.fixture(params=[1, 5], ids=["action_dim=1", "action_dim=5"])
def spec(request: pytest.FixtureRequest) -> PolicySpec:
    """PolicySpec for single-plant and multi-plant agents (A1)."""
    return make_spec(action_dim=request.param)


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Make torch initialisation deterministic per test."""
    torch.manual_seed(0)
