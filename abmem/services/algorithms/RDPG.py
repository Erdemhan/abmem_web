"""Recurrent Deterministic Policy Gradient (RDPG) in normalised spaces.

Reference: Heess, Hunt, Lillicrap and Silver (2015), "Memory-based control with
recurrent neural networks", Algorithm 1.

Notation follows the paper. ``h_t = (o_1, a_1, ..., a_{t-1}, o_t)`` is the
history up to the observation at step ``t``. The recurrent critic ``Q(h_t, a_t)``
reads the pairs ``(o_j, a_j)`` with the action actually taken at every step
(B3), and the recurrent actor ``mu(h_t)`` reads the observations. Histories are
truncated to windows of ``seq_len`` consecutive steps, and every window (in
training and in acting) is processed from a zero LSTM state, so acting and
learning use identically constructed inputs (B3, PLAN 1.6).

Update per sampled window ``k = 1..L`` (Algorithm 1):

* target ``y_k = r_k + gamma * (1 - terminated_k) * Q'(h_{k+1}, mu'(h_{k+1}))``
  where ``h_{L+1}`` extends the window by the final ``next_obs``,
* critic loss ``mean_k (Q(h_k, a_k) - y_k)^2``,
* actor loss ``-mean_k Q(h_k, mu(h_k))`` (deterministic policy gradient
  through time),
* Polyak update of both target networks with ``tau``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


from .algorithm_config import RDPGConfig
from .base import ActionResult, BaseAgentAlgorithm, PolicySpec, UpdateStats
from .networks import (
    RecurrentDeterministicActor,
    RecurrentQCritic,
    hard_update,
    soft_update,
)
from .replay_buffer import ReplayBuffer, SequenceBatch


def substituted_action_q(
    critic: RecurrentQCritic,
    obs_seq: Tensor,
    taken_actions: Tensor,
    query_actions: Tensor,
) -> Tensor:
    """Return ``Q(h_k, query_k)`` for every step ``k`` of a window.

    The history ``h_k`` contains the observations ``o_1..o_k`` and the actions
    actually taken at steps ``1..k-1`` (``taken_actions``). Only the action at
    step ``k`` itself is replaced by ``query_actions[:, k]``. This is the
    quantity ``Q(h_t, mu(h_t))`` of Heess et al. (2015), Algorithm 1, used by the
    target computation and the actor loss. The LSTM starts from a zero state at
    the first step of the window.

    Args:
        critic: Recurrent critic whose ``lstm`` input at step ``t`` is the
            concatenation of ``o_t`` and ``a_t``.
        obs_seq: Observations, shape ``(B, L, obs_dim)``.
        taken_actions: Actions taken in the window, shape ``(B, L, action_dim)``.
            The entry of the last step is never read.
        query_actions: Actions evaluated at each step, shape
            ``(B, L, action_dim)``. Gradients flow through them.

    Returns:
        Q-values of shape ``(B, L, 1)``.

    Raises:
        ValueError: If the shapes of the inputs do not match.
    """
    if obs_seq.dim() != 3 or taken_actions.shape != query_actions.shape:
        raise ValueError(
            f"incompatible shapes: obs {tuple(obs_seq.shape)}, taken "
            f"{tuple(taken_actions.shape)}, query {tuple(query_actions.shape)}"
        )
    if obs_seq.shape[:2] != taken_actions.shape[:2]:
        raise ValueError(
            f"obs {tuple(obs_seq.shape)} and actions {tuple(taken_actions.shape)} "
            "differ in batch or sequence length"
        )
    batch, length, _ = obs_seq.shape
    hidden_size = critic.lstm.hidden_size
    state = (
        obs_seq.new_zeros(1, batch, hidden_size),
        obs_seq.new_zeros(1, batch, hidden_size),
    )
    q_values: list[Tensor] = []
    for k in range(length):
        obs_k = obs_seq[:, k : k + 1]
        features, _ = critic.lstm(
            torch.cat([obs_k, query_actions[:, k : k + 1]], dim=-1), state
        )
        q_values.append(critic.head(features))
        if k < length - 1:
            _, state = critic.lstm(
                torch.cat([obs_k, taken_actions[:, k : k + 1]], dim=-1), state
            )
    return torch.cat(q_values, dim=1)


class RDPGAlgorithm(BaseAgentAlgorithm):
    """RDPG policy with windowed histories (Heess et al., 2015, Algorithm 1).

    Acting history: one deque holds the last ``seq_len`` observations passed to
    :meth:`select_action`, one per period. The runtime calls ``select_action``
    once per period followed by ``end_period``. A repeated call within the same
    period (before ``end_period``) is a new query for the same decision and
    replaces that period's observation instead of appending it, so with
    ``explore=False`` the action is a deterministic function of the history and
    the observation. With fewer than ``seq_len`` observations the window is
    shorter (no padding). The history is kept while frozen, since the
    evaluation phase acts with it.

    Attributes:
        actor: Online recurrent actor ``mu``.
        actor_target: Target actor ``mu'``.
        critic: Online recurrent critic ``Q``.
        critic_target: Target critic ``Q'``.
        actor_optimizer: Adam optimiser of the actor.
        critic_optimizer: Adam optimiser of the critic.
    """

    key = "RDPG"
    label = "RDPG"
    description = "Recurrent deterministic policy gradient over observation sequences (Heess et al., 2015)."
    is_learning = True
    is_on_policy = False
    config_class = RDPGConfig

    config: RDPGConfig
    buffer: ReplayBuffer

    def __init__(
        self, spec: PolicySpec, config: RDPGConfig, rng: np.random.Generator
    ) -> None:
        """Build the networks, their targets, the optimisers and the history.

        Args:
            spec: Problem description.
            config: RDPG hyperparameters.
            rng: Generator for exploration noise and sequence sampling.
        """
        super().__init__(spec, config, rng)
        obs_dim, action_dim = spec.obs_dim, spec.action_dim
        lstm_hidden, hidden = config.lstm_hidden, config.hidden_sizes
        self.actor = RecurrentDeterministicActor(
            obs_dim, action_dim, lstm_hidden, hidden
        ).to(self.device)
        self.actor_target = RecurrentDeterministicActor(
            obs_dim, action_dim, lstm_hidden, hidden
        ).to(self.device)
        self.critic = RecurrentQCritic(obs_dim, action_dim, lstm_hidden, hidden).to(
            self.device
        )
        self.critic_target = RecurrentQCritic(
            obs_dim, action_dim, lstm_hidden, hidden
        ).to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        for target in (self.actor_target, self.critic_target):
            target.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.lr_critic
        )
        self._history: deque[np.ndarray] = deque(maxlen=config.seq_len)
        self._period_open = False

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

    def _extra_state(self) -> dict[str, Any]:
        """Return the acting history window and the period flag."""
        return {
            "history": torch.from_numpy(self.history),
            "period_open": bool(self._period_open),
        }

    def _check_extra_state(self, extra: Mapping[str, Any]) -> None:
        """Require a history of shape ``(n, obs_dim)``.

        Raises:
            ValueError: If an entry is missing or the history has the wrong
                shape.
        """
        missing = {"history", "period_open"} - set(extra)
        if missing:
            raise ValueError(f"RDPG state lacks {sorted(missing)}")
        history = extra["history"]
        if (
            not isinstance(history, torch.Tensor)
            or history.dim() != 2
            or history.shape[1] != self.spec.obs_dim
        ):
            raise ValueError(
                f"RDPG state: history must have shape (n, {self.spec.obs_dim})"
            )

    def _load_extra_state(
        self, extra: Mapping[str, Any], *, restore_runtime: bool
    ) -> None:
        """Restore the acting history with ``restore_runtime``.

        A warm start keeps an empty history: the new run is a new trajectory
        and its windows must not contain observations of the source run.
        """
        self._history.clear()
        self._period_open = False
        if restore_runtime:
            for row in extra["history"].cpu().numpy().astype(np.float32):
                self._history.append(row.copy())
            self._period_open = bool(extra["period_open"]) and bool(self._history)

    # Acting -----------------------------------------------------------------

    @property
    def history(self) -> np.ndarray:
        """Observations of the current acting window, oldest first.

        Returns:
            A float32 copy of shape ``(n, obs_dim)`` with ``n <= seq_len``.
        """
        if not self._history:
            return np.zeros((0, self.spec.obs_dim), dtype=np.float32)
        return np.stack(list(self._history)).astype(np.float32, copy=True)

    def _record_observation(self, obs: np.ndarray) -> None:
        """Put ``obs`` into the history as the observation of the current period."""
        if self._period_open and self._history:
            self._history[-1] = obs.copy()
        else:
            self._history.append(obs.copy())
            self._period_open = True

    def select_action(self, obs: np.ndarray, explore: bool) -> ActionResult:
        """Return ``mu(h_t)`` of the current window, plus Gaussian noise if exploring.

        The actor runs over the window from a zero LSTM state and the action
        of the last step is used. Exploration noise ``N(0, sigma^2 I)`` is
        added only when ``explore`` is True, then clipped to ``[-1, 1]``.

        Args:
            obs: Normalised observation of this period, shape ``(obs_dim,)``.
            explore: Add exploration noise if True.

        Returns:
            The normalised action of shape ``(action_dim,)``.
        """
        self._record_observation(obs)
        window = self._as_tensor(self.history).unsqueeze(0)
        actions, _ = self.actor(window, None)
        u = actions[0, -1].cpu().numpy().astype(np.float32)
        if explore:
            u = self._add_exploration_noise(u)
        return ActionResult(u=u)

    def end_period(self) -> None:
        """Close the period of the history and apply the exploration decay."""
        self._period_open = False
        super().end_period()

    # Learning ---------------------------------------------------------------

    @property
    def ready_to_learn(self) -> bool:
        """True if ``learning_starts`` transitions and one full window are stored."""
        return super().ready_to_learn and len(self.buffer) >= self.config.seq_len

    def _critic_targets(self, batch: SequenceBatch) -> Tensor:
        """Return ``y_k = r_k + gamma * (1 - terminated_k) * Q'(h_{k+1}, mu'(h_{k+1}))``.

        The target networks run over the window extended by the final
        ``next_obs``, from a zero state.

        Args:
            batch: Sampled windows.

        Returns:
            Targets of shape ``(B, L, 1)``, without gradient.
        """
        with torch.no_grad():
            obs_ext = torch.cat([batch.obs, batch.next_obs[:, -1:]], dim=1)
            target_actions, _ = self.actor_target(obs_ext, None)
            taken_ext = torch.cat([batch.actions, target_actions[:, -1:]], dim=1)
            q_next = substituted_action_q(
                self.critic_target, obs_ext, taken_ext, target_actions
            )[:, 1:]
            return batch.rewards + self.config.gamma * (1.0 - batch.terminated) * q_next

    def _update(self) -> UpdateStats:
        """Run ``config.gradient_steps`` RDPG updates on windows of ``seq_len``.

        Returns:
            Mean critic and actor losses over the gradient steps and the mean
            ``Q(h_k, a_k)`` of the last batch.
        """
        critic_losses: list[float] = []
        actor_losses: list[float] = []
        mean_q = 0.0
        for _ in range(self.config.gradient_steps):
            batch = self.buffer.sample_sequences(
                self.config.batch_size, self.config.seq_len, self.device
            )
            targets = self._critic_targets(batch)

            q_taken = self.critic(batch.obs, batch.actions, None)
            critic_loss = nn.functional.mse_loss(q_taken, targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self._clip_gradients(self.critic.parameters())
            self.critic_optimizer.step()

            policy_actions, _ = self.actor(batch.obs, None)
            actor_loss = -substituted_action_q(
                self.critic, batch.obs, batch.actions, policy_actions
            ).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self._clip_gradients(self.actor.parameters())
            self.actor_optimizer.step()

            soft_update(self.critic_target, self.critic, self.config.tau)
            soft_update(self.actor_target, self.actor, self.config.tau)

            critic_losses.append(float(critic_loss.item()))
            actor_losses.append(float(actor_loss.item()))
            mean_q = float(q_taken.detach().mean().item())
        return UpdateStats(
            critic_loss=float(np.mean(critic_losses)),
            actor_loss=float(np.mean(actor_losses)),
            mean_q=mean_q,
        )
