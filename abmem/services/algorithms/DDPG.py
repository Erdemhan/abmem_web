import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from .actor_critic import RNNActor, RNNCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))           # shape: [24, 4]
            batch_next_states.append(np.array(next_state, copy=False)) # shape: [24, 4]
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states),           # [batch_size, 24, 4]devam
            np.array(batch_next_states),      # [batch_size, 24, 4]
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )

# RDPG Algoritması
class RDPG:
    def __init__(self, replay_buffer: ReplayBuffer, action_dim: int,
                 learning_rate_critic: float = 1e-3, learning_rate_actor: float = 1e-4,
                 noise: float = 0.01, noise_decay: float = 0.99, noise_min: float = 0.01):

        self.state_dim = 4
        self.action_dim = action_dim
        self.action_space_limits = [(0, 200)]

        self.actor = RNNActor(self.state_dim, action_dim, self.action_space_limits,
                              noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target = RNNActor(self.state_dim, action_dim, self.action_space_limits,
                                     noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

        self.critic = RNNCritic(self.state_dim, action_dim).to(device)
        self.critic_target = RNNCritic(self.state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        self.replay_buffer = replay_buffer

    def select_action(self, state_seq, hidden):
        state_tensor = torch.Tensor(state_seq).to(device)
        if state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(0)  # [seq, feature] → [1, seq, feature]
        
        action, hidden = self.actor(state_tensor, hidden)
        return action.squeeze(0).cpu().data.numpy(), hidden

    def train(self, iterations, batch_size=128, discount=0.95, tau=0.001):
        for it in range(iterations):
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)

            state = torch.Tensor(state).to(device)               # [batch, seq, feature]
            next_state = torch.Tensor(next_state).to(device)
            action = torch.Tensor(action).unsqueeze(1).to(device)  # [batch, 1, action_dim]
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)

            h0 = torch.zeros(1, batch_size, 32).to(device)
            c0 = torch.zeros(1, batch_size, 32).to(device)

            next_action, _ = self.actor_target(next_state, (h0, c0))
            target_Q = self.critic_target(next_state, next_action, (h0, c0))
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            current_Q = self.critic(state, action, (h0, c0))

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            new_actions, _ = self.actor(state, (h0, c0))
            actor_loss = -self.critic(state, new_actions, (h0, c0)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            #print(f"[Train] Iter {it} - Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}")

    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, next_state, action, reward, done))
