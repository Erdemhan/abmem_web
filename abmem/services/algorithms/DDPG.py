import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .actor_critic import Actor, Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Add experiences to the buffer
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    # Add to buffer
    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    # Sample a random number of elements from the buffer
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))

        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )
    


class DDPG(object):
    def __init__(self,replay_buffer:ReplayBuffer,action_dim:int,
                 learning_rate_critic: float = 1e-3, learning_rate_actor: float = 1e-4,
                 noise: float = 0.01, noise_decay: float = 0.99, noise_min: float = 0.01):

        action_dim = action_dim
        state_dim = 4
        action_space_limits = [(0,200)]

        self.actor = Actor(state_dim, action_dim, action_space_limits,noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_space_limits,noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = learning_rate_actor)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = learning_rate_critic)

        self.replay_buffer = replay_buffer
        self.action_space_limits = action_space_limits


    # Action Selection - No Random Action
    def select_action(self, state):
        # Convert state to tensor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        # Get action from forward pass and convert to array
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    def train(self, iterations, batch_size=128, discount=0.95, tau=0.001):
        for it in range(iterations):
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.Tensor(state).to(device)
            next_state = torch.Tensor(next_state).to(device)
            action = torch.Tensor(action).to(device)
            reward = torch.Tensor(reward).to(device)
            done = torch.Tensor(done).to(device)

            # Train the Critic
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * discount * target_Q).detach()
            current_Q = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Train the Actor
            actor_loss = -self.critic(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Add to replay buffer
    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, next_state, action, reward, done))
