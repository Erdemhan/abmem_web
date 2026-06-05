import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .actor_critic import Actor, Critic

# Force CPU usage
device = torch.device("cpu")


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
            batch_states.append(np.asarray(state))
            batch_next_states.append(np.asarray(next_state))
            batch_actions.append(np.asarray(action))
            batch_rewards.append(np.asarray(reward))
            batch_dones.append(np.asarray(done))

        return (
            np.array(batch_states),
            np.array(batch_next_states),
            np.array(batch_actions),
            np.array(batch_rewards).reshape(-1, 1),
            np.array(batch_dones).reshape(-1, 1)
        )
    


class TD3(object):
    def __init__(self,replay_buffer:ReplayBuffer,action_dim:int,
                 learning_rate_critic: float = 1e-3, learning_rate_actor: float = 1e-4,
                 noise: float = 0.01, noise_decay: float = 0.99, noise_min: float = 0.01):

        action_dim = action_dim
        state_dim = 4
        action_space_limits = [(0,200)]

        # Failstack mekanizması için yeni parametreler
        self.failstack = 0
        self.failstack_decay = 0.95  # Başarılı işlemlerde failstack'in azalma oranı
        self.failstack_increase = 0.1  # Başarısız işlemlerde failstack'in artma oranı
        self.max_failstack = 1.0  # Maksimum failstack değeri
        self.failstack_threshold = 0.5  # Failstack etkisinin başlayacağı eşik değeri

        self.actor = Actor(state_dim, action_dim, action_space_limits,noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target = Actor(state_dim, action_dim, action_space_limits,noise_std=noise, noise_decay=noise_decay, noise_min=noise_min).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

        # TD3 uses two critics
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate_critic)

        self.replay_buffer = replay_buffer
        self.action_space_limits = action_space_limits
        
        # TD3 specific parameters
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.total_it = 0

    def update_failstack(self, success: bool):
        """Failstack değerini günceller"""
        if success:
            self.failstack = max(0, self.failstack - self.failstack_decay)
        else:
            self.failstack = min(self.max_failstack, self.failstack + self.failstack_increase)

    def get_failstack_adjustment(self, action):
        """Failstack değerine göre aksiyonu ayarlar"""
        try:
            # Action'ı numpy array'e çevir
            if isinstance(action, (list, np.ndarray)):
                action_array = np.array(action)
            else:
                action_array = np.array([action])
            
            # İlk elemanı al ve float'a çevir
            action_value = float(action_array[0])

            # Failstack kontrolü
            if self.failstack > self.failstack_threshold:
                # Failstack yüksekse, aksiyonu daha düşük fiyata doğru ayarla
                adjustment = (self.failstack - self.failstack_threshold) * (action_value - 0)
                return max(0, action_value - adjustment)
            return action_value
        except Exception as e:
            print(f"Failstack adjustment error: {e}")
            # Hata durumunda orijinal action'ı döndür
            if isinstance(action, (list, np.ndarray)):
                return float(action[0])
            elif isinstance(action, (int, float)):
                return float(action)
            else:
                return 0.0  # Varsayılan değer

    def select_action(self, state):
        try:
            with torch.no_grad():
                state = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = self.actor(state).cpu().data.numpy().flatten()
                # Failstack'e göre aksiyonu ayarla
                adjusted_action = self.get_failstack_adjustment(action)
                # List olarak döndür (agent_service.py actions[counter] için)
                if isinstance(adjusted_action, (int, float)):
                    return [float(adjusted_action)]
                else:
                    return [float(adjusted_action[0])] if len(adjusted_action) > 0 else [0.0]
        except Exception as e:
            print(f"Select action error: {e}")
            # Hata durumunda varsayılan bir değer döndür
            return [0.0]

    def train(self, iterations, batch_size=128, discount=0.95, tau=0.001):
        for it in range(iterations):
            self.total_it += 1
            
            # Buffer size kontrolü
            if len(self.replay_buffer.storage) < batch_size:
                return
            
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)

            # Failstack'e göre reward'ı ayarla
            adjusted_reward = reward * (1 - self.failstack)  # Failstack yüksekse reward düşer

            with torch.no_grad():
                # Select next action according to target policy with noise
                noise = torch.randn_like(action) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                
                next_action = self.actor_target(next_state) + noise
                next_action = next_action.clamp(-1, 1)

                # Compute target Q-values with adjusted reward
                target_Q1 = self.critic1_target(next_state, next_action)
                target_Q2 = self.critic2_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = adjusted_reward + ((1 - done) * discount * target_Q).detach()

            # Update critics
            current_Q1 = self.critic1(state, action)
            current_Q2 = self.critic2(state, action)
            
            critic1_loss = F.mse_loss(current_Q1, target_Q)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # Delayed policy updates
            if self.total_it % self.policy_freq == 0:
                # Update actor
                actor_loss = -self.critic1(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Add to replay buffer
    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, next_state, action, reward, done)) 