import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal

# Force CPU usage
device = torch.device("cpu")


# PPO için deneyim depolama (Episode bazlı - ReplayBuffer'dan farklı)
class RolloutBuffer:
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)

    def clear(self):
        self.storage = []
        self.ptr = 0

    def get_all(self):
        if not self.storage:
            return [], [], [], [], [], []
        
        states, actions, rewards, next_states, dones, log_probs = zip(*self.storage)
        return (
            np.array(states),
            np.array(actions), 
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(log_probs)
        )

    def size(self):
        return len(self.storage)


# PPO Policy Network - Stokastik politika
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_limits, hidden_dim=64, noise_std=0.05, noise_decay=0.99, noise_min=0.01):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Noise parametreleri
        self.noise_scale = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        
        # Risk duyarlılık parametreleri
        self.risk_tolerance = 0.5  # 0: risk-averse, 1: risk-seeking
        self.risk_learning_rate = 0.01
        
        # Aksiyon limitleri
        self.action_space_limits = action_space_limits
        
        # Ağ katmanları
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Ortalama ve standart sapma için ayrı katmanlar
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Log standart sapma için sınırlar
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action_and_log_prob(self, state):
        """Aksiyon seçer ve log probability döndürür"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Risk toleransını uygula
        if self.risk_tolerance < 0.5:
            std = std * (0.5 + self.risk_tolerance)
        else:
            std = std * (1 + (self.risk_tolerance - 0.5))
        
        # Normal dağılım oluştur
        normal = Normal(mean, std)
        
        # Aksiyonu örnekle
        z = normal.sample()
        
        # Aksiyon limitlerine göre ölçekle
        lower_bound = torch.tensor([0 for _ in self.action_space_limits], device=device)
        upper_bound = torch.tensor([limit[1] for limit in self.action_space_limits], device=device)
        action = torch.clamp(z, lower_bound, upper_bound)
        
        # Log olasılık hesapla
        log_prob = normal.log_prob(z).sum(dim=1, keepdim=True)
        
        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().numpy().flatten()
    
    def evaluate(self, state, action):
        """Policy evaluate için kullanılır"""
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        normal = Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=1, keepdim=True)
        entropy = normal.entropy().sum(dim=1, keepdim=True)
        
        return log_prob, entropy
    
    def update_risk_tolerance(self, reward, action):
        """Risk toleransını günceller"""
        if reward > 0:
            self.risk_tolerance = min(1.0, self.risk_tolerance + self.risk_learning_rate)
        else:
            self.risk_tolerance = max(0.0, self.risk_tolerance - self.risk_learning_rate)
    
    def update_noise(self):
        """Noise seviyesini günceller"""
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, self.noise_min)


# Value Network - Durum değer tahmini
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        
        return value


# PPO Algoritması
class PPO:
    def __init__(self, 
                 rollout_buffer,
                 action_dim,
                 learning_rate_actor=0.0003,
                 learning_rate_critic=0.0003,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=10,
                 noise=0.01,
                 noise_decay=0.99,
                 noise_min=0.01):
        
        self.action_dim = action_dim
        self.state_dim = 4
        self.action_space_limits = [(0, 200)]
        
        # PPO parametreleri
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        
        # Failstack mekanizması
        self.failstack = 0
        self.failstack_decay = 0.95
        self.failstack_increase = 0.1
        self.max_failstack = 1.0
        self.failstack_threshold = 0.5
        
        # Policy ve Value ağları
        self.policy = PolicyNetwork(
            self.state_dim, 
            self.action_dim, 
            self.action_space_limits,
            noise_std=noise, 
            noise_decay=noise_decay, 
            noise_min=noise_min
        ).to(device)
        
        self.value_net = ValueNetwork(self.state_dim).to(device)
        
        # Optimizer'lar
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate_actor)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate_critic)
        
        # Buffer
        self.rollout_buffer = rollout_buffer

    def update_failstack(self, success):
        """Failstack değerini günceller"""
        if success:
            self.failstack = max(0, self.failstack - self.failstack_decay)
        else:
            self.failstack = min(self.max_failstack, self.failstack + self.failstack_increase)

    def get_failstack_adjustment(self, action):
        """Failstack değerine göre aksiyonu ayarlar"""
        try:
            if isinstance(action, (list, np.ndarray)):
                action_array = np.array(action)
            else:
                action_array = np.array([action])
            
            action_value = float(action_array[0])

            if self.failstack > self.failstack_threshold:
                adjustment = (self.failstack - self.failstack_threshold) * (action_value - 0)
                return max(0, action_value - adjustment)
            return action_value
        except Exception as e:
            print(f"Failstack adjustment error: {e}")
            if isinstance(action, (list, np.ndarray)):
                return float(action[0])
            elif isinstance(action, (int, float)):
                return float(action)
            else:
                return 0.0

    def select_action(self, state):
        """Duruma göre aksiyon seçer - TD3 formatı ile uyumlu"""
        try:
            with torch.no_grad():
                # State'i numpy array'e çevir
                if not isinstance(state, np.ndarray):
                    state = np.array(state)
                
                # Policy'den aksiyon ve log prob al
                action, log_prob = self.policy.get_action_and_log_prob(state)
                
                # Failstack'e göre aksiyonu ayarla
                adjusted_action = self.get_failstack_adjustment(action)
                
                # List olarak döndür (agent_service.py actions[counter] için)
                if isinstance(adjusted_action, (int, float)):
                    return [float(adjusted_action)]
                else:
                    return [float(adjusted_action[0])] if len(adjusted_action) > 0 else [0.0]
                    
        except Exception as e:
            print(f"Select action error: {e}")
            return [0.0]

    def compute_gae(self, rewards, values, next_values, dones):
        """Generalized Advantage Estimation hesaplar"""
        advantages = []
        advantage = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i] if not dones[i] else 0
            else:
                next_value = values[i + 1]
            
            # Failstack'e göre reward'ı ayarla
            adjusted_reward = rewards[i] * (1 - self.failstack)
            
            delta = adjusted_reward + self.gamma * next_value * (1 - dones[i]) - values[i]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * advantage
            advantages.insert(0, advantage)
        
        return advantages

    def train(self, iterations=1, batch_size=64, discount=0.99, tau=0.001):
        """PPO training - TD3 formatı ile uyumlu"""
        if self.rollout_buffer.size() < batch_size:
            return
            
        # Buffer'dan verileri al
        states, actions, rewards, next_states, dones, old_log_probs = self.rollout_buffer.get_all()
        
        if len(states) == 0:
            return
        
        # Tensor'lara çevir
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(device)
        
        # Values hesapla
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
        
        # GAE hesapla
        advantages = self.compute_gae(
            rewards.cpu().numpy(), 
            values.cpu().numpy(),
            next_values.cpu().numpy(), 
            dones.cpu().numpy()
        )
        advantages = torch.FloatTensor(advantages).to(device)
        
        # Returns hesapla
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Mini batch'ler için rasgele indeksler
            batch_size_actual = min(batch_size, len(states))
            indices = np.random.randint(0, len(states), size=batch_size_actual)
            
            # Mini batch
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_advantages = advantages[indices]
            batch_returns = returns[indices]
            batch_old_log_probs = old_log_probs[indices]
            
            # Current policy evaluate et
            log_probs, entropy = self.policy.evaluate(batch_states, batch_actions)
            values_pred = self.value_net(batch_states).squeeze()
            
            # Ratio hesapla
            ratio = torch.exp(log_probs - batch_old_log_probs.unsqueeze(1))
            
            # Clipped surrogate loss
            surr1 = ratio * batch_advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pred, batch_returns)
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Her iki optimizer'ı da sıfırla
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            
            # Tek backward çağrısı - total_loss tüm komponentleri içeriyor
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            
            # Her iki optimizer'ı da güncelle
            self.policy_optimizer.step()
            self.value_optimizer.step()
        
        # Buffer'ı temizle
        self.rollout_buffer.clear()

    def add_to_rollout_buffer(self, state, action, next_state, reward, done):
        """Rollout buffer'a deneyim ekler - TD3 formatı ile uyumlu"""
        # Log probability hesapla
        with torch.no_grad():
            # Gerçek action'ın log olasılığını hesapla
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)
            log_prob_tensor, _ = self.policy.evaluate(state_tensor, action_tensor)
            log_prob = log_prob_tensor.cpu().numpy().flatten()
        
        # Buffer'a ekle
        self.rollout_buffer.add((state, action, reward, next_state, done, log_prob[0])) 