import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class SAC(object):
    def __init__(self, replay_buffer, action_dim, 
                 learning_rate_critic=3e-4, learning_rate_actor=3e-4, learning_rate_alpha=3e-4,
                 initial_alpha=0.2, target_entropy=None, tau=0.005, gamma=0.99):
        
        state_dim = 4
        action_space_limits = [(0, 200)]
        
        from .sac_networks import SACPolicy, SoftQNetwork
        
        # Initialize the policy network
        self.policy = SACPolicy(state_dim, action_dim, action_space_limits).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate_actor)
        
        # Initialize two Q-networks to mitigate positive bias
        self.q_net1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.q_net2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.q_net1_target = SoftQNetwork(state_dim, action_dim).to(device)
        self.q_net2_target = SoftQNetwork(state_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.q_net1_target.load_state_dict(self.q_net1.state_dict())
        self.q_net2_target.load_state_dict(self.q_net2.state_dict())
        
        # Setup optimizers
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=learning_rate_critic)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=learning_rate_critic)
        
        # Set replay buffer
        self.replay_buffer = replay_buffer
        
        # Set entropy-related parameters
        self.target_entropy = -action_dim if target_entropy is None else target_entropy
        self.log_alpha = torch.tensor(np.log(initial_alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate_alpha)
        
        # Set other hyperparameters
        self.tau = tau
        self.gamma = gamma
        self.action_space_limits = action_space_limits
        
        # Eğitim istatistikleri için
        self.train_iteration = 0
        self.q1_loss_history = []
        self.q2_loss_history = []
        self.policy_loss_history = []
        self.alpha_history = []
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, evaluate=False):
        # Convert state to tensor
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # Get action from policy
        with torch.no_grad():
            if evaluate:
                action, _ = self.policy.sample(state, deterministic=True)
            else:
                action, _ = self.policy.sample(state, deterministic=False)
        return action.cpu().numpy().flatten()
    
    def train(self, iterations, batch_size=256):
        
        if len(self.replay_buffer.storage) < batch_size:
            return
        
        # Loss değerlerini takip etmek için
        avg_q1_loss = 0
        avg_q2_loss = 0
        avg_policy_loss = 0
        avg_alpha = 0
            
        for it in range(iterations):
            self.train_iteration += 1
            
            # Sample from replay buffer
            state, next_state, action, reward, done = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).to(device)
            done = torch.FloatTensor(done).to(device)
            
            # Gradient clipping için maximum norm
            max_grad_norm = 1.0
            
            # Train Q-networks
            with torch.no_grad():
                # Sample action from policy
                next_action, next_log_prob = self.policy.sample(next_state)
                
                # Target Q-values
                next_q1 = self.q_net1_target(next_state, next_action)
                next_q2 = self.q_net2_target(next_state, next_action)
                next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
                target_q = reward + (1 - done) * self.gamma * next_q
            
            # Current Q-values
            current_q1 = self.q_net1(state, action)
            current_q2 = self.q_net2(state, action)
            
            # Compute Q-network losses with gradient clipping
            q1_loss = F.mse_loss(current_q1, target_q)
            self.q_optimizer1.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), max_grad_norm)
            self.q_optimizer1.step()
            
            q2_loss = F.mse_loss(current_q2, target_q)
            self.q_optimizer2.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), max_grad_norm)
            self.q_optimizer2.step()
            
            # Train policy network
            new_action, log_prob = self.policy.sample(state)
            q1 = self.q_net1(state, new_action)
            q2 = self.q_net2(state, new_action)
            q = torch.min(q1, q2)
            
            policy_loss = (self.alpha * log_prob - q).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
            self.policy_optimizer.step()
            
            # Update alpha (temperature parameter)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # Update target networks with soft update
            for param, target_param in zip(self.q_net1.parameters(), self.q_net1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.q_net2.parameters(), self.q_net2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            # İstatistik tutma
            self.q1_loss_history.append(q1_loss.item())
            self.q2_loss_history.append(q2_loss.item())
            self.policy_loss_history.append(policy_loss.item())
            self.alpha_history.append(self.alpha.item())
            
            # Hesaplanan ortalamaları güncelle
            avg_q1_loss += q1_loss.item()
            avg_q2_loss += q2_loss.item()
            avg_policy_loss += policy_loss.item()
            avg_alpha += self.alpha.item()
                
        # Ortalamaları hesapla ve yazdır
        avg_q1_loss /= iterations
        avg_q2_loss /= iterations
        avg_policy_loss /= iterations
        avg_alpha /= iterations

    
    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add((state, next_state, action, reward, done))

    