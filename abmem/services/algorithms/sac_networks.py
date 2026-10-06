import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# İlklendirme fonksiyonu - ortogonal ilklendirme
def init_weights(layer, gain=1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        nn.init.constant_(layer.bias.data, 0)
    return layer

# Layer Normalization ekleyen basit bir blok
class NormalizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim, use_ln=True):
        super(NormalizedLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(out_dim)
        
        # Ortogonal ilklendirme ile ağırlıkları ayarla
        init_weights(self.linear)
    
    def forward(self, x):
        x = self.linear(x)
        if self.use_ln:
            x = self.ln(x)
        return x

# SAC için Q-Network - Layer Normalization ile
class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SoftQNetwork, self).__init__()
        
        # Daha büyük hidden_dim (128->256) ve layer normalization
        self.net = nn.Sequential(
            NormalizedLinear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            NormalizedLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NormalizedLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NormalizedLinear(hidden_dim // 2, 1, use_ln=False)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# SAC için Politika Ağı (Gaussian Policy) - Layer Normalization ile
class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_limits, hidden_dim=256, 
                 log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(SACPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        self.action_space_limits = action_space_limits
        self.epsilon = epsilon
        
        # Daha büyük hidden_dim ve layer normalization
        self.shared = nn.Sequential(
            NormalizedLinear(state_dim, hidden_dim),
            nn.ReLU(),
            NormalizedLinear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Ortalama değer için katman
        self.mean_net = nn.Sequential(
            NormalizedLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NormalizedLinear(hidden_dim // 2, action_dim, use_ln=False)
        )
        
        # Standart sapma için katman
        self.log_std_net = nn.Sequential(
            NormalizedLinear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            NormalizedLinear(hidden_dim // 2, action_dim, use_ln=False)
        )
        
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_net(x)
        log_std = self.log_std_net(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            # Deterministik mod için direkt olarak ortalama değeri kullan
            tanh_mean = torch.tanh(mean)
            action = self._scale_action(tanh_mean)
            log_prob = None
            return action, log_prob
        
        # Normal dağılım oluştur
        normal = Normal(mean, std + self.epsilon)  # Numerical stability için epsilon eklendi
        
        # Reparameterization trick kullanarak örnek al
        x = normal.rsample()
        
        # Tanh squashing
        y = torch.tanh(x)
        
        # Aksiyon aralığına scale et
        action = self._scale_action(y)
        
        # Log olasılık hesabı (Enforced Action Bounds)
        log_prob = normal.log_prob(x) - torch.log(1 - y.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def _scale_action(self, action_normalized):
        """
        Tanh ile [-1, 1] aralığına sıkıştırılmış aksiyonları, 
        gerçek aksiyon aralığına dönüştürür
        """
        # Ayrı aksiyon limitleri için vektör dönüştürme
        lower_bound = torch.tensor([limit[0] for limit in self.action_space_limits], device=action_normalized.device)
        upper_bound = torch.tensor([limit[1] for limit in self.action_space_limits], device=action_normalized.device)
        
        # [-1, 1] aralığından aksiyon aralığına dönüştürme
        action = lower_bound + (action_normalized + 1.0) * 0.5 * (upper_bound - lower_bound)
        
        # Aksiyon sınırlarında clipping
        return torch.clamp(action, lower_bound, upper_bound) 