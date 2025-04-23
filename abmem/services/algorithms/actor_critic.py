import torch
import torch.nn as nn
import torch.nn.functional as F

# Decision-making actor network.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,  action_space_limits, noise_std=0.05, noise_decay=0.99, noise_min=0.01):
        super(Actor, self).__init__()

        self.noise_scale = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min

        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, action_dim)

        self.action_space_limits = action_space_limits

    def forward(self, x):
        x = (x - x.mean()) / (x.std() + 1e-6)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = torch.sigmoid(self.layer_4(x))

        noise = torch.normal(mean=0, std=self.noise_scale, size=x.size()).to(x.device)
        x = x + noise

        lower_bound = torch.tensor([0 for limit in self.action_space_limits], device=x.device)
        upper_bound = torch.tensor([limit[1] for limit in self.action_space_limits], device=x.device)
        x = lower_bound + x * (upper_bound - lower_bound)
        x = torch.clamp(x, lower_bound, upper_bound)

        return x

    def update_noise(self):
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, self.noise_min)

# Düzenlenmiş Critic sınıfı
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 128)  # input_dim = 32 (LSTM out) + action_dim
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_4(x)


# RNN tabanlı actor
class RNNActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_space_limits, noise_std=0.05, noise_decay=0.99, noise_min=0.01):
        super(RNNActor, self).__init__()

        self.noise_scale = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        self.action_space_limits = action_space_limits

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=32, batch_first=True)

        # LSTM çıkışından sonra dense katmanlar (32 → ...)
        self.layer_1 = nn.Linear(32, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, action_dim)

    def forward(self, x, hidden):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x, hidden = self.lstm(x, hidden)
        x = x[:, -1, :]  # son adım

        x = (x - x.mean()) / (x.std() + 1e-6)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = torch.sigmoid(self.layer_4(x))
        #print(f"[Actor] Sigmoid output (before noise): {x.detach().cpu().numpy()}")

        noise = torch.normal(mean=0, std=self.noise_scale, size=x.size()).to(x.device)
        x = x + noise
        #print(f"[Actor] Output + noise: {x.detach().cpu().numpy()}")

        lower_bound = torch.tensor([0 for limit in self.action_space_limits], device=x.device)
        upper_bound = torch.tensor([limit[1] for limit in self.action_space_limits], device=x.device)
        x = lower_bound + x * (upper_bound - lower_bound)
        x = torch.clamp(x, lower_bound, upper_bound)

        return x.unsqueeze(1), hidden

    def update_noise(self):
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, self.noise_min)


# Düzenlenmiş RNNCritic sınıfı
class RNNCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim + action_dim, hidden_size=32, batch_first=True)
        self.critic = Critic(input_dim=32 + action_dim)

    def forward(self, x, u, hidden):
        if u.size(1) == 1 and u.size(1) != x.size(1):
            u = u.repeat(1, x.size(1), 1)

        xu = torch.cat([x, u], dim=-1)
        x, _ = self.lstm(xu, hidden)
        # x: [batch, seq, 32], u: [batch, seq, action_dim]
        x = x[:, -1, :]             # [batch, 32]
        u_last = u[:, -1, :]       # [batch, action_dim]
        return self.critic(torch.cat([x, u_last], dim=1))  # [batch, 32+action_dim]


