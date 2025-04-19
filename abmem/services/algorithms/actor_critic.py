import torch
import torch.nn as nn
import torch.nn.functional as F


# Decision-making actor network.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,  action_space_limits, noise_std=0.05,noise_decay=0.99,noise_min=0.01):
        super(Actor, self).__init__()

        self.noise_scale = noise_std
        # Decay of noise every step
        self.noise_decay = noise_decay
        self.noise_min = noise_min

        # Network layers: input state_dim, output action_dim
        # action_space_limits is needed to scale the 0-1 sigmoid output
        self.layer_1 = nn.Linear(state_dim, 128)
        self.layer_2 = nn.Linear(128,64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, action_dim)
        
        self.action_space_limits = action_space_limits

    def forward(self, x):

        # First, normalize the input (state) (sigmoid gives exact value outside the -6/+6 range)
        x = (x - x.mean()) / x.std()
        # Pass the input through the network
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        # Get output between 0-1 with sigmoid
        x = F.sigmoid(self.layer_4(x))

        # Add noise to the output for exploration
        noise = torch.normal(mean=0, std=self.noise_scale, size=x.size())
        x = x + noise

        # Separate action limits according to the vector
        lower_bound = torch.tensor([0 for limit in self.action_space_limits], device=x.device)
        upper_bound = torch.tensor([limit[1] for limit in self.action_space_limits], device=x.device)

        # Scale the noise-added output according to action limits
        x = lower_bound + x * (upper_bound - lower_bound)

        # Clip the actions to be within the specified bounds
        x = torch.clamp(x, lower_bound, upper_bound)

        
        return x
    

    def update_noise(self):
        # Decrease noise level
        self.noise_scale *= self.noise_decay
        # To limit lowest noise
        self.noise_scale = max(self.noise_scale, self.noise_min)
        



# Network that evaluates decisions
class Critic(nn.Module):
    def __init__(self, state_dim,action_dim):
        # Network layers: input state_dim + action_dim, output 1
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        self.layer_4 = nn.Linear(32, 1)

        

    def forward(self, x, u):
        x = F.relu(self.layer_1(torch.cat([x, u], 1)))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_4(x)
