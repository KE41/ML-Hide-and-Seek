"""Neural Network controlling humanoid robot movement"""

import torch
import torch.nn as nn
import torch.distributions as D

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        return mean, std

    def sample(self, x):
        mean, std = self.forward(x)
        dist = D.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    