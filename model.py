import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)
        # Only actor needs log_std
        if action_dim > 1:
            self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        else:
            self.log_std = None

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out