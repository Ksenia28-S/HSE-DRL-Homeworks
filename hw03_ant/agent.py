import random
import numpy as np
import os
import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)

class Agent:
    def __init__(self):
        self.model = Actor(28, 8).to(DEVICE)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state]), dtype=torch.float, device=DEVICE)
            return self.model(state).cpu().numpy()[0]

    def reset(self):
        pass

