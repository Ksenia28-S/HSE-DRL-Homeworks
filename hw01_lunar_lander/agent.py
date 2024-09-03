import random
import numpy as np
import torch
from torch import nn

class DQN_Model(torch.nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.linear1 = nn.Linear(state_dim, 64)
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, action_dim)

    def forward(self, state):
        h = self.activation(self.linear1(state))
        h = self.activation(self.linear2(h))
        output = self.linear3(h)
        return output

class Agent:
    def __init__(self):
        self.model = DQN_Model(8, 4)
        results = torch.load(__file__[:-8] + "/agent.pkl")
        self.model.load_state_dict(results)
        self.model.eval()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action = np.argmax(self.model(state).numpy())
        return action
        