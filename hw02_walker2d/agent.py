import random
import numpy as np
import os
import torch

class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action, action_sample, dist = self.model.act(state)
        return action.numpy()[0] # TODO

    def reset(self):
        pass

