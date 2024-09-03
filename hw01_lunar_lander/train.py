from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque
import random
import copy

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 600000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 5e-4

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
        
class Memory:
  
    def __init__(self):
        self.maxlen = 10**4
        self.a, self.inext = 0, 0
        self.state, self.action, self.next_state, self.reward, self.done   = None, None, None, None, None

    def append(self, transition):
        state, action, next_state, reward, done = transition
        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        if self.state is None:
            self.state = torch.empty([self.maxlen] + list(state.shape), dtype=torch.float32)
            self.action = torch.empty(self.maxlen, dtype=torch.long)
            self.next_state = torch.empty([self.maxlen] + list(state.shape), dtype=torch.float32)
            self.reward = torch.empty(self.maxlen, dtype=torch.float32)
            self.done = torch.empty(self.maxlen, dtype=torch.long)
        self.state[self.inext] = state
        self.action[self.inext] = action
        self.next_state[self.inext] = next_state
        self.reward[self.inext] = reward
        self.done[self.inext] = done

        self.inext = (self.inext + 1) % self.maxlen
        self.a = min(self.maxlen, self.a + 1)

    def get_batch(self):
        ids = torch.randperm(self.a)[:BATCH_SIZE]
        return (self.state[ids], self.action[ids].view(-1, 1), self.next_state[ids], self.reward[ids].view(-1, 1), self.done[ids].view(-1, 1))
    

class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0 # Do not change
        self.buffer = Memory()
        self.model = DQN_Model(state_dim, action_dim)
        self.target_model = DQN_Model(state_dim, action_dim)
        self.target_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr = LEARNING_RATE)
        self.criterion = nn.MSELoss()


    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.buffer.append(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        return self.buffer.get_batch()

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state_batch, action_batch, new_state_batch, reward_batch, done_batch = batch
        with torch.no_grad():
            model_target_next = self.target_model(new_state_batch).max(1)[0].unsqueeze(1)
        model_target = reward_batch + GAMMA * model_target_next * (1 - done_batch)
        model_expected = self.model(state_batch).gather(1, action_batch)
        self.optimizer.zero_grad()
        loss = self.criterion(model_expected, model_target)
        loss.backward()
        self.optimizer.step()
        

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(self.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            action = np.argmax(self.model(state).numpy())
        self.model.train()
        return action

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")
        
        
def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, *_ = env.step(agent.act(state))
            total_reward += reward
        returns.append(total_reward)
    return returns
    
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    set_seed(42)
    env = make("LunarLander-v2")
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    state = env.reset()
    
    for _ in range(INITIAL_STEPS):
        action = env.action_space.sample()
    
        next_state, reward, done, *_ = env.step(action)
        dqn.consume_transition((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()

        
    for i in range(TRANSITIONS):
        #Epsilon-greedy policy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = dqn.act(state)

        next_state, reward, done, *_ = env.step(action)
        dqn.update((state, action, next_state, reward, done))
        
        state = next_state if not done else env.reset()
        
        if (i + 1) % (TRANSITIONS//100) == 0:
            rewards = evaluate_policy(dqn, 5)
            dqn.save()    
