import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        # one fully connected hidden layer with 30 neurons
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        # returns are Q(state, action) values
        # input_state -> fc1 -> relu -> fc2 -> output_action
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


class ReplayMemory(object):

    def __init__(self, capacity):
        # sample a subset from the capacity list for agent to learn instead of all
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        # event: [s_t, s_t+1, a_t, r_t]
        self.memory.append(event)
        # cut off if memory list exceeds capacity
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # zip samples in [(s1, s2, ...), (a1, a2, ...)]
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn(object):
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # size 1x5
        self.last_action = 0
        self.last_reward = 0

    def select_action(self,  state):
        # softmax([1, 2, 3]) => (0.11, 0.15, 0.74) | softmax([1, 2, 3]*3) => (0.0, 0.1, 0.9)
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)  # Temperature = 100
        action = probs.multinomial()
        return action.data([0, 0])

