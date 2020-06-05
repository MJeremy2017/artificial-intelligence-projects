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
        self.fc2 = nn.Linear(30, 50)
        self.fc3 = nn.Linear(50, nb_action)

    def forward(self, state):
        # returns are Q(state, action) values
        # input_state -> fc1 -> relu -> fc2 -> output_action
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
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
        probs = F.softmax(self.model(Variable(state, volatile=True)) * 100)  # Temperature = 100 | 0 to deactivate
        # randomly select an action
        action = probs.multinomial()
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # https://stackoverflow.com/questions/50999977/what-does-the-gather-function-do-in-pytorch-in-layman-terms
        # only pickup action that is acted
        # 5 x 100 => model => 3 x 100 => gather => 100 x 1 => squeeze => 100,
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  # Q(s, a)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  # max[Q(s', a')]
        target = self.gamma*next_outputs + batch_reward
        loss = F.smooth_l1_loss(outputs, target)
        # initialize optimizer
        self.optimizer.zero_grad()
        loss.backward(retain_variables=True)
        # update weights
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # long tensor only accept int type
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # update dqn
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)

        action = self.select_action(new_state)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(1. + len(self.reward_window))

    # save model in python dict
    def save(self):
        # state_dict: layer -> parameters
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, 'last_brain.pth')

    def load(self):
        file_name = 'last_brain.pth'
        if os.path.isfile(file_name):
            print("loading model ...")
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done!")
        else:
            print("file does not exist!")






