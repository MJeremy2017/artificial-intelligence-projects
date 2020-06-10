import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from doom import experience_replay, image_preprocessing
from torch.autograd import Variable
import gym


class CNN(nn.Module):
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_nn((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(40, number_actions)

    def count_nn(self, image_dim):
        # create fake random get data to get size after convolution layer
        # take input as a variable
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # view == reshape
        x = x.view(x.size(0), -1)  # x.size is the batch size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SoftmaxBody(nn.Module):
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T

    def forward(self, output):
        # select action from CNN's output (Variable)
        probs = F.softmax(output * self.T)
        actions = probs.multinomial()
        return actions


# ensembles brain and body
class AI:
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body

    def __call__(self, inputs):
        # The __call__ method enables Python programmers to write classes
        # where the instances behave like functions and can be called like a function.
        x = Variable(torch.from_numpy(np.array(inputs, dtype=np.float32)))
        outputs = self.brain(x)
        actions = self.body(outputs)
        return actions.data.numpy()


# Getting the Doom environment
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width=80, height=80, grayscale=True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force=True)
number_actions = doom_env.action_space.n

# Initialise Agent
cnn = CNN(number_actions)
body = SoftmaxBody(1.0)
ai = AI(cnn, body)

# Setting up Experience Replay
n_steps = experience_replay.NStepProgress(env=doom_env, ai=ai, n_step=10)
memory = experience_replay.ReplayMemory(n_steps=n_steps, capacity=10000)


# n-step sarsa
# batch: n_batch x n_steps
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        # take the first and last state from the n-steps
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype=np.float32)))
        output = cnn(input)
        # calculate total reward: R_total = R_t + gamma^1 * R_t+1 + gamma^2 * R_t+2 + ... + gamma^n * max(Q(s_T, a_T))
        cummul_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cummul_reward = step.reward + gamma * cummul_reward
        # target equals the accumulative reward of the specific action
        target = output[0].data
        target[series[0].action] = cummul_reward

        targets.append(target)
        inputs.append(series[0].state)
    # returns batches of [input of state] and [accumulative reward of the state]
    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


