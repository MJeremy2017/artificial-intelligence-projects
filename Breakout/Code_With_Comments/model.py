import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))  # thanks to this initialization, we have var(out) = std^2
    return out


# Initializing the weights of the neural network in an optimal way for the learning
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())  # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4])  # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]  # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out))  # weight bound
        # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.weight.data.uniform_(-w_bound, w_bound)
        # initializing all the bias with zeros
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]  # dim1
        fan_out = weight_shape[0]  # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        # input size is 1 image
        self.lstm = nn.LSTMCell(32*3*3, 256)  # 32x3x3 is the size after convolution layer
        num_outputs = action_space.n
        self.linear_critic = nn.Linear(256, 1)  # V(s)
        self.linear_actor = nn.Linear(256, num_outputs)  # Q(s,a)
        # init weights
        self.apply(weights_init)

        self.linear_critic.weight.data = normalized_columns_initializer(self.linear_critic.weight.data, 1.)
        self.linear_critic.bias.fill_(0)

        self.linear_actor.weight.data = normalized_columns_initializer(self.linear_actor.weight.data, .01)
        self.linear_actor.bias.fill_(0)
        # fill lstm bias
        self.lstm.bias_hh.fill_(0)
        self.lstm.bias_ih.fill_(0)
        # training mode
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*3*3)
        (hx, cx) = self.lstm(x, (hx, cx))
        x = hx
        return self.linear_critic(x), self.linear_actor(x), (hx, cx)




















# # AI for Breakout
#
# # Importing the librairies
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # Initializing and setting the variance of a tensor of weights
# def normalized_columns_initializer(weights, std=1.0):
#     out = torch.randn(weights.size())
#     out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
#     return out
#
# # Initializing the weights of the neural network in an optimal way for the learning
# def weights_init(m):
#     classname = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
#     if classname.find('Conv') != -1: # if the connection is a convolution
#         weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
#         fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
#         fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
#         w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
#         m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
#         m.bias.data.fill_(0) # initializing all the bias with zeros
#     elif classname.find('Linear') != -1: # if the connection is a full connection
#         weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
#         fan_in = weight_shape[1] # dim1
#         fan_out = weight_shape[0] # dim0
#         w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
#         m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
#         m.bias.data.fill_(0) # initializing all the bias with zeros
#
# # Making the A3C brain
#
# class ActorCritic(torch.nn.Module):
#
#     def __init__(self, num_inputs, action_space):
#         super(ActorCritic, self).__init__()
#         self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1) # first convolution
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # second convolution
#         self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # third convolution
#         self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1) # fourth convolution
#         self.lstm = nn.LSTMCell(32 * 3 * 3, 256) # making an LSTM (Long Short Term Memory) to learn the temporal properties of the input - we obtain a big encoded vector S of size 256 that encodes an event of the game
#         num_outputs = action_space.n # getting the number of possible actions
#         self.critic_linear = nn.Linear(256, 1) # full connection of the critic: output = V(S)
#         self.actor_linear = nn.Linear(256, num_outputs) # full connection of the actor: output = Q(S,A)
#         self.apply(weights_init) # initilizing the weights of the model with random weights
#         self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
#         self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros
#         self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
#         self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros
#         self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
#         self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros
#         self.train() # setting the module in "train" mode to activate the dropouts and batchnorms
#
#     def forward(self, inputs):
#         inputs, (hx, cx) = inputs # getting separately the input images to the tuple (hidden states, cell states)
#         x = F.elu(self.conv1(inputs)) # forward propagating the signal from the input images to the 1st convolutional layer
#         x = F.elu(self.conv2(x)) # forward propagating the signal from the 1st convolutional layer to the 2nd convolutional layer
#         x = F.elu(self.conv3(x)) # forward propagating the signal from the 2nd convolutional layer to the 3rd convolutional layer
#         x = F.elu(self.conv4(x)) # forward propagating the signal from the 3rd convolutional layer to the 4th convolutional layer
#         x = x.view(-1, 32 * 3 * 3) # flattening the last convolutional layer into this 1D vector x
#         hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
#         x = hx # getting the useful output, which are the hidden states (principle of the LSTM)
#         return self.critic_linear(x), self.actor_linear(x), (hx, cx) # returning the output of the critic (V(S)), the output of the actor (Q(S,A)), and the new hidden & cell states ((hx, cx))
