#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module that contains definition of a simple PyTorch NN with user-defined number of dense layers.
Will be used to approximate policy and value functions.
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list of int): Number of nodes in hidden layers between input and output
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.hidden = nn.ModuleList()
        layers = [state_size] + fc_units + [action_size]
        for i_d in range(len(layers) - 1):
            self.hidden.append(nn.Linear(layers[i_d], layers[i_d + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i_f, f in enumerate(self.hidden):
            if i_f < len(self.hidden) - 1:
                f.weight.data.uniform_(*hidden_init(f))
            else:
                f.weight.data.uniform_(-3e-3, 3e-3)
            f.bias.data.fill_(1e-1)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        for i_f, f in enumerate(self.hidden):
            x = F.relu(f(x)) if i_f < len(self.hidden) - 1 else f(x)
        return F.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units, cat_action_layer=1):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list of int): Number of nodes in hidden layers between input and output
            cat_action_layer (int): in which hidden layer from 1 to available to map actions into layers.
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.cat_layer = cat_action_layer

        self.hidden = nn.ModuleList()
        layers = [state_size] + fc_units + [1]  # output to one node: value
        for i_d in range(len(layers) - 1):
            self.hidden.append(
                nn.Linear(layers[i_d] + (action_size if i_d == self.cat_layer else 0),
                          layers[i_d + 1]))

        self.reset_parameters()

    def reset_parameters(self):
        for i_f, f in enumerate(self.hidden):
            if i_f < len(self.hidden) - 1:
                f.weight.data.uniform_(*hidden_init(f))
            else:
                f.weight.data.uniform_(-3e-3, 3e-3)
            f.bias.data.fill_(1e-1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = state
        for i_f, f in enumerate(self.hidden):
            if i_f == self.cat_layer:
                x = torch.cat((x, action), dim=1)
            x = F.leaky_relu(f(x)) if i_f < len(self.hidden) - 1 else f(x)
        return x
