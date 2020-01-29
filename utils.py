"""
Utility functions/classes for deep RL settings.
"""
import os
import numpy as np
import random
import copy
from collections import deque

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=1e-2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Experience:
    """ Simple class to hold one instance of an experience."""

    def __init__(self, state, action, reward, next_state, priority, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.priority = priority
        self.done = done


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
       Prioritization can be activated.
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, cfg):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            cfg (object): further config settings passed through
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        self.cfg = cfg
        self.prioritized = cfg.prioritized_replay
        self.alpha = cfg.priority_alpha
        self.beta = cfg.priority_beta
        self.eps = cfg.priority_eps
        self.max_prio = 1

    def add(self, state, action, reward, next_state, priority, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, priority, done)
        self.max_prio = max(self.max_prio, priority)
        self.memory.append(e)

    def get_max_priority(self):
        return 1 if len(self.memory) == 0 else self.max_prio

    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        indices = None
        priorities = None
        if not self.prioritized:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            # calculating priorities for prioritized experience replay
            prios = np.asfarray(
                [e.priority for e in self.memory if e is not None]) + self.cfg.priority_eps
            prios = prios ** self.cfg.priority_alpha
            prios /= np.sum(prios)

            indices = np.random.choice(np.arange(len(prios)), size=self.batch_size,
                                       replace=False, p=prios)
            experiences = [self.memory[ie] for ie in indices]
            priorities = prios[indices]

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        if priorities is not None:
            priorities = torch.from_numpy(np.vstack(priorities)).float().to(device)

        return (states, actions, rewards, next_states, priorities, dones, indices)

    def update_prios(self, inds, new_prios):
        """ Update priorities of experiences corresponding to given indices.
            Also updates maximum priority.
        """
        self.max_prio = max(self.max_prio, max(new_prios))
        for i_ind, ind in enumerate(inds):
            self.memory[ind].priority = new_prios[i_ind]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
