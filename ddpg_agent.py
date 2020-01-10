import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, cfg):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        print("Initializing DDPG agent!")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(cfg.random_seed)

        self.cfg = cfg

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, cfg.random_seed,
                                 cfg.dense_layers_actor).to(device)
        self.actor_target = Actor(state_size, action_size, cfg.random_seed,
                                  cfg.dense_layers_actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=cfg.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, cfg.random_seed,
                                   cfg.dense_layers_critic).to(device)
        self.critic_target = Critic(state_size, action_size, cfg.random_seed,
                                    cfg.dense_layers_critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=cfg.lr_critic,
                                           weight_decay=cfg.weight_decay)

        self.copy_weights(self.critic_local, self.critic_target)
        self.copy_weights(self.actor_local, self.actor_target)

        self.t_step = 0

        # Noise process
        self.noise = OUNoise(action_size, cfg.random_seed,
                             theta=cfg.theta_ou, sigma=cfg.sigma_ou)

        # Replay memory
        self.memory = ReplayBuffer(action_size, cfg.buffer_size, cfg.batch_size, cfg.random_seed, cfg)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        max_prio = self.memory.get_max_priority()
        self.memory.add(state, action, reward, next_state, max_prio, done)

        # Learn, if enough samples are available in memory
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.cfg.update_every
        if self.t_step == 0:
            if len(self.memory) > self.cfg.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.cfg.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.t_step = 0
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', prio, done, indices) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, priorities, dones, indices = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if self.cfg.prioritized_replay:
            weights = 1. / ((self.cfg.batch_size * priorities) ** self.cfg.priority_beta)
            weights /= max(weights)
            # calculating new transition priorities based on residuals between target and local network predictions
            diffs = Q_targets - Q_expected  # TD-error
            diffs = np.abs(np.squeeze(diffs.tolist()))
            self.memory.update_prios(indices, diffs)
            # bias-annealing weights
            Q_expected *= weights
            Q_targets *= weights

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.cfg.tau)
        self.soft_update(self.actor_local, self.actor_target, self.cfg.tau)

    def copy_weights(self, local_model, target_model):
        """Update model parameters.

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.1, sigma=1e-4):
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
            prios = np.asfarray([e.priority for e in self.memory if e is not None]) + self.cfg.priority_eps
            prios = prios ** self.cfg.priority_alpha
            prios /= np.sum(prios)

            indices = np.random.choice(np.arange(len(prios)), size=self.batch_size,
                                       replace=False, p=prios)
            experiences = [self.memory[ie] for ie in indices]
            priorities = prios[indices]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
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
