"""
Core implementations of DDPG agents, both single and multiple-agent scenarios.
"""
import os
import numpy as np
import random

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import ReplayBuffer, OUNoise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiDDPGAgent():
    """ Multi-agent DDPG implementation."""

    def __init__(self, state_size, action_size, num_agents, cfg):
        """Initialize a MADDPG Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): Number of agents in environment
            cfg (config object): main configuration with other settings
        """
        print("Initializing MADDPG agent!")

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(cfg.random_seed)

        self.cfg = cfg

        # initializing list of single agents (2 for tennis)
        self.agents = []
        for aid in range(num_agents):
            agent = SingleDDPGAgent(state_size, action_size, cfg,
                                    num_agents=num_agents, agent_id=aid)
            self.agents.append(agent)

        self.t_step = 0

        # Noise process
        self.noise_scale = self.cfg.noise_scale
        self.noise = OUNoise(action_size, cfg.random_seed,
                             theta=cfg.theta_ou, sigma=cfg.sigma_ou)

        # as long as active, will fill replay buffer with random memories, no learning
        self.prefetching = True

        # Replay memory for shared experiences (all agents)
        self.memory = ReplayBuffer(action_size, cfg.buffer_size, cfg.batch_size, cfg.random_seed,
                                   cfg)

    def add_noise(self):
        if self.cfg.use_ou:
            return self.noise_scale * self.noise.sample()
        else:
            # Gaussian noise
            return self.noise_scale * np.random.normal(0, 1.0, self.action_size)

    def reset(self):
        self.t_step = 0
        self.noise.reset()

    def act(self, state_all):
        """
        Let all agents act.
        Receives full state tensor of all agents
        and outputs all actions (num_agents x action_size).
        """
        actions = []
        for aid in range(self.num_agents):
            # only add noise after pre-loading memories
            noise = 0
            if not self.prefetching:
                noise = self.add_noise()
            actions.append(self.agents[aid].act(state_all[aid],
                                                add_noise=False) + noise)

        return actions

    def _target_act(self, states_all):
        """
        Internal function used by learn function.
        Gets target network actions for all agents.
        """
        target_actions = []
        for aid in range(self.num_agents):
            # TODO probably incorrect shape here...
            target_actions.append(self.agents[aid]._target_act(states_all[aid]))

        return target_actions

    def step(self, states, actions, rewards, next_states, dones):
        """ Save experiences in global memory.
            If memory large enough, use it to learn each agent.
        """
        max_prio = self.memory.get_max_priority()
        self.memory.add(states, actions, rewards,
                        next_states, max_prio, dones)

        # start training if memory size large enough.
        if len(self.agents[0].memory) >= max(self.cfg.batch_size, self.cfg.init_replay):
            if self.prefetching:
                self.prefetching = False
                print("Pre-loading memories complete, starting training!")
        else:
            return

        self.t_step = (self.t_step + 1) % self.cfg.learn_every
        if self.t_step == 0:
            for _ in range(self.cfg.learn_steps):
                self.learn_all()

        self.noise_scale = max(self.noise_scale * self.cfg.noise_decay, self.cfg.noise_min)
        self.t_step += 1

    def learn_all(self):
        """Generates full batch input and performs individual learning steps."""
        samples = self.memory.sample()
        for aid in range(self.num_agents):
            self.learn(samples, aid)

    def learn(self, samples, agent_number):
        """
            Update critic and actor networks of given agent using provided
            samples from replay memory.
        """

        states, actions, rewards, next_states, priorities, dones, indices = samples

        from IPython import embed
        embed()

        batch_size = full_state.shape[0]

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network

        # change the shape to batch_size X num_agents X remaining.
        # by changing the shape we can select local observations by agents by selecting [:, i, :] for i in range(num_agents)

        target_actions = self.target_act(next_state.view(batch_size, self.num_agents, -1))

        # turn a list of 2x2 into a batch_size x (action_size * num_agent)
        target_actions = torch.cat(target_actions, dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(full_next_state, target_actions.to(device))

        # shape of reward is batch_size X num_agents so [:, agent_number] is needed to pick the rewards for the specific agent
        # that is being updated.
        y = reward[:, agent_number].view(-1, 1) + self.discount_factor * q_next * (
                1 - done[:, agent_number].view(-1, 1))

        q = agent.critic(full_state, action.view(batch_size, -1))

        critic_loss = F.mse_loss(q, y.detach())

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [
            self.maddpg_agent[i].actor(
                state.view([batch_size, self.num_agents, -1])[:, i, :]) if i == agent_number else
            self.maddpg_agent[i].actor(
                state.view([batch_size, self.num_agents, -1])[:, i, :]).detach() for i in
            range(self.num_agents)]

        q_input = torch.cat(q_input, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already

        # get the policy gradient
        actor_loss = -agent.critic(full_state, q_input).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1)
        agent.actor_optimizer.step()

        # soft update the models
        # having update in here as well makes the model converge faster
        # I could increase the trasnfer rate as well. instead of having updates called twice.
        agent.soft_update(agent.critic_local, agent.critic_target, self.cfg.tau)
        agent.soft_update(agent.actor_local, agent.actor_target, self.cfg.tau)

    def soft_update_all(self):
        """soft update targets"""
        for agent in self.agents:
            agent.soft_update(agent.critic_local, agent.critic_target, self.cfg.tau)
            agent.soft_update(agent.actor_local, agent.actor_target, self.cfg.tau)

    def save_weights(self, model_save_path):
        """
        Simple method to save network weights.
        """
        for aid, agent in enumerate(self.agents):
            agent.save_weights(model_save_path, suffix="_{:d}".format(aid))

    def load_weights(self, model_save_path):
        """
        Method to load network weights from saved files.
        """
        for aid, agent in enumerate(self.agents):
            agent.load_weights(model_save_path, suffix="_{:d}".format(aid))


class SingleDDPGAgent():
    """
        Single agent DDPG.
        Interacts with and learns from the environment.
    """

    def __init__(self, state_size, action_size, cfg, num_agents=1, agent_id=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            cfg (config object): main configuration with other passed settings
            num_agents (int): optional (default: 1). If >1 will multiply state and action
                            space sizes for critic. Used for usage with MADDPG.
            agent_id (int): optional (default: 0). Set agent id for MADDPG.
        """
        print("Initializing single DDPG agent!")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(cfg.random_seed)
        self.n_agents = num_agents
        self.agent_id = agent_id

        self.cfg = cfg

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, cfg.random_seed,
                                 cfg.dense_layers_actor).to(device)
        self.actor_target = Actor(state_size, action_size, cfg.random_seed,
                                  cfg.dense_layers_actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=cfg.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size * num_agents, action_size * num_agents,
                                   cfg.random_seed,
                                   cfg.dense_layers_critic).to(device)
        self.critic_target = Critic(state_size * num_agents, action_size * num_agents,
                                    cfg.random_seed,
                                    cfg.dense_layers_critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=cfg.lr_critic,
                                           weight_decay=cfg.weight_decay)

        self.hard_copy_weights(self.critic_local, self.critic_target)
        self.hard_copy_weights(self.actor_local, self.actor_target)

        self.t_step = 0

        # Noise process
        self.noise = OUNoise(action_size, cfg.random_seed,
                             theta=cfg.theta_ou, sigma=cfg.sigma_ou)

        # Replay memory
        self.memory = ReplayBuffer(action_size, cfg.buffer_size, cfg.batch_size, cfg.random_seed,
                                   cfg)

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

    def _target_act(self, state):
        """ Let target network return action."""
        state = torch.from_numpy(state).float().to(device)

        with torch.no_grad():
            action_target = self.actor_target(state).cpu().data.numpy()

        return action_target

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
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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

    def hard_copy_weights(self, local_model, target_model):
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

    def save_weights(self, model_save_path, suffix=""):
        """
        Simple method to save network weights.
        """
        # actors
        torch.save(self.actor_local.state_dict(),
                   os.path.join(model_save_path, "weights_actor_local{:s}.pth".format(suffix)))
        torch.save(self.actor_target.state_dict(),
                   os.path.join(model_save_path, "weights_actor_target{:s}.pth".format(suffix)))
        # critics
        torch.save(self.critic_local.state_dict(),
                   os.path.join(model_save_path, "weights_critic_local{:s}.pth".format(suffix)))
        torch.save(self.critic_target.state_dict(),
                   os.path.join(model_save_path, "weights_critic_target{:s}.pth".format(suffix)))

    def load_weights(self, model_save_path, suffix=""):
        """
        Method to load network weights from saved files.
        """
        self.actor_local.load_state_dict(torch.load(
            os.path.join(model_save_path, "weights_actor_local{:s}.pth".format(suffix))))
        self.actor_target.load_state_dict(torch.load(
            os.path.join(model_save_path, "weights_actor_target{:s}.pth".format(suffix))))

        self.critic_local.load_state_dict(torch.load(
            os.path.join(model_save_path, "weights_critic_local{:s}.pth".format(suffix))))
        self.critic_target.load_state_dict(torch.load(
            os.path.join(model_save_path, "weights_critic_target{:s}.pth".format(suffix))))