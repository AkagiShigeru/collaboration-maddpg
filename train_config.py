#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple config file to be used for navigation project.
"""
import os
from datetime import datetime

### paths
app_path = os.path.join("..", "Tennis.app")
result_path_general = os.path.join("experiments")
experiment_path = os.path.join(result_path_general, datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
model_save_path = os.path.join(experiment_path, "model_weights")  # directory to store weights in
save_scores = os.path.join(experiment_path, "training_scores.hdf")
train_model = True  # if False, will show trained agent, if True: will train new model and save weights in above path
maddpg = True  # will perform multi-agent ddpg with shared critic, otherwise single agent ddpg (bad performance)
graphic_mode = True  # if True, will show unity window, if False will not show window
eps_to_watch = 5  # if watching trained agents in action, how many episodes to watch

### general options
random_seed = 1337  # random seed

### qnetwork options

# define number and dimension of dense layers
# input size of first layer and output of last are given by state and action space sizes, respectively.
dense_layers_actor = [256, 128, 128, 64]
dense_layers_critic = [256, 128, 128, 64]

### agent options
buffer_size = int(1e6)  # replay buffer size
batch_size = 256  # mini-batch size
gamma = 0.95  # discount factor
tau = 1e-2  # for soft update of target parameters
lr_actor = 1e-4  # actor learning rate
lr_critic = 1e-3  # critic learning rate
loss_l = 2  # Lx norm to use for critic loss, options: 1 or 2
weight_decay = 0  # L2 weight decay
learn_every = 1  # learn every update_every steps of environment
learn_steps = 1  # how many learning steps per environment step

# noise
use_ou = True  # use Ornstein-Uhlenbeck noise, otherwise Gaussian noise
sigma_ou = 0.2
theta_ou = 0.1
noise_scale = 1.
noise_decay = 0.9995
noise_min = 1e-4

# parameters for prioritized experience replay
init_replay = 1e4
prioritized_replay = False
priority_alpha = 0.6  # priority weighting
priority_beta = 0.4  # for sampling weight computation
priority_eps = 1e-5  # epsilon to add to priorities to avoid zeros

### training options
n_episodes = 20000  # max. number of episodes (if not done before)
max_t = 10000  # max time per episode, intentionally set high to let environment cut-off
avg_score_cutoff = 1.0  # stop training when score (last 100 episodes) exceeds value
