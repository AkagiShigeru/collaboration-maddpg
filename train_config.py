#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple config file to be used for navigation project.
"""
import os
from datetime import datetime

### paths
app_path = os.path.join("..", "Reacher.app")
result_path_general = os.path.join("experiments")
experiment_path = os.path.join(result_path_general, datetime.utcnow().strftime("%Y%m%d-%H%M%S"))
save_path_actor = os.path.join(experiment_path, "model_weights_actor.pth")
save_path_critic = os.path.join(experiment_path, "model_weights_critic.pth")
save_scores = os.path.join(experiment_path, "training_scores.hdf")
train_model = True  # if False, will show trained agent, if True: will train new model and save weights in above path

### general options
random_seed = 1337  # random seed

### qnetwork options

# define number and dimension of dense layers
# input size of first layer and output of last are given by state and action space sizes, respectively.
dense_layers_actor = [128, 128]
dense_layers_critic = [128, 128]

### agent options
buffer_size = int(1e5)  # replay buffer size
batch_size = 256  # mini-batch size
gamma = 0.90  # discount factor
tau = 1e-3  # for soft update of target parameters
lr_actor = 1e-3  # actor learning rate
lr_critic = 1e-3  # critic learning rate
weight_decay = 0  # L2 weight decay
update_every = 1  # target network is updated each update_every steps

# noise
sigma_ou = 1e-3
theta_ou = 0.

# parameters for prioritized experience replay
prioritized_replay = False
priority_alpha = 0.6  # priority weighting
priority_beta = 0.4  # for sampling weight computation
priority_eps = 1e-5  # epsilon to add to priorities to avoid zeros

### training options
n_episodes = 5000  # max. number of episodes (if not done before)
max_t = 1000  # max time per episode, intentionally set high to let environment cut-off
# eps_start = 1.0  # start epsilon for epsilon-greedy
# eps_end = 0.01  # end epsilon
# eps_decay = 0.99  # exponential decay
avg_score_cutoff = 31  # stop training when score (last 100 episodes) exceeds value
