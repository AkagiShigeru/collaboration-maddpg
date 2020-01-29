#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module to train/watch an agent in Unity ML navigation environment.
Will import/make use of classes and helpers defined in other Python files within this directory.
"""
import os
import shutil

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sb

import torch

from ddpg_agent import Agent

from unityagents import UnityEnvironment


# activate for dark-mode plots
# sb.set(style="ticks", context="talk")
# plt.style.use("dark_background")


def init_environment(path_to_app):
    env = UnityEnvironment(file_name=path_to_app)
    return env


def plot_scores(scores, cfg):
    scores = pd.Series(scores)
    # plot the scores
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    ax.plot(scores.index, scores.values, ls="-", color="k", label="Scores")
    ma = scores.rolling(20).median()
    ax.plot(ma.index, ma.values, ls="-", color="r", label="Moving median scores (20 episodes)")

    ma = scores.rolling(100).median()
    ax.plot(ma.index, ma.values, ls="-", color="b", label="Moving median scores (100 episodes)")

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(cfg.experiment_path, "scores.pdf"))
    plt.savefig(os.path.join(cfg.experiment_path, "scores.png"))


def ddpg_learning(env, agent, brain_name,
                  n_episodes=2000, max_t=100000,
                  avg_score_cutoff=15,
                  save_path_actor=None, save_path_critic=None):
    """Function to perform Deep Deterministic Policy Gradient learning.

    Params
    ======
        env: environment that the agent interacts with
        agent: agent instance that does all the computation/optimization in the background
        brain_name: name of the Unity brain instance to select correct agent/window
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time-steps per episode
        avg_score_cutoff (float): training will be stopped if avg score over last 100 episodes exceeds this value
        save_path_{actor/critic}: paths to save model weights in
    """
    print("Training an agent with DDPG.")

    scores = []  # list containing scores from each episode

    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name]
        state = state.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores.append(score)  # save most recent score

        last100mean = np.mean(scores[-100:])
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, last100mean), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, last100mean))

        if last100mean >= avg_score_cutoff:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, last100mean))

            break

    # save trained models
    if save_path_actor is not None:
        print("Saving actor network weights to file {:s}".format(save_path_actor))
        torch.save(agent.actor_local.state_dict(), save_path_actor)

    if save_path_critic is not None:
        print("Saving critic network weights to file {:s}".format(save_path_critic))
        torch.save(agent.critic_local.state_dict(), save_path_critic)

    return pd.Series(scores)


def train_or_play(cfg):
    # initialize the environment and obtain state/action sizes and other parameters
    env = init_environment(cfg.app_path)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size, action_size, cfg)

    if cfg.train_model:

        scores = ddpg_learning(env, agent, brain_name,
                               n_episodes=cfg.n_episodes, max_t=cfg.max_t,
                               avg_score_cutoff=cfg.avg_score_cutoff,
                               save_path_actor=cfg.save_path_actor,
                               save_path_critic=cfg.save_path_critic)

        if cfg.save_scores:
            print("Saving scores to file {:s}".format(cfg.save_scores))
            scores.to_hdf(cfg.save_scores, "scores")

        plot_scores(scores, cfg)

    else:  # visualize trained model and scores

        assert os.path.exists(
            cfg.save_path_actor), "Saved model weights need to exist before you can watch a trained agent!"
        assert os.path.exists(
            cfg.save_path_critic), "Saved model weights need to exist before you can watch a trained agent!"

        print("Visualizing the trained agent!")

        env_info = env.reset(train_mode=False)[brain_name]
        agent.actor_local.load_state_dict(torch.load(cfg.save_path_actor))
        agent.critic_local.load_state_dict(torch.load(cfg.save_path_critic))

        score = 0  # initialize the score
        state = env_info.vector_observations[0]
        while True:
            action = agent.act(state, add_noise=False)  # take step without noise
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break

        if os.path.exists(cfg.save_scores):
            plot_scores(pd.read_hdf(cfg.save_scores, "scores"), cfg)

    env.close()


if __name__ == "__main__":
    import argparse
    import importlib.machinery

    parser = argparse.ArgumentParser(
        "Python script to train/watch an agent solve an environment based on Unity ML-agents.")
    parser.add_argument("fn", nargs=1, metavar="<config file>",
                        help="Configuration file containing desired paths and settings.")
    args = parser.parse_args()

    cfgfn = args.fn[0]
    cfg = importlib.machinery.SourceFileLoader("cfg", cfgfn).load_module()

    # creating path to store results, if not existing
    if not os.path.exists(cfg.experiment_path):
        os.makedirs(cfg.experiment_path)
        print("Created new directory {0}.".format(cfg.experiment_path))

    # copy cfg file to log exact settings used for each run
    if cfg.train_model:
        shutil.copyfile(cfgfn, os.path.join(cfg.experiment_path, os.path.split(cfgfn)[1]))

    train_or_play(cfg)
