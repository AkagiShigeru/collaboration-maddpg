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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sb

from ddpg_agents import SingleDDPGAgent, MultiDDPGAgent

from unityagents import UnityEnvironment


def ActivateDarkMode():
    # activate for dark-mode plots
    sb.set(style="ticks", context="talk")
    plt.style.use("dark_background")


def init_environment(path_to_app, graphic_mode=True):
    env = UnityEnvironment(file_name=path_to_app,
                           no_graphics=not graphic_mode)
    return env


def plot_scores(scores, cfg, plot_agents=True, mas=[100], plot_individual_episodes=False):
    """Plot achieved training scores with some simple options."""
    scores = pd.Series(scores.max(axis=1))

    if plot_agents:
        scores_all = pd.DataFrame(scores)

    # plot the scores
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    for ma_i in mas:

        if plot_individual_episodes:
            ax.plot(scores.index, scores.values, ls="-", color="k", label="Scores")

        ma = scores.rolling(ma_i).mean()
        ax.plot(ma.index, ma.values, ls="-", color="r", lw=2,
                label="Moving average scores ({:d} episodes)".format(ma_i))

        # need to define other colors for individual agents
        if plot_agents and scores_all.shape[1] > 1:
            for iagent in range(scores_all.shape[1]):
                scores_i = scores_all[iagent]

                if plot_individual_episodes:
                    ax.plot(scores_all.index, scores_i, ls="-", color="k",
                            label="Scores (Agent {:d})".format(iagent))

                ma = scores.rolling(ma_i).mean()
                ax.plot(ma.index, ma.values, ls="-", color="r", lw=2,
                        label="Moving average scores ({:d} episodes, Agent {:d})".format(ma_i,
                                                                                         iagent))

    plt.gca().xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend(loc="upper left")

    plt.savefig(os.path.join(cfg.experiment_path, "scores.pdf"))
    plt.savefig(os.path.join(cfg.experiment_path, "scores.png"))


def ddpg_learning(env, agent, brain_name, cfg,
                  n_episodes=2000, max_t=100000,
                  avg_score_cutoff=15,
                  model_save_path=None):
    """Function to perform Deep Deterministic Policy Gradient learning.

    Params
    ======
        env: environment that the agent interacts with
        agent: agent instance that does all the computation/optimization in the background
        brain_name: name of the Unity brain instance to select correct agent/window
        cfg: config settings
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of time-steps per episode
        avg_score_cutoff (float): training will be stopped
                                  if avg score over last 100 episodes exceeds this value
        model_save_path: path to directory to save model weights in
    """
    print("Training an agent with DDPG.")

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = env.brains[brain_name].vector_action_space_size
    # state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)

    if not os.path.exists(model_save_path):
        print("Creating directory {:s} to save model weights into!".format(model_save_path))
        os.mkdir(model_save_path)

    all_scores = []  # list containing scores from each episode

    for i_episode in range(1, n_episodes + 1):

        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        scores = np.zeros(num_agents)

        for t in range(max_t):

            if cfg.maddpg:
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]
            else:
                actions = agent.act(states.reshape(-1))
                env_info = env.step(actions.reshape(num_agents, action_size))

            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            if cfg.maddpg:
                agent.step(states, actions, rewards, next_states, dones)
            else:
                # single agent with states and actions stacked together
                agent.step(states.reshape(-1), actions.reshape(-1),
                           np.max(rewards), next_states.reshape(-1),
                           np.any(dones))

            states = next_states
            scores += rewards
            if np.any(dones):
                break

        all_scores.append(scores)  # save most recent score

        last100mean = np.mean(np.max(np.atleast_2d(all_scores), axis=1)[-100:])
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(
                i_episode, last100mean))

            if model_save_path is not None:
                agent.save_weights(model_save_path)

            if cfg.save_scores:
                pd.DataFrame(scores).to_hdf(cfg.save_scores, "scores")

        if last100mean >= avg_score_cutoff:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(
                i_episode, last100mean))

            break

    # save trained models a final time
    if model_save_path is not None:
        agent.save_weights(model_save_path)

    return pd.DataFrame(all_scores)


def train_or_play(cfg):
    # initialize the environment and obtain state/action sizes and other parameters
    env = init_environment(cfg.app_path, graphic_mode=cfg.graphic_mode)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = env_info.vector_observations.shape[1]
    num_agents = len(env_info.agents)

    if cfg.maddpg:
        agent = MultiDDPGAgent(state_size, action_size, num_agents, cfg)
    else:
        agent = SingleDDPGAgent(state_size, action_size, cfg)

    print(cfg.model_save_path)
    if os.path.exists(cfg.model_save_path):
        print("Loading existing weights from path {:s}!".format(cfg.model_save_path))
        agent.load_weights(cfg.model_save_path)

    if cfg.train_model:

        scores = ddpg_learning(env, agent, brain_name, cfg,
                               n_episodes=cfg.n_episodes, max_t=cfg.max_t,
                               avg_score_cutoff=cfg.avg_score_cutoff,
                               model_save_path=cfg.model_save_path)

        if cfg.save_scores:
            print("Saving scores to file {:s}".format(cfg.save_scores))
            scores.to_hdf(cfg.save_scores, "scores")

        plot_scores(scores, cfg)

    else:  # visualize trained model and scores

        assert os.path.exists(cfg.model_save_path), \
            "Saved model weights need to exist before you can watch a trained agent!"

        print("Visualizing the trained agent!")

        print("Re-plotting scores")
        if os.path.exists(cfg.save_scores):
            plot_scores(pd.read_hdf(cfg.save_scores, "scores"), cfg)

        for epi in range(1, cfg.eps_to_watch + 1):

            print("Playing episode {:d}".format(epi))

            env_info = env.reset(train_mode=False)[brain_name]

            scores = np.zeros(num_agents)  # initialize the score
            states = env_info.vector_observations
            while True:

                if cfg.maddpg:
                    actions = agent.act(states, add_noise=False)  # take step without noise
                    env_info = env.step(actions)[brain_name]
                else:
                    actions = agent.act(states.reshape(-1), add_noise=False)
                    env_info = env.step(actions.reshape(num_agents, action_size))

                states = env_info.vector_observations  # new_states
                rewards = env_info.rewards
                dones = env_info.local_done
                scores += rewards
                if np.any(dones):
                    break

            # warning: n_agent 2 logic hardcoded here
            print("Episode {:d} ended with score: {:.2f} (agent 1: {:.2f}, agent 2: {:.2f})!".
                  format(epi, np.max(scores), scores[0], scores[1]))

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

    # ActivateDarkMode()
    train_or_play(cfg)
