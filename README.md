# Solving the Tennis multi-agent task with a Multi-Agent Deep Deterministic Policy Gradient

This repository provides code to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment
with Deep Reinforcement Learning, specifically with the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) method.
The code in this repository can be used to train an agent or inspect/visualize a pre-trained agent.
The code of this project was in parts adapted from code given in the Udacity Deep RL nanodegree program.

## Prerequisites

The environment is based on a Unity application, so please make sure to install
the [ml-agents repository](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
in order to interact with the environment via the Python code of this project.

Furthermore, the full Python environment as given in `environment.yml` is required to work with the code.  
Please consider using conda package management and install the environment with
```
conda env create -f environment.yml
```

This will create a new environment with the name `tennis-ddpg` and will install all dependencies.

The actual Tennis Unity application can be downloaded from the links provided here:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

## Details about the environment

In this environment, two agents control rackets to bounce a ball over a net.
If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.
Thus, the goal of each agent is to keep the ball in play as long as possible.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket.
The state space of size 24 consists of 3 instances of the observation space for the last three environment steps.
Each agent receives its own, local observation.
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## How to use the code

To train a reinforcement agent with this MADDPG code, you should run:

```
python collaboration.py train_config.py
```

The file `train_config.py` includes reasonable standard settings and configuration parameters.
The resulting network weights of the trained RL agent will be saved in files and the training scores are plotted.

To view a pre-trained agent in action, just change the option *train_model* in the config file to `False` and specify a valid path to a file containing the model weights.
Afterwards, run the code as given above.

Here is a brief description of the included files and their functions:

- `models.py`: Defines PyTorch neural networks for actor and critic with user-specified number of dense layers between in- and output and ReLU activation after each hidden layer.
- `ddpg_agents.py`: Includes the implementation of the single and multi RL agents that uses the actor and critic neural networks to adapt and learn the environment. Also includes an implementation of an experience replay memory with optional prioritization.
- `collaboration.py`: The main driver file that controls and steers the training and plotting.
- `utils.py`: Some utilities like replay memory and noise process class.
- `train_config.py`: A config file with settings to train a new agent.
- `replay_config.py`: A config file with settings to inspect a trained agent, based on the experiment results provided in this package.

To view a trained agent, please run:
```
python collaboration.py replay_config.py
```

For more details, the comments within the code should give further insights on how it functions.

## Persistence of results

By default, the code will store resulting plots, weights and configuration files of each experiment in timestamped subdirectories of 
the `experiments` directory. This ensures that the results of experiments are properly persisted and are not accidentally overwritten.