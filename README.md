# Solving the Tennis multi-agent task with a Multi-Agent Deep Deterministic Policy Gradient

This repository provides code to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment
with Deep Reinforcement Learning, specifically with the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) method.
The code in this repository can be used to train an agent or inspect/visualize a pre-trained agent.
The code of this project was in parts adapted from code given in the Udacity Deep RL nanodegree program.

## Prerequisites

The environment is based on a Unity application, so please make sure to install
the [ml-agents repository](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md)
in order to interact with the environment via the Python code of this project.

Furthermore, the full environment as given in `environment.yml` is required to work with the code. Please install it with
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

We consider the environment with one agent for the purpose of this project.
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
Hence, the agent should learn to maximize the time in which it keeps its hand at the target location.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## How to use the code

To train a reinforcement agent with this code, you should run:

```
python cont_control.py train_config.py
```

The file `train_config.py` includes reasonable standard settings and configuration parameters.
The resulting network weights of the trained RL agent will be saved in a file and the training scores are plotted.

To view a pre-trained agent in action, just change the option *train_model* in the config file to `False` and specify a valid path to a file containing the model weights.
Afterwards, run the code as given above.

Here is a brief description of the included files and their functions:

- `models.py`: Defines PyTorch neural networks for actor and critic with user-specified number of dense layers between in- and output and ReLU activation after each hidden layer.
- `ddpg_agent.py`: Includes the implementation of the RL agent that uses the actor and critic neural networks to adapt and learn the environment. Also includes an implementation of an experience replay memory with optional prioritization.
- `navigation.py`: The main driver file that controls and steers the training and plotting.
- `train_config.py`: A config file with settings to train a new agent.
- `replay_config.py`: A config file with settings to inspect a trained agent, based on the experiment results provided in this package.

To view a trained agent, please run:
```
python cont_control.py replay_config.py
```

For more details, the comments within the code should give further insights on how it functions.

## Persistence of results

By default, the code will store resulting plots, weights and configuration files of each experiment in timestamped subdirectories of 
the `experiments` directory. This ensures that the results of experiments are properly persisted and are not accidentally overwritten.