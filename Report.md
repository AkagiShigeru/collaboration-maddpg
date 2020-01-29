# Report to solving the Tennis environment with MADDPG

## Approach

The **Tennis environment** consists of two agents that are both rewarded by
keeping the ball in play. If each agent maximizes only its own profit, it will therefore
not lead to an optimal solution. The agents have to learn to collaborate to maximize both
of their rewards/scores.

For this solution, we used **DDPG**, which is a special type of actor-critic method.

Those are the summarized main points underlying the method:
- Neural networks with dense layers and ReLu activations are used to approximate both the actor and critic networks.
- To provide a more stable convergence, both networks are split into separate but identical local and target networks. At each optimization step, the target network parameters are updated with a soft update.
- The Q-learning side is tackled by optimizing the TD-error/MSBE by continuously minimizing the squared Bellman-equation residual.
- The actions in the target Q-function are taken directly from the target policy. The policy is kept fixed for the Q-learning.
- The policy is trained by simply performing gradient descent on the negative of the current Q-function, given the actions by the policy. The Q-function parameters are kept fixed for this step, only the policy parameters are changed.

More details about the DDPG method in general are nicely summarized here: [https://spinningup.openai.com/en/latest/algorithms/ddpg.html]

### Implementation details

All relevant parameters used for training can be specified in the config file and are automatically passed to the relevant parts of the code.
The PyTorch networks representing actor and critic were written such as to allow a customizable number and size of dense layers.
We used two layers with 128 nodes each for both networks.
Experience replay with a memory/buffer size of 10000 was used to obtain training frames from the environment. By default, the actual training samples are
drawn randomly from this memory (with uniform probability). We also implemented prioritized experience replay on the Q-learning part based on the
[original paper](https://arxiv.org/pdf/1511.05952.pdf). However, we couldn't see a clear benefit of using the prioritized replay in our tests. It might need to be further adapted or tuned for DDPG. 

We found that a typical discount factor of 0.99 leads to a worse training and slower convergence than a slightly reduced factor of 0.9.
The parameter tau controlling the soft update, as well as both actor and critic learning rates were set to 0.001.
We also noticed an improved training when the standard deviation of the noise added to actions is reduced.
We use uncorrelated Gaussian noise instead of sampling from the Ornstein-Uhlenbeck process. The former seems to work better in our case, and seems to be more intuitive in general.

## Results

According to the project description, the task is considered solved if the agent obtains a score above 30 when averaged over the last
100 episodes. Our agent is able to solve this task after less than 250 training episodes. Detailed tests with the prioritized replay have not been performed.
We had set our own threshold a bit higher to an average score of 31 over the last 100 episodes, which was reached after around 270 episodes.
The following image contains the training score as a function of the episodes for the latest training run:

<img src="scores.png" alt="training scores" width="500">

It is apparent that, while the environment can be formally considered solved, the variance in scores for different individual episodes is still very large
towards the end of the training period. This is probably due to the fact that the speed and rotation of the ball in the environment changes often.
The agent has learned some modes perfectly while it is still struggling with other ones. The agent should therefore be trained even longer
and possibly the criterion for the solution of 30 should be increased a bit to maybe 35.

The figure, model weights and the config file to reproduce the results are stored in the corresponding folder
in `experiments`. The trained agent can be watched by setting `train_mode=False` and adjusting the paths.

The config file `replay_config.py` is already set-up and configured to replay the trained agent.
So, if you would like to visualize the trained agent, just execute:

`python navigation.py replay_config.py`

## Future steps and improvements

We found that the solution of the environment with DDPG is quite sensitive to training parameters, the network architecture, and also the noise
that has to be added to the actions. Thus, it seems a bit fragile. The solution could be improved by tuning the prioritized replay (settings),
optimizing hyperparameters etc. One could also implement a pure exploration phase at the beginning where actions are uniformly sampled and only switch
to normal agent exploration afterwards. Another improvement would be to change/reduce the noise with increased time or decreased distance to the target score.
Even another potential improvement would be to change the network architectures by adding dropout layers or using batch normalization. This could lead to a better generalization and faster convergence.

Apart from these optimizations in the context of DDPG, it would be worthwhile to study the same environment with different methods, for Trust Region Policy Optimization
or Proximal Policy Optimization (PPO).

One can parallelize the collection of experiences by using the environment with multiple agents and train only one agent in the background on all of those experiences.

We will continue studying some of those improvements.
