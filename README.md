# Lunar Lander DQN
##### by Liam Bouffard

![](img/Screenshot%20from%202024-08-08%2011-27-06.png)

## Jumping Right In
To watch the network learn and review code
- train_lunar_lander.ipynb

To watch a previously trained network in action
- run_prelearned.py

## Description
The Lunar Lander environment in OpenAI's Gymnasium is a control problem where the objective is to safely land a lunar lander on a dynamic moon surface. The lunar lander has a thruster on the left, right, and underside, and need to be controlled harmoniously. I deployed a deep q-learning network with a single hidden layer as a solution to this problem.

## State Space
The state space of the LunarLander-v2 environment is an 8-dimensional continuous space that 
represents the state of the lunar lander. Each state is a vector containing the following elements:
- x position: Horizontal coordinate of the lander (float).
- y position: Vertical coordinate of the lander (float).
- x velocity: Horizontal velocity of the lander (float).
- y velocity: Vertical velocity of the lander (float).
- Angle: Orientation of the lander (radians, float).
- Angular velocity: Rate of change of the lander's angle (float).
- Left leg contact: Boolean value indicating whether the left leg is in contact with the ground (0 or 1).
- Right leg contact: Boolean value indicating whether the right leg is in contact with the ground (0 or 1).


## Action Space
The action space of the LunarLander-v2 environment is discrete and consists of 4 possible actions that control the lunar lander's engines:
- Action 0: Do nothing.
- Action 1: Fire the left orientation engine (applies a force to rotate the lander counterclockwise).
- Action 2: Fire the main engine (applies a vertical force to slow the descent or lift the lander).
- Action 3: Fire the right orientation engine (applies a force to rotate the lander clockwise).

## Reward Space
- +100  Reward for a successful landing (varies based on landing quality).
- -100: Penalty for crashing.
- -0.3 per frame: Penalty for firing the main engine to discourage excessive use.
- Small positive/negative rewards are given for keeping the lander upright and minimizing velocity.

# Solution Description
This project will require the following dependencies and should be run in python 3.11.
```bash
pip install gymnasium
pip install torch
pip install tqdm
pip install numpy
pip install matplotlib
```
```python
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import random
import numpy as np
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
```
For a solution to the lunar-landing problem, I chose to implement a Deep Q-Network (DQN). Deep Q-Networks have become famous for their significant advancements within reinforcement learning and for achieving human-level performance on Atari games.
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

My DQN consists of an input layer for observations of dimension 8, one hidden layer with 128 neurons, and an action output layer with dimension 4. In order to account for the continuous observation space, I’ll be using a neural network to represent state-action values. In my model, both the input layer and the single hidden layer are set with ReLU activation functions to introduce non-linearity to the model. This non-linearity will help the model map a complex non-linear function with the purpose of landing our lunar-landing vehicle safely.
```python
class DQN(nn.Module): 
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
```
The ReplayBuffer class will hold our off-policy samples derived from interaction with the environment and will be used for stochastic gradient descent.
```python
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminated"))

class ReplayBuffer():
    def __init__(self, replay_buff_capacity):
        self.ReplayBuffer = deque([], maxlen=replay_buff_capacity)

    def push(self, *args):
        self.ReplayBuffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.ReplayBuffer, batch_size)
    
    def __len__(self):
        return len(self.ReplayBuffer)
```
The plot_rewards function will continiously plot our sum reward allowing us to visualize the network improving.
```
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

episode_rewards = []

def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
```
My optimize_network function performs gradient descent on the loss. In my code, I use the Adam Optimizer due to its adaptive learning rates and momentum estimation.

The Adam optimization equations are:

$$
\mathbf{m}_t = \beta_m \mathbf{m}_{t-1} + (1 - \beta_m) g_t
$$

$$
\mathbf{v}_t = \beta_v \mathbf{v}_{t-1} + (1 - \beta_v) g_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_m^t},
\qquad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_v^t}
$$

$$
\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\alpha \hat{\mathbf{m}}_t} {\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

Momentum is a technique to accelerate gradient descent by considering past gradients to smooth out the updates. It helps avoid local minima and can speed up convergence. Adaptive learning rate methods adjust the learning rate dynamically to improve convergence.

To perform the Q-learning update, I use a target network and a policy network to help stabilize training and improve the convergence of the algorithm.

The loss function used here is the Huber loss, which combines advantages from both Mean Squared Error (MSE) and Mean Absolute Error (MAE) by being less sensitive to outliers than MSE and more stable than MAE.

$$
L_\delta(y, \hat{y}) =
\begin{cases} 
\frac{1}{2} (y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta \\
\delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{otherwise}
\end{cases}
$$

where:
- y is the true value,
- y^ is the predicted value, and
- delta is a threshold parameter that determines the point where the loss function changes from quadratic to linear.
