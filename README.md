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

```python
def optimize_network(functions, hp, replay: ReplayBuffer):
    if len(replay) < hp["minibatch_size"]:
        return
    device = functions["device"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    loss_function = functions["loss_function"]
    optimizer = functions["optimizer"]

    batch = replay.sample(hp["minibatch_size"])
    batch = Transition(*zip(*batch))

    states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=device)
    terminated = torch.tensor(batch.terminated, dtype=int, device=device)

    predicted  = policy_nn(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_states_qvalues = torch.max(target_nn(next_states), dim=1).values * (1 - terminated)
    target = rewards + 0.99 * next_states_qvalues
    loss = loss_function(predicted, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)

    optimizer.step()
```
Below I've written code for two different action selection functions. Decaying Epsilon-Greedy seemed to work better across the board then undecaying softmax which is why we'll use it in the final run.
```python

def e_greedy(q_values, functions, hp):
    env = functions["env"]
    device = functions["device"]
    eps_threshold = hp["eps_end"] + (hp["eps_start"] - hp["eps_end"]) \
                    * math.exp(-1. * hp["steps_done"] / hp["eps_decay"])
    hp["steps_done"] += 1
    sample = torch.rand(1).item()

    if sample > eps_threshold:
        with torch.no_grad():
            return q_values.max(0).indices.item()
    else:
        action = torch.randint(0, env.action_space.n, size=(1,)).item()
        return torch.tensor([action], device=device, dtype=torch.long).item()

def softmax(state_qvalues, functions, hp):
    state_qvalues_probabilities = torch.softmax(state_qvalues, dim=0)
    state_qvalues_dis = torch.distributions.Categorical(state_qvalues_probabilities)
    action = state_qvalues_dis.sample().item()
    return action
```

Here is where our policy neural network is tested and trained. After continuously interacting with the environment, the replay buffer will be large enough to start gradient descent. After the policy network is optimized, I perform a soft update on the target network. Soft updating provides stability by preventing the target network from shifting too rapidly and smooths convergence by reducing the risk of overfitting to recent experiences or changes in the policy network.

$$
\theta_{\text{target}} \leftarrow \tau \theta_{\text{policy}} + (1 - \tau) \theta_{\text{target}}
$$

```python
def training_loop(functions, hp):
    env = functions["env"]
    policy_nn = functions["policy_nn"]
    target_nn = functions["target_nn"]
    device = functions["device"]
    select_action = functions["select_action"]
    replay = ReplayBuffer(hp["ReplayBuffer_capacity"])
    reward_sum = 0

    for episode in range(hp["episodes"]):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        terminated = False

        for t in count():
            with torch.no_grad():
                state_qvalues = policy_nn(state)
            action = select_action(state_qvalues, functions, hp)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            replay.push(state.tolist(), action, reward, next_state, terminated)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

            for _ in range(hp["replay_steps"]):
                optimize_network(functions, hp, replay)

            # soft update
            TAU = 0.005
            target_net_state_dict = target_nn.state_dict()
            policy_net_state_dict = policy_nn.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_nn.load_state_dict(target_net_state_dict)
        
            if terminated or truncated:
                episode_rewards.append(reward_sum) # TODO
                reward_sum = 0
                plot_rewards()
                break
            
    print('Complete')
    plot_rewards(show_result=True)
    plt.ioff()
    plt.show()
```
Main() is where we finally populate the hyperparameters, choose our functions, and run our code.
```python
def main():
    # env = gym.make('LunarLander-v2', render_mode="human")
    env = gym.make('LunarLander-v2')


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_nn = DQN(state_dim, action_dim).to(device)
    target_nn = DQN(state_dim, action_dim).to(device)
    target_nn.load_state_dict(policy_nn.state_dict())
    lr = 1e-4

    functions = {
        "env": env,
        "policy_nn": policy_nn,
        "target_nn": target_nn,
        "loss_function": nn.SmoothL1Loss(),
        "optimizer": optim.Adam(policy_nn.parameters(), lr=lr),
        "device": device,
        "select_action": e_greedy # softmax or e_greedy
    }
    hp = {
        "episodes": 500,
        "graph_increment": 10,
        "replay_steps": 2,
        "learning_rate": lr,
        "tau": 0.001,
        "ReplayBuffer_capacity": 10000,
        "minibatch_size": 64,
        "eps_end": 0.05,
        "eps_start": 0.9,
        "eps_decay": 500,
        "steps_done": 0,
    }

    training_loop(functions, hp)
    # torch.save(policy_nn.state_dict(), "learned_policy.pth")
    env.close()
    return policy_nn

policy_nn = main()
```
After running our network for 1000 episodes, it's converged to a solution and is ready to be tested below.
```
env = gym.make('LunarLander-v2', render_mode="human")
for episode in range(5):
    state, _ = env.reset()
    for t in count():
        state = torch.tensor(state, dtype=torch.float32, device=device)
        state_qvalues = policy_nn(state)
        action = state_qvalues.max(0).indices.item()
        next_state, reward, terminated, truncated, info = env.step(action)

        state = next_state

        if terminated or truncated:
            break
```
