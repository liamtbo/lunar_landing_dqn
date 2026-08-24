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

