# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

## Software Requirements
Python 3.x Jupyter Notebook / Google Colab / VS Code Gymnasium – for the FrozenLake environment NumPy – for numerical operations and Q-table Matplotlib – for plotting the learning curve 


## Environment Description

FrozenLake is a grid-world environment provided by Gymnasium where an agent must move from a starting position to a goal while avoiding holes.

The environment consists of a 4 × 4 grid with 16 states. There are 4 possible actions: Left (L), Down (D), Right (R), and Up (U). The agent starts from the initial state and must reach the goal state. Frozen tiles (F) are safe, while holes (H) cause the episode to end. The goal tile (G) gives a reward of 1, while other movements generally give 0 reward. Since is_slippery=True, the agent's movement is stochastic, making the task more challenging. SARSA learns the Q-values for each state-action pair and gradually develops a policy for reaching the goal.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm
1.Initialize the FrozenLake environment and Q-table with zeros. 
2.Set hyperparameters: learning rate α, discount factor γ, and exploration rate ε. 
3.Reset the environment and obtain the initial state.
4.Select an action using the ε-greedy strategy.
5.Execute the action and observe the next state and reward. 
6.Select the next action using the same ε-greedy strategy. 
7.Update the Q-value using the SARSA equation: 
8.Q(S,A) ← Q(S,A) + α[R + γQ(S′,A′) − Q(S,A)] 
9.Set the current state and action to the next state and action. 
10.Repeat steps 5–8 until the episode terminates. 
11.Decrease ε gradually to reduce exploration and increase exploitation. 
12.Repeat the process for the specified number of episodes. 
13.Extract the state-value function and learned policy from the final Q-table. 
14.Plot the learning curve to evaluate the agent's performance.

## Python Program

```python

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

custom_map = [
    "FSFF",
    "FFFH",
    "FFGF",
    "HFFF"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_map,
    is_slippery=False
)

n_states = env.observation_space.n
n_actions = env.action_space.n

# Custom start and goal states
START_STATE = 1
GOAL_STATE = 10


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1          # Learning rate
gamma = 0.99         # Discount factor

epsilon = 1.0        # Initial exploration rate
epsilon_min = 0.05   # Minimum exploration rate
epsilon_decay = 0.9995


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros((n_states, n_actions))


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):

    # Exploration
    if np.random.random() < epsilon:
        return env.action_space.sample()

    # Exploitation
    else:
        return np.argmax(Q[state])


# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Reset environment first
    env.reset()

    # Set custom starting state
    env.unwrapped.s = START_STATE

    state = START_STATE

    # Choose first action
    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Custom goal
        if next_state == GOAL_STATE:
            reward = 1.0
            terminated = True

        total_reward += reward

        # -----------------------------------------
        # If episode has ended
        # -----------------------------------------

        if terminated or truncated:

            # SARSA terminal update
            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

            break

        # -----------------------------------------
        # Choose next action
        # -----------------------------------------

        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # -----------------------------------------
        # SARSA Update
        # -----------------------------------------

        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        # Move to next state/action
        state = next_state
        action = next_action

    # Store reward
    episode_rewards.append(total_reward)

    # -----------------------------------------
    # Epsilon Decay
    # -----------------------------------------

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# -------------------------------------------------
# State-Value Function
# -------------------------------------------------

state_values = np.max(Q, axis=1)


# -------------------------------------------------
# Learned Policy
# -------------------------------------------------

learned_policy = np.argmax(Q, axis=1)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")

    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nStart State:", START_STATE)

print("Goal State:", GOAL_STATE)

print("\nFinal Q-table:")

print(
    np.round(Q, 3)
)

print_value_function(
    state_values
)

print_policy(
    learned_policy
)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)

print(
    "\nFinal epsilon:",
    epsilon
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(10, 5))

plt.plot(
    range(
        window - 1,
        num_episodes
    ),
    moving_average
)

plt.xlabel("Episode")

plt.ylabel("Average Reward")

plt.title(
    "SARSA Learning Curve - Custom FrozenLake"
)

plt.grid(True)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()






```
---

## Output

<img width="722" height="682" alt="image" src="https://github.com/user-attachments/assets/8a46c3af-0d0f-42aa-97aa-dd0ca384cf72" />

<img width="976" height="537" alt="image" src="https://github.com/user-attachments/assets/5856cfa7-1c64-4fb1-9438-4574d99ef150" />




---

## Result
```text
The SARSA algorithm was successfully implemented in the Gymnasium FrozenLake environment. The agent learned Q-values through repeated interaction with the environment and developed a policy for reaching the goal while avoiding holes. The learning curve showed improvement in the agent's performance over the training episodes.


```

---

## Inference
```text

The experiment demonstrates that SARSA can learn an effective policy through on-policy reinforcement learning. As training progresses, the agent improves its decision-making by balancing exploration and exploitation using the ε-greedy strategy. The final Q-table represents the learned state-action values, which helps the agent choose better actions to reach the goal.

```
---

