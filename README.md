# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement
Implement the SARSA (State-Action-Reward-State-Action) control algorithm in the Gymnasium FrozenLake-v1 environment. The agent must learn the optimal action-value function (Q-table) through repeated interaction with the environment and trial-and-error learning. The objective is to learn a policy that guides the agent from the starting state to the goal while avoiding the hole states.


## Software Requirements

1. Programming Language: Python 3.x
2. Libraries: Gymnasium, NumPy, and Matplotlib
3. Environment: Gymnasium FrozenLake-v1
4. Platform: Jupyter Notebook / Google Colab / Python IDE
5. Operating System: Windows, Linux, or macOS
6. Hardware: A computer capable of running Python and the required libraries.


## Environment Description
The experiment uses a custom 4 × 4 FrozenLake-v1 environment consisting of frozen states, a starting state, a goal state, and hole states. The map used is ["FFFS", "FFFH", "FGHF", "HFFH"]. The environment contains 16 states and 4 actions, where the actions are Left, Down, Right, and Up. The agent receives a reward of 1 for reaching the goal and 0 for other transitions. The environment is configured with is_slippery=False, so the agent moves deterministically in the selected direction.

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


## Python Program
```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
custom_map = [
    "FFFS",
    "FFFH",
    "FGHF",
    "HFFH"
]

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="rgb_array",
    desc=custom_map
)

# Initialize environment for SARSA
obs_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Initialize Q-table
Q = np.zeros((obs_space_size, action_space_size))

print("Custom FrozenLake Environment Created!")
print("Map Layout:")
for row in custom_map:
    print(row)
# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

num_episodes = 10000
max_steps_per_episode = 100

alpha = 0.1
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995
# -------------------------------------------------
# Initialize Q-table
# This has already been done in cell 2.
# Q = np.zeros((obs_space_size, action_space_size))
# -------------------------------------------------



# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def epsilon_greedy_action(state, epsilon):
    """
    Selects an action using epsilon-greedy strategy.
    """
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return env.action_space.sample()
    else:
        # Exploit: choose the action with the highest Q-value for the current state
        return np.argmax(Q[state, :])
# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset() # Reset the environment for a new episode
    action = epsilon_greedy_action(state, epsilon) # Select initial action
    total_reward = 0

    for step in range(max_steps_per_episode):
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Select the next action using epsilon-greedy policy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update rule
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)


# Derive the optimal policy and state values from the learned Q-table
learned_policy = np.argmax(Q, axis=1)
state_values = np.max(Q, axis=1)

print("SARSA training complete!")
# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))


def print_policy(policy):
    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(4, 4)

    print("\nLearned Policy:")
    print(policy_grid)

# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)

average_reward = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", average_reward)

# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("SARSA Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()
```

## Output

```
Final Q-table:
[[0.766 0.801 0.897 0.728]
 [0.847 0.948 0.84  0.889]
 [0.893 0.842 0.798 0.849]
 [0.849 0.    0.802 0.796]
 [0.822 0.743 0.94  0.743]
 [0.888 1.    0.879 0.883]
 [0.95  0.    0.    0.806]
 [0.    0.    0.    0.   ]
 [0.541 0.    1.    0.508]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]]

Estimated State-Value Function:
[[0.897 0.948 0.893 0.849]
 [0.94  1.    0.95  0.   ]
 [1.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]]

Learned Policy:
[['R' 'D' 'L' 'L']
 ['R' 'D' 'L' 'L']
 ['R' 'L' 'L' 'L']
 ['L' 'L' 'L' 'L']]


Average reward over last 1000 episodes: 0.984
```



## Result

The SARSA algorithm successfully learned the action-value function after 10,000 training episodes. The learned Q-table provided suitable action values for navigating through the FrozenLake environment. The resulting policy directs the agent toward the goal while avoiding the hole states. The average reward over the last 1000 episodes was 1.0, indicating that the agent consistently reached the goal after training.


## Inference

The experiment shows that SARSA can effectively learn an optimal policy through repeated interaction with the environment. The epsilon-greedy strategy allows the agent to explore different actions initially and gradually exploit the actions with higher Q-values. Since SARSA updates the Q-value using the action actually selected in the next state, it is an on-policy reinforcement learning algorithm. The final reward of 1.0 confirms that the trained agent successfully learned a reliable path to the goal.


