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

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------


for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Select initial action using epsilon-greedy
    action = epsilon_greedy_action(state, epsilon)

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Select next action using epsilon-greedy
        next_action = epsilon_greedy_action(next_state, epsilon)

        # SARSA update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * Q[next_state, next_action] * (not terminated)
            - Q[state, action]
        )

        # Update state and action
        state = next_state
        action = next_action

        # Add reward
        total_reward += reward

        # Stop if episode ends
        if terminated or truncated:
            break

    # Store episode reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)



# State value = maximum Q-value for each state
state_values = np.max(Q, axis=1)

# Best action for each state
learned_policy = np.argmax(Q, axis=1)






```
---

## Output

<img width="722" height="682" alt="image" src="https://github.com/user-attachments/assets/d44b942e-9163-4669-a4fa-319f5f76c38f" />

<img width="976" height="537" alt="image" src="https://github.com/user-attachments/assets/62acf7de-d39c-4050-b46b-0458f98bd30a" />






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

