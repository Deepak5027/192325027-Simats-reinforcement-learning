import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

states = env.observation_space.n
actions = env.action_space.n

Q = np.zeros((states, actions))

alpha = 0.8
gamma = 0.95
epsilon = 0.2
episodes = 2000

for episode in range(episodes):

    state, _ = env.reset()
    done = False

    while not done:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        Q[state, action] += alpha * (
            reward +
            gamma * np.max(Q[next_state]) -
            Q[state, action]
        )

        state = next_state

print("Learned Q-Table:")
print(np.round(Q, 2))

state, _ = env.reset()
path = [state]
done = False

while not done:

    action = np.argmax(Q[state])

    next_state, reward, terminated, truncated, _ = env.step(action)

    state = next_state
    path.append(state)

    done = terminated or truncated

print("\nOptimal Navigation Path:")
print(" -> ".join(map(str, path)))

print("Goal Reached:", reward == 1)

env.close()