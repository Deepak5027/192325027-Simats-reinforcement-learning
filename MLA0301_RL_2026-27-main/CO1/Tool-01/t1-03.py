import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon = 0.8, 0.95, 1.0
rewards = []

for episode in range(2000):
    state, info = env.reset()
    done = False
    while not done:
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = int(np.argmax(q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        q[state, action] += alpha * (reward + gamma * np.max(q[next_state]) - q[state, action])
        state = next_state
    rewards.append(reward)
    epsilon = max(0.01, epsilon * 0.999)

env.close()
print(f"Average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}\n")
print("Learned policy (< v > ^):")
symbols = {0: "<", 1: "v", 2: ">", 3: "^"}
for row in np.argmax(q, axis=1).reshape(4, 4):
    print("  " + "  ".join(symbols[a] for a in row))
