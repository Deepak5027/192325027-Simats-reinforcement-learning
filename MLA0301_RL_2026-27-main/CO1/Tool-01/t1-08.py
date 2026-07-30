import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

np.random.seed(0)
env = gym.make("FrozenLake-v1", is_slippery=True)
q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, epsilon = 0.8, 0.95, 1.0
rewards = []

for episode in range(2000):
    state, info = env.reset()
    done, total = False, 0
    while not done:
        action = env.action_space.sample() if np.random.random() < epsilon else int(np.argmax(q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        q[state, action] += alpha * (reward + gamma * np.max(q[next_state]) - q[state, action])
        state = next_state
        total += reward
    rewards.append(total)
    epsilon = max(0.01, epsilon * 0.999)

env.close()
print(f"Average reward (last 100): {np.mean(rewards[-100:]):.3f}")

smoothed = np.convolve(rewards, np.ones(50) / 50, mode="valid")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(rewards, color="tab:blue", alpha=0.6)
axes[0].set(title="Reward per Episode", xlabel="Episode", ylabel="Reward")
axes[1].plot(np.cumsum(rewards), color="tab:green")
axes[1].set(title="Cumulative Reward", xlabel="Episode", ylabel="Cumulative Reward")
axes[2].plot(smoothed, color="tab:red")
axes[2].set(title="Learning Curve (50-ep avg)", xlabel="Episode", ylabel="Average Reward")
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("t1-08_reward_visualization.png", dpi=120)
plt.show()
