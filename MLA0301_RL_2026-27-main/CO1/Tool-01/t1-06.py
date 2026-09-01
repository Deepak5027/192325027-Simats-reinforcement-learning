import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
K, STEPS = 10, 1000
true_values = np.random.normal(0, 1, K)


def run(strategy, epsilon=0.1):
    q = np.zeros(K)
    counts = np.zeros(K)
    rewards = np.zeros(STEPS)
    for t in range(STEPS):
        if strategy == "random":
            a = np.random.randint(K)
        elif strategy == "greedy":
            a = int(np.argmax(q))
        else:  # epsilon-greedy
            a = np.random.randint(K) if np.random.random() < epsilon else int(np.argmax(q))
        reward = np.random.normal(true_values[a], 1)
        counts[a] += 1
        q[a] += (reward - q[a]) / counts[a]
        rewards[t] = reward
    return rewards


print(f"Optimal arm: {np.argmax(true_values)}\n")
plt.figure(figsize=(10, 6))
for strat in ["random", "greedy", "epsilon-greedy"]:
    rewards = run(strat)
    print(f"{strat:<15} average reward = {np.mean(rewards):.3f}")
    plt.plot(np.cumsum(rewards) / (np.arange(STEPS) + 1), label=strat)

plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Exploration vs. Exploitation Strategies")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("t1-06_exploration_vs_exploitation.png", dpi=120)
plt.show()
