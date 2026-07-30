import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
K, STEPS = 10, 1000
true_values = np.random.normal(0, 1, K)


def epsilon_greedy(epsilon=0.1):
    q, counts, rewards = np.zeros(K), np.zeros(K), np.zeros(STEPS)
    for t in range(STEPS):
        a = np.random.randint(K) if np.random.random() < epsilon else int(np.argmax(q))
        reward = np.random.normal(true_values[a], 1)
        counts[a] += 1
        q[a] += (reward - q[a]) / counts[a]
        rewards[t] = reward
    return rewards


def ucb(c=2.0):
    q, counts, rewards = np.zeros(K), np.zeros(K), np.zeros(STEPS)
    for t in range(STEPS):
        if t < K:
            a = t
        else:
            a = int(np.argmax(q + c * np.sqrt(np.log(t + 1) / (counts + 1e-9))))
        reward = np.random.normal(true_values[a], 1)
        counts[a] += 1
        q[a] += (reward - q[a]) / counts[a]
        rewards[t] = reward
    return rewards


print(f"Optimal arm: {np.argmax(true_values)}\n")
eg, u = epsilon_greedy(), ucb()
print(f"Epsilon-Greedy | avg = {np.mean(eg):.3f} | total = {np.sum(eg):.1f}")
print(f"UCB            | avg = {np.mean(u):.3f} | total = {np.sum(u):.1f}")

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(eg), label="Epsilon-Greedy (eps=0.1)")
plt.plot(np.cumsum(u), label="UCB (c=2.0)")
plt.xlabel("Steps")
plt.ylabel("Cumulative Reward")
plt.title("Multi-Armed Bandit: Epsilon-Greedy vs. UCB")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("t1-07_bandit_comparison.png", dpi=120)
plt.show()
