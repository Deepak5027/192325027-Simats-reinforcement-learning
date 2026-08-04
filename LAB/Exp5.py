import numpy as np
import random

true_rates = [0.1, 0.3, 0.6]
n_ads = len(true_rates)

counts = np.zeros(n_ads)
values = np.zeros(n_ads)

epsilon = 0.1
trials = 1000
total_reward = 0

for _ in range(trials):

    if random.random() < epsilon:
        ad = random.randint(0, n_ads - 1)
    else:
        ad = np.argmax(values)

    reward = 1 if random.random() < true_rates[ad] else 0

    counts[ad] += 1
    total_reward += reward

    values[ad] += (reward - values[ad]) / counts[ad]

print("Advertisement Selection Counts:")
for i in range(n_ads):
    print("Ad", i + 1, ":", int(counts[i]))

print("\nEstimated Click Rates:")
for i in range(n_ads):
    print("Ad", i + 1, ":", round(values[i], 3))

print("\nBest Advertisement: Ad", np.argmax(values) + 1)
print("Total Clicks:", total_reward)