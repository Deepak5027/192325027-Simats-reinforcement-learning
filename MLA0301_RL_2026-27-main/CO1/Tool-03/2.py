import numpy as np
import random

arms = [0.2, 0.5, 0.8]
epsilons = [0.1, 0.3, 0.5]

for epsilon in epsilons:
    Q = np.zeros(3)
    N = np.zeros(3)

    reward_sum = 0

    for i in range(500):
        if random.random() < epsilon:
            arm = random.randint(0,2)
        else:
            arm = np.argmax(Q)

        reward = 1 if random.random() < arms[arm] else 0

        N[arm] += 1
        Q[arm] += (reward - Q[arm]) / N[arm]
        reward_sum += reward

    print("Epsilon:", epsilon)
    print("Rewards:", reward_sum)