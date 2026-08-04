import numpy as np
import matplotlib.pyplot as plt

algorithms = ["DQN", "DDQN", "Dueling DQN", "PER"]

waiting_time = [38, 30, 24, 19]

best = algorithms[np.argmin(waiting_time)]

print("Average Vehicle Waiting Time\n")

for i in range(len(algorithms)):
    print(algorithms[i], ":", waiting_time[i], "seconds")

print("\nBest Algorithm:", best)

plt.bar(algorithms, waiting_time)
plt.xlabel("Algorithm")
plt.ylabel("Waiting Time (seconds)")
plt.title("Traffic Signal Control Comparison")
plt.show()