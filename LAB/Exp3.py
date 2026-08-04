import numpy as np

states = ["A", "B", "C", "Goal"]
rewards = [-2, -2, -2, 20]
gamma = 0.8

values = np.zeros(len(states))

for _ in range(10):
    new_values = values.copy()
    for i in range(len(states)-1):
        new_values[i] = rewards[i] + gamma * values[i+1]
    new_values[-1] = rewards[-1]
    values = new_values

print("Warehouse Robot State Values")
for s, v in zip(states, values):
    print(s, ":", round(v,2))

print("\nOptimal Route")
for i in range(len(states)-1):
    print(states[i], "->", states[i+1])