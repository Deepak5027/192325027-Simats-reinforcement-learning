import numpy as np

states = ["S0", "S1", "S2", "Goal"]
rewards = [-1, -1, -1, 10]
gamma = 0.9

V = np.zeros(len(states))

for _ in range(10):
    new_V = V.copy()
    for i in range(len(states)-1):
        new_V[i] = rewards[i] + gamma * V[i+1]
    new_V[-1] = rewards[-1]
    V = new_V

print("State Values")
for s, v in zip(states, V):
    print(s, ":", round(v,2))

print("\nOptimal Policy")
for i in range(len(states)-1):
    print(states[i], "->", states[i+1])