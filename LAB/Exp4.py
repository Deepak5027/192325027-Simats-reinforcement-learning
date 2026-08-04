import numpy as np

cost = np.array([
    [0, 2, 5, 0],
    [0, 0, 1, 6],
    [0, 0, 0, 2],
    [0, 0, 0, 0]
])

states = ["A", "B", "C", "D"]
goal = 3

V = [float("inf")] * 4
V[goal] = 0
policy = [-1] * 4

for i in range(2, -1, -1):
    for j in range(i + 1, 4):
        if cost[i][j] > 0:
            value = cost[i][j] + V[j]
            if value < V[i]:
                V[i] = value
                policy[i] = j

path = []
state = 0

while state != goal:
    path.append(states[state])
    state = policy[state]

path.append(states[goal])

print("Optimal Path:", " -> ".join(path))
print("Minimum Travel Cost:", V[0])