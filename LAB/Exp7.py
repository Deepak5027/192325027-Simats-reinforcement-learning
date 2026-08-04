import numpy as np

rewards = np.array([
    [-1, -1, -1, -1],
    [-1, -10, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, 10]
])

rows, cols = rewards.shape
V = np.zeros((rows, cols))
gamma = 0.9

actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

for _ in range(100):
    new_V = V.copy()

    for i in range(rows):
        for j in range(cols):

            if (i, j) == (3, 3):
                new_V[i, j] = 10
                continue

            values = []

            for di, dj in actions:
                ni, nj = i + di, j + dj

                if 0 <= ni < rows and 0 <= nj < cols:
                    values.append(
                        rewards[ni, nj] +
                        gamma * V[ni, nj]
                    )

            new_V[i, j] = max(values)

    V = new_V

print("Optimal State Values:")
print(np.round(V, 2))

state = (0, 0)
goal = (3, 3)
path = [state]

while state != goal:

    best_value = -float("inf")
    best_state = state

    for di, dj in actions:

        ni = state[0] + di
        nj = state[1] + dj

        if 0 <= ni < rows and 0 <= nj < cols:

            value = rewards[ni, nj] + gamma * V[ni, nj]

            if value > best_value:
                best_value = value
                best_state = (ni, nj)

    state = best_state
    path.append(state)

print("\nOptimal Taxi Route:")
print(" -> ".join(map(str, path)))