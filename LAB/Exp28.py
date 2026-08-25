import numpy as np

# -----------------------------------------
# Grid Environment
# -----------------------------------------

rows = 5
cols = 5

start = (0, 0)
goal = (4, 4)

gamma = 0.9

# Actions:
# 0 -> Up
# 1 -> Down
# 2 -> Left
# 3 -> Right

actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

V = np.zeros((rows, cols))

# -----------------------------------------
# Bellman Value Iteration
# -----------------------------------------

for iteration in range(100):

    new_V = V.copy()

    for r in range(rows):

        for c in range(cols):

            if (r, c) == goal:

                new_V[r, c] = 10
                continue

            values = []

            for dr, dc in actions:

                nr = r + dr
                nc = c + dc

                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                ):

                    if (nr, nc) == goal:

                        reward = 10

                    else:

                        reward = -1

                    value = (
                        reward
                        + gamma * V[nr, nc]
                    )

                    values.append(value)

            if values:

                new_V[r, c] = max(values)

    if np.max(
        np.abs(new_V - V)
    ) < 0.001:

        V = new_V
        break

    V = new_V


# -----------------------------------------
# Display Value Function
# -----------------------------------------

print("Optimal State-Value Function:\n")

print(
    np.round(V, 2)
)


# -----------------------------------------
# Find Optimal Path
# -----------------------------------------

state = start

path = [state]

total_reward = 0

for step in range(30):

    if state == goal:
        break

    r, c = state

    best_value = -float("inf")
    best_state = state
    best_reward = 0

    for dr, dc in actions:

        nr = r + dr
        nc = c + dc

        if (
            0 <= nr < rows
            and 0 <= nc < cols
        ):

            if (nr, nc) == goal:

                reward = 10

            else:

                reward = -1

            value = (
                reward
                + gamma * V[nr, nc]
            )

            if value > best_value:

                best_value = value
                best_state = (nr, nc)
                best_reward = reward

    state = best_state

    path.append(state)

    total_reward += best_reward


print("\nOptimal Path:")

for position in path:

    print(position)

print(
    "\nTotal Reward:",
    total_reward
)

if state == goal:

    print("Goal Reached Successfully")