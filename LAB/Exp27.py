import numpy as np
import random

# -----------------------------------------
# Road Network
# -----------------------------------------

# States represent intersections
#
# 0 -> Start
# 1 -> Intersection 1
# 2 -> Intersection 2
# 3 -> Intersection 3
# 4 -> Destination

n_states = 5

# Actions:
# 0 -> Move Forward
# 1 -> Turn Left
# 2 -> Turn Right

n_actions = 3

destination = 4


# -----------------------------------------
# Legal Actions
# -----------------------------------------

legal_actions = {

    0: [0, 1, 2],

    1: [0, 2],

    2: [0, 1],

    3: [0, 2],

    4: []
}


# -----------------------------------------
# Road Transitions
# -----------------------------------------

transitions = {

    (0, 0): 1,
    (0, 1): 2,
    (0, 2): 3,

    (1, 0): 4,
    (1, 2): 2,

    (2, 0): 4,
    (2, 1): 3,

    (3, 0): 4,
    (3, 2): 2
}


# -----------------------------------------
# Q-Table
# -----------------------------------------

Q = np.zeros(
    (n_states, n_actions)
)


# -----------------------------------------
# Parameters
# -----------------------------------------

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 5000
max_steps = 20


# -----------------------------------------
# Training
# -----------------------------------------

for episode in range(episodes):

    state = 0

    for step in range(max_steps):

        actions = legal_actions[state]

        if not actions:
            break

        # Epsilon-greedy selection

        if random.random() < epsilon:

            action = random.choice(
                actions
            )

        else:

            values = [
                Q[state, a]
                for a in actions
            ]

            action = actions[
                np.argmax(values)
            ]

        # ---------------------------------
        # Invalid Action Check
        # ---------------------------------

        if (state, action) not in transitions:

            reward = -10

            next_state = state

        else:

            next_state = transitions[
                (state, action)
            ]

            # Safe movement reward

            reward = 2

            # Destination reward

            if next_state == destination:

                reward = 20

        # ---------------------------------
        # Q-Learning Update
        # ---------------------------------

        Q[state, action] += alpha * (
            reward
            + gamma *
            np.max(Q[next_state])
            - Q[state, action]
        )

        state = next_state

        if state == destination:

            break


# -----------------------------------------
# Display Learned Q-Table
# -----------------------------------------

print("Learned Q-Table:\n")

print(
    np.round(Q, 2)
)


# -----------------------------------------
# Evaluate Learned Policy
# -----------------------------------------

state = 0

path = [state]

total_reward = 0

print("\nAutonomous Car Evaluation\n")

for step in range(max_steps):

    actions = legal_actions[state]

    if not actions:
        break

    values = [
        Q[state, action]
        for action in actions
    ]

    action = actions[
        np.argmax(values)
    ]

    next_state = transitions[
        (state, action)
    ]

    reward = 2

    if next_state == destination:

        reward = 20

    total_reward += reward

    path.append(next_state)

    print(
        "State:", state,
        "| Action:", action,
        "| Next State:", next_state,
        "| Reward:", reward
    )

    state = next_state

    if state == destination:
        break


# -----------------------------------------
# Result
# -----------------------------------------

print("\nOptimal Road Path:")

print(
    " -> ".join(
        map(str, path)
    )
)

print(
    "\nTotal Reward:",
    total_reward
)

if state == destination:

    print(
        "Destination Reached Safely"
    )

else:

    print(
        "Destination Not Reached"
    )