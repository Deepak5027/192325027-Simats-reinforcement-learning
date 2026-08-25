import numpy as np

# -----------------------------------------
# Traffic Signal Environment
# -----------------------------------------

# States:
# 0 -> Low traffic
# 1 -> Medium traffic
# 2 -> High traffic

states = 3

# Actions:
# 0 -> Green for North-South
# 1 -> Green for East-West

actions = 2

gamma = 0.9


# -----------------------------------------
# Rewards
# -----------------------------------------

# Reward represents reduction in waiting time.
#
# Low traffic:
# Either signal is acceptable.
#
# Medium traffic:
# Appropriate signal receives higher reward.
#
# High traffic:
# Appropriate signal receives highest reward.

reward = np.array([
    [5, 4],
    [3, 7],
    [2, 10]
])


# -----------------------------------------
# Transition Probabilities
# -----------------------------------------

# P[state][action][next_state]

P = np.zeros(
    (states, actions, states)
)

# Low traffic

P[0, 0] = [0.7, 0.3, 0.0]
P[0, 1] = [0.6, 0.4, 0.0]

# Medium traffic

P[1, 0] = [0.3, 0.6, 0.1]
P[1, 1] = [0.5, 0.4, 0.1]

# High traffic

P[2, 0] = [0.1, 0.5, 0.4]
P[2, 1] = [0.2, 0.5, 0.3]


# -----------------------------------------
# Initial Policy
# -----------------------------------------

policy = np.zeros(
    states,
    dtype=int
)


# -----------------------------------------
# Policy Iteration
# -----------------------------------------

while True:

    # -------------------------------------
    # Policy Evaluation
    # -------------------------------------

    V = np.zeros(states)

    for _ in range(1000):

        new_V = np.zeros(states)

        for s in range(states):

            a = policy[s]

            new_V[s] = (
                reward[s, a]
                + gamma *
                np.sum(
                    P[s, a] * V
                )
            )

        if np.max(
            np.abs(new_V - V)
        ) < 0.001:

            V = new_V
            break

        V = new_V


    # -------------------------------------
    # Policy Improvement
    # -------------------------------------

    stable = True

    for s in range(states):

        old_action = policy[s]

        action_values = []

        for a in range(actions):

            value = (
                reward[s, a]
                + gamma *
                np.sum(
                    P[s, a] * V
                )
            )

            action_values.append(value)

        best_action = np.argmax(
            action_values
        )

        policy[s] = best_action

        if old_action != best_action:

            stable = False

    if stable:

        break


# -----------------------------------------
# Display Results
# -----------------------------------------

state_names = [
    "Low Traffic",
    "Medium Traffic",
    "High Traffic"
]

action_names = [
    "North-South Green",
    "East-West Green"
]

print("Optimal Traffic Signal Policy\n")

for s in range(states):

    print(
        state_names[s],
        "->",
        action_names[policy[s]]
    )


print("\nOptimal State Values:")

for s in range(states):

    print(
        state_names[s],
        ":",
        round(V[s], 2)
    )


# -----------------------------------------
# Simulate Traffic
# -----------------------------------------

print("\nTraffic Signal Simulation\n")

state = 0
total_reward = 0

for step in range(10):

    action = policy[state]

    current_reward = reward[
        state,
        action
    ]

    total_reward += current_reward

    probabilities = P[
        state,
        action
    ]

    next_state = np.random.choice(
        states,
        p=probabilities
    )

    print(
        "Step:", step + 1,
        "| State:", state_names[state],
        "| Signal:", action_names[action],
        "| Reward:", current_reward,
        "| Next State:",
        state_names[next_state]
    )

    state = next_state


print(
    "\nTotal Simulation Reward:",
    total_reward
)