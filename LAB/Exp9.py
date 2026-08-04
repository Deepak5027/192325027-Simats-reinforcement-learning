import numpy as np
import random

n_states = 6
n_actions = 2

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

# ---------------- TD(0) ----------------

V = np.zeros(n_states)

for _ in range(episodes):

    state = 0

    while state < 5:

        next_state = state + 1

        reward = 10 if next_state == 5 else -1

        V[state] += alpha * (
            reward +
            gamma * V[next_state] -
            V[state]
        )

        state = next_state

print("TD(0) State Values:")
print(np.round(V, 2))


# ---------------- SARSA ----------------

Q_sarsa = np.zeros((n_states, n_actions))

def choose_action(Q, state):

    if random.random() < epsilon:
        return random.randint(0, 1)

    return np.argmax(Q[state])


for _ in range(episodes):

    state = 0
    action = choose_action(Q_sarsa, state)

    while state < 5:

        if action == 0:
            next_state = min(state + 1, 5)
        else:
            next_state = max(state - 1, 0)

        reward = 10 if next_state == 5 else -1

        next_action = choose_action(
            Q_sarsa,
            next_state
        )

        Q_sarsa[state, action] += alpha * (
            reward +
            gamma *
            Q_sarsa[next_state, next_action] -
            Q_sarsa[state, action]
        )

        state = next_state
        action = next_action

print("\nSARSA Q-Table:")
print(np.round(Q_sarsa, 2))


# ---------------- Q-LEARNING ----------------

Q_learning = np.zeros((n_states, n_actions))

for _ in range(episodes):

    state = 0

    while state < 5:

        action = choose_action(
            Q_learning,
            state
        )

        if action == 0:
            next_state = min(state + 1, 5)
        else:
            next_state = max(state - 1, 0)

        reward = 10 if next_state == 5 else -1

        Q_learning[state, action] += alpha * (
            reward +
            gamma *
            np.max(Q_learning[next_state]) -
            Q_learning[state, action]
        )

        state = next_state

print("\nQ-Learning Q-Table:")
print(np.round(Q_learning, 2))

print("\nOptimal Warehouse Policy:")

for state in range(5):

    action = np.argmax(
        Q_learning[state]
    )

    if action == 0:
        print(
            "State", state,
            "-> Move Forward"
        )
    else:
        print(
            "State", state,
            "-> Move Backward"
        )