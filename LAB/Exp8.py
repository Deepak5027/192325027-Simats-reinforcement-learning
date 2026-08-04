import numpy as np
import random

rooms = 5
actions = ["Clean", "Move"]
episodes = 1000

Q = np.zeros((rooms, 2))
returns = [[[] for _ in range(2)] for _ in range(rooms)]

def get_reward(state, action):
    if action == 0:
        return 10
    return -1

for _ in range(episodes):

    episode = []
    state = random.randint(0, rooms - 1)

    for step in range(5):

        action = random.randint(0, 1)
        reward = get_reward(state, action)

        episode.append((state, action, reward))

        if action == 1:
            state = min(state + 1, rooms - 1)

    G = 0

    for state, action, reward in reversed(episode):

        G = reward + 0.9 * G

        returns[state][action].append(G)

        Q[state][action] = np.mean(
            returns[state][action]
        )

print("Learned Q-Values:")
print(np.round(Q, 2))

print("\nOptimal Cleaning Policy:")

for state in range(rooms):

    best_action = np.argmax(Q[state])

    print(
        "Room", state + 1,
        "->", actions[best_action]
    )