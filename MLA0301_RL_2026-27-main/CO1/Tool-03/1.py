import numpy as np
import random

grid_size = 5
goal = (4, 4)
obstacles = [(2, 2), (1, 3)]

actions = [(0,1),(0,-1),(1,0),(-1,0)]
Q = np.zeros((grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

for episode in range(500):
    state = (0,0)

    while state != goal:
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[state[0], state[1]])

        move = actions[action]
        next_state = (
            max(0,min(grid_size-1,state[0]+move[0])),
            max(0,min(grid_size-1,state[1]+move[1]))
        )

        if next_state in obstacles:
            reward = -10
            next_state = state
        elif next_state == goal:
            reward = 100
        else:
            reward = -1

        Q[state[0],state[1],action] += alpha * (
            reward + gamma * np.max(Q[next_state[0],next_state[1]])
            - Q[state[0],state[1],action]
        )

        state = next_state

print(Q)