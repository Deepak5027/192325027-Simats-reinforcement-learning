import numpy as np
import random

# -----------------------------------------
# Grid World Pac-Man Environment
# -----------------------------------------

class PacmanEnv:

    def __init__(self):

        self.rows = 5
        self.cols = 5

        # Pac-Man starts at (0,0)
        self.start = (0, 0)

        # Food position
        self.food = (4, 4)

        # Ghost positions
        self.ghosts = [(2, 2), (3, 3)]

        self.actions = 4

        # 0 = Up
        # 1 = Down
        # 2 = Left
        # 3 = Right

    def reset(self):

        self.position = self.start

        return self.position

    def step(self, action):

        r, c = self.position

        if action == 0:
            r -= 1

        elif action == 1:
            r += 1

        elif action == 2:
            c -= 1

        elif action == 3:
            c += 1

        # Prevent movement outside grid
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))

        self.position = (r, c)

        # Food reached
        if self.position == self.food:

            return self.position, 20, True

        # Ghost reached
        if self.position in self.ghosts:

            return self.position, -20, True

        # Normal movement
        return self.position, -1, False


# -----------------------------------------
# Q-Learning
# -----------------------------------------

env = PacmanEnv()

Q = np.zeros(
    (
        env.rows,
        env.cols,
        env.actions
    )
)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 5000
max_steps = 50

# -----------------------------------------
# Training
# -----------------------------------------

for episode in range(episodes):

    state = env.reset()

    for step in range(max_steps):

        r, c = state

        # Epsilon-greedy action

        if random.random() < epsilon:

            action = random.randint(
                0,
                env.actions - 1
            )

        else:

            action = np.argmax(
                Q[r, c]
            )

        next_state, reward, done = env.step(
            action
        )

        nr, nc = next_state

        # Q-learning update

        Q[r, c, action] += alpha * (
            reward
            + gamma * np.max(Q[nr, nc])
            - Q[r, c, action]
        )

        state = next_state

        if done:
            break


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

path = [state]
total_reward = 0

print("Pac-Man Q-Learning Evaluation\n")

for step in range(max_steps):

    r, c = state

    action = np.argmax(
        Q[r, c]
    )

    next_state, reward, done = env.step(
        action
    )

    path.append(next_state)

    total_reward += reward

    state = next_state

    if done:
        break

print("Path:")

for position in path:
    print(position)

print("\nTotal Reward:", total_reward)

if state == env.food:
    print("Food Collected Successfully")

elif state in env.ghosts:
    print("Pac-Man Hit a Ghost")

else:
    print("Maximum Steps Reached")