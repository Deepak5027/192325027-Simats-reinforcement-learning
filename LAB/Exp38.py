import numpy as np
import random

# -----------------------------------------
# POMDP Robot Navigation Environment
# -----------------------------------------

class RobotEnv:

    def __init__(self):

        # Hidden locations
        # 0 -> Start
        # 1 -> Middle
        # 2 -> Goal

        self.states = 3

        # Actions
        # 0 -> Move Left
        # 1 -> Move Right

        self.actions = 2

    def reset(self):

        self.state = 0

        return self.observe()

    # -------------------------------------
    # Partial Observation
    # -------------------------------------

    def observe(self):

        if self.state == 0:

            observations = [
                "Start",
                "Unknown"
            ]

        elif self.state == 1:

            observations = [
                "Middle",
                "Unknown"
            ]

        else:

            observations = [
                "Goal",
                "Unknown"
            ]

        return random.choice(
            observations
        )

    # -------------------------------------
    # Environment Step
    # -------------------------------------

    def step(self, action):

        # Move Right

        if action == 1:

            if self.state < 2:

                self.state += 1

        # Move Left

        elif action == 0:

            if self.state > 0:

                self.state -= 1

        # Reward

        if self.state == 2:

            reward = 20
            done = True

        else:

            reward = -1
            done = False

        observation = self.observe()

        return observation, reward, done


# -----------------------------------------
# Belief Update
# -----------------------------------------

def update_belief(
    observation
):

    if observation == "Start":

        return np.array([
            0.9,
            0.1,
            0.0
        ])

    elif observation == "Middle":

        return np.array([
            0.1,
            0.8,
            0.1
        ])

    elif observation == "Goal":

        return np.array([
            0.0,
            0.1,
            0.9
        ])

    else:

        return np.array([
            1 / 3,
            1 / 3,
            1 / 3
        ])


# -----------------------------------------
# Q-Learning on Belief States
# -----------------------------------------

env = RobotEnv()

Q = np.zeros(
    (3, 2)
)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 3000

for episode in range(episodes):

    observation = env.reset()

    belief = update_belief(
        observation
    )

    state = np.argmax(belief)

    for step in range(20):

        # Epsilon-greedy

        if random.random() < epsilon:

            action = random.randint(
                0,
                1
            )

        else:

            action = np.argmax(
                Q[state]
            )

        next_observation, reward, done = (
            env.step(action)
        )

        next_belief = update_belief(
            next_observation
        )

        next_state = np.argmax(
            next_belief
        )

        # Q-learning update

        Q[state, action] += alpha * (
            reward
            +
            gamma *
            np.max(Q[next_state])
            -
            Q[state, action]
        )

        belief = next_belief

        state = next_state

        if done:

            break


# -----------------------------------------
# Evaluation
# -----------------------------------------

observation = env.reset()

belief = update_belief(
    observation
)

state = np.argmax(belief)

total_reward = 0

path = [env.state]

print(
    "POMDP Robot Navigation\n"
)

for step in range(20):

    action = np.argmax(
        Q[state]
    )

    next_observation, reward, done = (
        env.step(action)
    )

    next_belief = update_belief(
        next_observation
    )

    next_state = np.argmax(
        next_belief
    )

    print(
        "Step:", step + 1,
        "| Observation:",
        next_observation,
        "| Action:",
        "Left" if action == 0
        else "Right",
        "| Reward:",
        reward
    )

    path.append(
        env.state
    )

    total_reward += reward

    belief = next_belief
    state = next_state

    if done:

        break


print(
    "\nLearned Q-Table:"
)

print(
    np.round(Q, 2)
)

print(
    "\nPath:",
    path
)

print(
    "Total Reward:",
    total_reward
)

if env.state == 2:

    print(
        "Robot Reached Goal Successfully"
    )

else:

    print(
        "Robot Failed to Reach Goal"
    )