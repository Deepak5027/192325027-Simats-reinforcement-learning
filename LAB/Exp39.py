import numpy as np
import random

# -----------------------------------------
# Healthcare Management Environment
# -----------------------------------------

class HealthcareEnv:

    def __init__(self):

        # Patient states
        # 0 -> Stable
        # 1 -> Moderate
        # 2 -> Critical

        self.states = 3

        # Actions
        # 0 -> Routine Treatment
        # 1 -> Priority Treatment
        # 2 -> Emergency Treatment

        self.actions = 3

    def reset(self):

        self.state = random.randint(
            0,
            2
        )

        return self.state

    def step(self, action):

        # ---------------------------------
        # Reward based on patient condition
        # ---------------------------------

        if self.state == 0:

            rewards = [
                8,
                5,
                2
            ]

        elif self.state == 1:

            rewards = [
                2,
                10,
                6
            ]

        else:

            rewards = [
                -10,
                5,
                15
            ]

        reward = rewards[action]

        # ---------------------------------
        # Next Patient Condition
        # ---------------------------------

        if action == 2:

            probabilities = [
                0.7,
                0.2,
                0.1
            ]

        elif action == 1:

            probabilities = [
                0.4,
                0.5,
                0.1
            ]

        else:

            probabilities = [
                0.2,
                0.5,
                0.3
            ]

        next_state = np.random.choice(
            3,
            p=probabilities
        )

        return next_state, reward


# -----------------------------------------
# Q-Learning
# -----------------------------------------

env = HealthcareEnv()

Q = np.zeros(
    (
        env.states,
        env.actions
    )
)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 5000

# -----------------------------------------
# Training
# -----------------------------------------

for episode in range(episodes):

    state = env.reset()

    for step in range(20):

        if random.random() < epsilon:

            action = random.randint(
                0,
                2
            )

        else:

            action = np.argmax(
                Q[state]
            )

        next_state, reward = env.step(
            action
        )

        Q[state, action] += alpha * (
            reward
            +
            gamma *
            np.max(Q[next_state])
            -
            Q[state, action]
        )

        state = next_state


# -----------------------------------------
# Learned Policy
# -----------------------------------------

state_names = [
    "Stable",
    "Moderate",
    "Critical"
]

action_names = [
    "Routine Treatment",
    "Priority Treatment",
    "Emergency Treatment"
]

print(
    "Learned Healthcare Management Policy\n"
)

for state in range(3):

    action = np.argmax(
        Q[state]
    )

    print(
        state_names[state],
        "->",
        action_names[action]
    )


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

print(
    "\nHealthcare Simulation\n"
)

for step in range(20):

    action = np.argmax(
        Q[state]
    )

    next_state, reward = env.step(
        action
    )

    print(
        "Step:", step + 1,
        "| Patient State:",
        state_names[state],
        "| Treatment:",
        action_names[action],
        "| Reward:",
        reward
    )

    total_reward += reward

    state = next_state


print(
    "\nLearned Q-Table:"
)

print(
    np.round(Q, 2)
)

print(
    "\nTotal Healthcare Reward:",
    total_reward
)