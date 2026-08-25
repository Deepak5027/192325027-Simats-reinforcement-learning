import numpy as np
import random

# -----------------------------------------
# Personalized Education Environment
# -----------------------------------------

class EducationEnv:

    def __init__(self):

        # Student knowledge states
        # 0 -> Beginner
        # 1 -> Intermediate
        # 2 -> Advanced

        self.states = 3

        # Teaching actions
        # 0 -> Easy Lesson
        # 1 -> Practice
        # 2 -> Advanced Lesson

        self.actions = 3

    def reset(self):

        self.state = 0

        return self.state

    def step(self, action):

        # ---------------------------------
        # Learning Outcomes
        # ---------------------------------

        if self.state == 0:

            rewards = [
                8,
                6,
                -5
            ]

            transitions = [
                [0.2, 0.7, 0.1],
                [0.5, 0.5, 0.0],
                [0.9, 0.1, 0.0]
            ]

        elif self.state == 1:

            rewards = [
                3,
                8,
                10
            ]

            transitions = [
                [0.4, 0.6, 0.0],
                [0.1, 0.7, 0.2],
                [0.0, 0.4, 0.6]
            ]

        else:

            rewards = [
                1,
                6,
                10
            ]

            transitions = [
                [0.0, 0.5, 0.5],
                [0.0, 0.3, 0.7],
                [0.0, 0.1, 0.9]
            ]

        reward = rewards[action]

        next_state = np.random.choice(
            3,
            p=transitions[action]
        )

        return next_state, reward


# -----------------------------------------
# Q-Learning
# -----------------------------------------

env = EducationEnv()

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

        # Epsilon-greedy

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

        # Q-learning update

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
# Personalized Policy
# -----------------------------------------

state_names = [
    "Beginner",
    "Intermediate",
    "Advanced"
]

action_names = [
    "Easy Lesson",
    "Practice",
    "Advanced Lesson"
]

print(
    "Personalized Learning Policy\n"
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
# Student Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

learning_path = []

print(
    "\nStudent Learning Simulation\n"
)

for step in range(15):

    action = np.argmax(
        Q[state]
    )

    next_state, reward = env.step(
        action
    )

    learning_path.append(
        action_names[action]
    )

    print(
        "Step:", step + 1,
        "| Knowledge Level:",
        state_names[state],
        "| Intervention:",
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
    "\nPersonalized Learning Path:"
)

for lesson in learning_path:

    print(
        "->",
        lesson
    )

print(
    "\nTotal Learning Reward:",
    total_reward
)

print(
    "Final Knowledge Level:",
    state_names[state]
)