import numpy as np
import random

# -----------------------------------------
# Smart Energy Management Environment
# -----------------------------------------

class EnergyEnv:

    def __init__(self):

        # States:
        # 0 -> Low demand
        # 1 -> Medium demand
        # 2 -> High demand

        self.states = 3

        # Actions:
        # 0 -> Reduce consumption
        # 1 -> Normal consumption
        # 2 -> Increase consumption

        self.actions = 3

    def reset(self):

        self.state = random.randint(0, 2)

        return self.state

    def step(self, action):

        # Energy cost
        energy_cost = [
            1,
            3,
            6
        ][action]

        # Comfort reward
        comfort = [
            -2,
            3,
            5
        ][action]

        # Safety constraint
        if self.state == 2 and action == 2:

            safety_penalty = -10

        else:

            safety_penalty = 0

        reward = (
            comfort
            - energy_cost
            + safety_penalty
        )

        # Next demand state
        next_state = random.randint(0, 2)

        return next_state, reward


# -----------------------------------------
# Q-Learning
# -----------------------------------------

env = EnergyEnv()

Q = np.zeros(
    (env.states, env.actions)
)

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 5000

for episode in range(episodes):

    state = env.reset()

    for step in range(20):

        # Epsilon-greedy selection

        if random.random() < epsilon:

            action = random.randint(
                0,
                env.actions - 1
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
            reward +
            gamma * np.max(
                Q[next_state]
            )
            - Q[state, action]
        )

        state = next_state


# -----------------------------------------
# Learned Policy
# -----------------------------------------

actions = [
    "Reduce Consumption",
    "Normal Consumption",
    "Increase Consumption"
]

states = [
    "Low Demand",
    "Medium Demand",
    "High Demand"
]

print("Smart Energy Management\n")

for state in range(env.states):

    best_action = np.argmax(Q[state])

    print(
        states[state],
        "->",
        actions[best_action]
    )


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

print("\nEvaluation\n")

for step in range(20):

    action = np.argmax(Q[state])

    next_state, reward = env.step(
        action
    )

    print(
        "Step:", step + 1,
        "| State:", states[state],
        "| Action:", actions[action],
        "| Reward:", reward
    )

    total_reward += reward

    state = next_state

print("\nLearned Q-Table:")
print(np.round(Q, 2))

print(
    "\nTotal Evaluation Reward:",
    total_reward
)