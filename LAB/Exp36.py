import numpy as np
import random

# -----------------------------------------
# Multi-Agent MAXQ Environment
# -----------------------------------------

class WarehouseEnv:

    def __init__(self):

        self.agents = 2

        # States:
        # 0 -> Start
        # 1 -> Item Picked
        # 2 -> Item Transported
        # 3 -> Item Stored
        # 4 -> Task Completed

        self.state = 0

    def reset(self):

        self.state = 0

        return self.state

    def step(self, agent, action):

        reward = -1

        # Agent 0 performs picking
        if agent == 0 and action == 0:

            if self.state == 0:

                self.state = 1
                reward = 10

        # Agent 1 performs transportation
        elif agent == 1 and action == 1:

            if self.state == 1:

                self.state = 2
                reward = 10

        # Agent 0 stores item
        elif agent == 0 and action == 2:

            if self.state == 2:

                self.state = 3
                reward = 10

        # Final cooperative completion
        if self.state == 3:

            self.state = 4
            reward += 20

        done = self.state == 4

        return self.state, reward, done


# -----------------------------------------
# MAXQ Hierarchy
# -----------------------------------------

# Composite tasks:
#
# ROOT
#   |
#   +-- PICK
#   |
#   +-- TRANSPORT
#   |
#   +-- STORE

env = WarehouseEnv()

# MAXQ completion values
C = {}

alpha = 0.1
gamma = 0.9

episodes = 3000


# -----------------------------------------
# MAXQ Value Function
# -----------------------------------------

def get_value(state, task):

    key = (state, task)

    if key not in C:

        C[key] = 0.0

    return C[key]


# -----------------------------------------
# Training
# -----------------------------------------

for episode in range(episodes):

    state = env.reset()

    done = False

    while not done:

        # ---------------------------------
        # Pick
        # ---------------------------------

        if state == 0:

            task = "PICK"

            agent = 0
            action = 0

        # ---------------------------------
        # Transport
        # ---------------------------------

        elif state == 1:

            task = "TRANSPORT"

            agent = 1
            action = 1

        # ---------------------------------
        # Store
        # ---------------------------------

        elif state == 2:

            task = "STORE"

            agent = 0
            action = 2

        else:

            break

        old_state = state

        next_state, reward, done = env.step(
            agent,
            action
        )

        old_value = get_value(
            old_state,
            task
        )

        next_value = 0

        if not done:

            next_value = get_value(
                next_state,
                task
            )

        # MAXQ completion-function update

        C[
            (old_state, task)
        ] = old_value + alpha * (
            reward
            + gamma * next_value
            - old_value
        )

        state = next_state


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

print("MAXQ Hierarchical Multi-Agent RL\n")

while state != 4:

    if state == 0:

        agent = 0
        task = "PICK"
        action = 0

    elif state == 1:

        agent = 1
        task = "TRANSPORT"
        action = 1

    elif state == 2:

        agent = 0
        task = "STORE"
        action = 2

    else:

        break

    next_state, reward, done = env.step(
        agent,
        action
    )

    print(
        "Task:", task,
        "| Agent:", agent + 1,
        "| Reward:", reward
    )

    total_reward += reward

    state = next_state


print("\nMAXQ Completion Values:")

for key, value in C.items():

    print(
        key,
        "->",
        round(value, 2)
    )

print(
    "\nTotal Reward:",
    total_reward
)

if state == 4:

    print(
        "Overall Cooperative Task Completed"
    )