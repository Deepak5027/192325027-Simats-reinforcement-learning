import gymnasium as gym
import numpy as np
import random

# -----------------------------
# Household Environment
# -----------------------------

class HouseholdEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.action_space = gym.spaces.Discrete(4)

        self.observation_space = gym.spaces.Discrete(5)

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.state = 0

        return self.state, {}

    def step(self, action):

        reward = -1

        # Task Order:
        # 0 -> Kitchen
        # 1 -> Pick Object
        # 2 -> Living Room
        # 3 -> Drop Object
        # 4 -> Charging Dock

        if action == self.state:

            self.state += 1
            reward = 10

        done = self.state == 4

        return min(self.state,4), reward, done, False, {}

# ---------------------------------------
# MAXQ-like Hierarchical Policy
# ---------------------------------------

env = HouseholdEnv()

tasks = [
    "Go to Kitchen",
    "Pick Object",
    "Go to Living Room",
    "Drop Object"
]

obs, _ = env.reset()

total_reward = 0

print("Hierarchical Task Execution\n")

for option in range(4):

    print("Subtask :", tasks[option])

    next_state, reward, done, _, _ = env.step(option)

    print("Reward :", reward)

    total_reward += reward

    if done:
        break

print("\nFinal State :", next_state)

print("Total Reward :", total_reward)

if done:
    print("Main Task Completed Successfully")