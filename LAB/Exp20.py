import numpy as np
import random

# -----------------------------------------
# POMDP Search and Rescue Environment
# -----------------------------------------

class RescueEnv:

    def __init__(self):
        # Hidden states:
        # 0 -> Empty
        # 1 -> Victim
        # 2 -> Obstacle

        self.states = 3

        # Actions:
        # 0 -> Search
        # 1 -> Move
        # 2 -> Rescue

        self.actions = 3

        self.reset()

    def reset(self):
        self.state = random.choice([0, 1, 2])
        return self.observe()

    def observe(self):

        if self.state == 0:
            observations = ["Nothing", "Nothing", "Obstacle"]
            return random.choice(observations)

        if self.state == 1:
            observations = ["Victim", "Victim", "Nothing"]
            return random.choice(observations)

        observations = ["Obstacle", "Obstacle", "Nothing"]
        return random.choice(observations)

    def step(self, action):

        observation = self.observe()

        # Search
        if action == 0:

            if self.state == 1:
                reward = 5
            else:
                reward = -1

        # Move
        elif action == 1:

            if self.state == 2:
                reward = -5
            else:
                reward = 1

        # Rescue
        else:

            if self.state == 1:
                reward = 20
            else:
                reward = -10

        return observation, reward


# -----------------------------------------
# Belief State
# -----------------------------------------

def update_belief(belief, observation):

    new_belief = belief.copy()

    if observation == "Victim":

        new_belief = np.array(
            [0.1, 0.8, 0.1]
        )

    elif observation == "Obstacle":

        new_belief = np.array(
            [0.1, 0.1, 0.8]
        )

    else:

        new_belief = np.array(
            [0.7, 0.2, 0.1]
        )

    return new_belief


# -----------------------------------------
# Q-Learning over Belief States
# -----------------------------------------

env = RescueEnv()

# Belief states are represented by
# the most probable hidden state.

Q = np.zeros((3, 3))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

episodes = 3000

for episode in range(episodes):

    belief = np.array(
        [1 / 3, 1 / 3, 1 / 3]
    )

    observation = env.reset()

    state = np.argmax(belief)

    for step in range(20):

        # Epsilon-greedy action selection

        if random.random() < epsilon:

            action = random.randint(0, 2)

        else:

            action = np.argmax(Q[state])

        observation, reward = env.step(action)

        # Update belief using observation

        belief = update_belief(
            belief,
            observation
        )

        next_state = np.argmax(belief)

        # Q-learning update

        Q[state, action] += alpha * (
            reward +
            gamma * np.max(Q[next_state]) -
            Q[state, action]
        )

        state = next_state


# -----------------------------------------
# Evaluation
# -----------------------------------------

belief = np.array(
    [1 / 3, 1 / 3, 1 / 3]
)

observation = env.reset()

total_reward = 0

print("POMDP Search and Rescue\n")

for step in range(20):

    state = np.argmax(belief)

    action = np.argmax(Q[state])

    observation, reward = env.step(action)

    belief = update_belief(
        belief,
        observation
    )

    total_reward += reward

    action_name = [
        "Search",
        "Move",
        "Rescue"
    ][action]

    print(
        "Step:", step + 1,
        "| Observation:", observation,
        "| Action:", action_name,
        "| Reward:", reward
    )

    if action == 2 and reward == 20:
        print("\nVictim Successfully Rescued!")
        break

print("\nLearned Q-Table:")
print(np.round(Q, 2))

print("\nTotal Evaluation Reward:",
      total_reward)