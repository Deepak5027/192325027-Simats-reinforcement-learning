import numpy as np
import random
import tensorflow as tf

from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


# -----------------------------------------
# Highway Environment
# -----------------------------------------

class HighwayEnv:

    def __init__(self):

        self.state_size = 4
        self.action_size = 3

        # Actions:
        # 0 -> Brake
        # 1 -> Maintain Speed
        # 2 -> Accelerate

        self.max_steps = 100

    def reset(self):

        self.speed = 10.0
        self.distance = 30.0
        self.front_speed = 10.0
        self.steps = 0

        return self.get_state()

    def get_state(self):

        return np.array([
            self.speed / 30,
            self.distance / 50,
            self.front_speed / 30,
            self.steps / self.max_steps
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        # ---------------------------------
        # Vehicle Action
        # ---------------------------------

        if action == 0:

            self.speed -= 2

        elif action == 2:

            self.speed += 2

        self.speed = np.clip(
            self.speed,
            0,
            30
        )

        # ---------------------------------
        # Vehicle Movement
        # ---------------------------------

        relative_speed = (
            self.speed -
            self.front_speed
        )

        self.distance -= (
            relative_speed * 0.1
        )

        # Random traffic variation

        self.front_speed = np.clip(
            self.front_speed +
            np.random.uniform(-1, 1),
            5,
            25
        )

        # ---------------------------------
        # Reward
        # ---------------------------------

        reward = self.speed * 0.1

        # Maintain safe distance

        if self.distance < 5:

            reward -= 20

        elif self.distance < 10:

            reward -= 5

        # Penalize excessive speed

        if self.speed > 25:

            reward -= 3

        # ---------------------------------
        # Collision
        # ---------------------------------

        collision = self.distance <= 0

        if collision:

            reward -= 50

        terminated = collision

        truncated = (
            self.steps >= self.max_steps
        )

        return (
            self.get_state(),
            reward,
            terminated,
            truncated
        )


# -----------------------------------------
# DQN Agent
# -----------------------------------------

class DQNAgent:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.learning_rate = 0.001

        self.memory = deque(
            maxlen=10000
        )

        self.model = self.build_model()

        self.target_model = self.build_model()

        self.update_target()

    # -------------------------------------
    # DQN Network
    # -------------------------------------

    def build_model(self):

        model = Sequential([

            tf.keras.Input(
                shape=(self.state_size,)
            ),

            Dense(
                64,
                activation="relu"
            ),

            Dense(
                64,
                activation="relu"
            ),

            Dense(
                self.action_size,
                activation="linear"
            )
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="mse"
        )

        return model

    # -------------------------------------
    # Store Experience
    # -------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # -------------------------------------
    # Epsilon-Greedy
    # -------------------------------------

    def act(self, state):

        if random.random() <= self.epsilon:

            return random.randrange(
                self.action_size
            )

        q_values = self.model.predict(
            state.reshape(1, -1),
            verbose=0
        )[0]

        return np.argmax(q_values)

    # -------------------------------------
    # Experience Replay
    # -------------------------------------

    def replay(self, batch_size):

        if len(self.memory) < batch_size:

            return

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = []
        targets = []

        for (
            state,
            action,
            reward,
            next_state,
            done
        ) in batch:

            target = self.model.predict(
                state.reshape(1, -1),
                verbose=0
            )[0]

            if done:

                target[action] = reward

            else:

                next_q = self.target_model.predict(
                    next_state.reshape(1, -1),
                    verbose=0
                )[0]

                target[action] = (
                    reward
                    + self.gamma *
                    np.max(next_q)
                )

            states.append(state)
            targets.append(target)

        self.model.fit(
            np.array(states),
            np.array(targets),
            epochs=1,
            verbose=0
        )

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

    # -------------------------------------
    # Target Network Update
    # -------------------------------------

    def update_target(self):

        self.target_model.set_weights(
            self.model.get_weights()
        )


# -----------------------------------------
# Training
# -----------------------------------------

env = HighwayEnv()

agent = DQNAgent(
    env.state_size,
    env.action_size
)

episodes = 200
batch_size = 32

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(
        env.max_steps
    ):

        action = agent.act(state)

        (
            next_state,
            reward,
            terminated,
            truncated
        ) = env.step(action)

        done = (
            terminated or
            truncated
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        state = next_state

        total_reward += reward

        agent.replay(
            batch_size
        )

        if done:

            break

    if episode % 10 == 0:

        agent.update_target()

        print(
            "Episode:",
            episode,
            "| Reward:",
            round(total_reward, 2),
            "| Epsilon:",
            round(agent.epsilon, 3)
        )


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

actions_taken = []

for step in range(
    env.max_steps
):

    q_values = agent.model.predict(
        state.reshape(1, -1),
        verbose=0
    )[0]

    action = np.argmax(q_values)

    actions_taken.append(action)

    (
        next_state,
        reward,
        terminated,
        truncated
    ) = env.step(action)

    total_reward += reward

    state = next_state

    if terminated or truncated:

        break


print("\nEvaluation")
print("Total Reward:", round(total_reward, 2))
print("Final Speed:", round(env.speed, 2))
print("Final Distance:", round(env.distance, 2))

if env.distance > 0:

    print("Driving Completed Safely")

else:

    print("Collision Occurred")