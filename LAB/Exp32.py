import numpy as np
import random
import tensorflow as tf

from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


# -----------------------------------------
# GridWorld Environment
# -----------------------------------------

class GridWorld:

    def __init__(self):

        self.size = 5

        self.start = (0, 0)
        self.goal = (4, 4)

        self.obstacles = [
            (1, 1),
            (2, 1),
            (3, 3)
        ]

        self.state_size = 2
        self.action_size = 4

        # 0 -> Up
        # 1 -> Down
        # 2 -> Left
        # 3 -> Right

    def reset(self):

        self.position = self.start

        return self.get_state()

    def get_state(self):

        return np.array([
            self.position[0] / 4,
            self.position[1] / 4
        ], dtype=np.float32)

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

        # Boundary checking

        r = max(0, min(4, r))
        c = max(0, min(4, c))

        new_position = (r, c)

        # Obstacle

        if new_position in self.obstacles:

            reward = -10
            new_position = self.position

        else:

            reward = -1

        self.position = new_position

        # Goal

        if self.position == self.goal:

            reward = 20
            done = True

        else:

            done = False

        return (
            self.get_state(),
            reward,
            done
        )


# -----------------------------------------
# Standard DQN
# -----------------------------------------

def build_dqn():

    model = Sequential([
        Input(shape=(2,)),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(4, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# -----------------------------------------
# Dueling DQN
# -----------------------------------------

def build_dueling_dqn():

    inputs = tf.keras.Input(
        shape=(2,)
    )

    x = Dense(
        64,
        activation="relu"
    )(inputs)

    x = Dense(
        64,
        activation="relu"
    )(x)

    # State-value stream

    value = Dense(
        1,
        activation="linear"
    )(x)

    # Advantage stream

    advantage = Dense(
        4,
        activation="linear"
    )(x)

    # Q(s,a) = V(s) +
    # A(s,a) - mean(A(s,a))

    q_values = (
        value
        +
        advantage
        -
        tf.reduce_mean(
            advantage,
            axis=1,
            keepdims=True
        )
    )

    model = tf.keras.Model(
        inputs=inputs,
        outputs=q_values
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse"
    )

    return model


# -----------------------------------------
# Training Function
# -----------------------------------------

def train_agent(model, episodes=300):

    env = GridWorld()

    target_model = tf.keras.models.clone_model(
        model
    )

    target_model.set_weights(
        model.get_weights()
    )

    memory = deque(
        maxlen=5000
    )

    gamma = 0.95

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    batch_size = 32

    rewards_history = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0

        for step in range(50):

            # Epsilon-greedy

            if random.random() < epsilon:

                action = random.randint(
                    0,
                    3
                )

            else:

                q = model.predict(
                    state.reshape(1, -1),
                    verbose=0
                )[0]

                action = np.argmax(q)

            next_state, reward, done = (
                env.step(action)
            )

            memory.append(
                (
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )
            )

            state = next_state

            total_reward += reward

            if len(memory) >= batch_size:

                batch = random.sample(
                    memory,
                    batch_size
                )

                states = []
                targets = []

                for (
                    s,
                    a,
                    r,
                    ns,
                    d
                ) in batch:

                    target = model.predict(
                        s.reshape(1, -1),
                        verbose=0
                    )[0]

                    if d:

                        target[a] = r

                    else:

                        next_q = (
                            target_model.predict(
                                ns.reshape(1, -1),
                                verbose=0
                            )[0]
                        )

                        target[a] = (
                            r
                            +
                            gamma *
                            np.max(next_q)
                        )

                    states.append(s)
                    targets.append(target)

                model.fit(
                    np.array(states),
                    np.array(targets),
                    epochs=1,
                    verbose=0
                )

            if done:
                break

        if epsilon > epsilon_min:

            epsilon *= epsilon_decay

        # Update target network

        if episode % 10 == 0:

            target_model.set_weights(
                model.get_weights()
            )

        rewards_history.append(
            total_reward
        )

    return rewards_history


# -----------------------------------------
# Evaluation
# -----------------------------------------

def evaluate(model):

    env = GridWorld()

    state = env.reset()

    total_reward = 0

    path = [env.position]

    for _ in range(50):

        q = model.predict(
            state.reshape(1, -1),
            verbose=0
        )[0]

        action = np.argmax(q)

        state, reward, done = (
            env.step(action)
        )

        total_reward += reward

        path.append(env.position)

        if done:
            break

    return total_reward, path


# -----------------------------------------
# Train Standard DQN
# -----------------------------------------

print("Training Standard DQN...")

dqn = build_dqn()

dqn_rewards = train_agent(
    dqn,
    episodes=300
)


# -----------------------------------------
# Train Dueling DQN
# -----------------------------------------

print(
    "\nTraining Dueling DQN..."
)

dueling = build_dueling_dqn()

dueling_rewards = train_agent(
    dueling,
    episodes=300
)


# -----------------------------------------
# Evaluation
# -----------------------------------------

dqn_reward, dqn_path = evaluate(
    dqn
)

dueling_reward, dueling_path = evaluate(
    dueling
)


print("\nDQN Evaluation")
print(
    "Reward:",
    dqn_reward
)

print(
    "Path:",
    dqn_path
)


print("\nDueling DQN Evaluation")
print(
    "Reward:",
    dueling_reward
)

print(
    "Path:",
    dueling_path
)


# -----------------------------------------
# Comparison
# -----------------------------------------

if dueling_reward > dqn_reward:

    print(
        "\nDueling DQN performed better."
    )

elif dqn_reward > dueling_reward:

    print(
        "\nStandard DQN performed better."
    )

else:

    print(
        "\nBoth algorithms performed equally."
    )