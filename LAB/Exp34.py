import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------------------
# Smart Home Environment
# -----------------------------------------

class SmartHomeEnv:

    def __init__(self):

        self.state_size = 2
        self.action_size = 3

        # Actions:
        # 0 -> Cooling
        # 1 -> No Change
        # 2 -> Heating

        self.target_temperature = 24
        self.max_steps = 50

    def reset(self):

        self.temperature = np.random.uniform(
            18,
            30
        )

        self.steps = 0

        return self.get_state()

    def get_state(self):

        return np.array([
            self.temperature / 40,
            self.target_temperature / 40
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        energy = 0

        # Cooling

        if action == 0:

            self.temperature -= 1
            energy = 2

        # No change

        elif action == 1:

            energy = 0

        # Heating

        elif action == 2:

            self.temperature += 1
            energy = 2

        # ---------------------------------
        # Comfort Error
        # ---------------------------------

        error = abs(
            self.temperature -
            self.target_temperature
        )

        comfort_reward = -error

        # Energy penalty

        energy_penalty = -0.5 * energy

        reward = (
            comfort_reward +
            energy_penalty
        )

        terminated = False

        truncated = (
            self.steps >=
            self.max_steps
        )

        return (
            self.get_state(),
            reward,
            terminated,
            truncated
        )


# -----------------------------------------
# Policy Network
# -----------------------------------------

model = Sequential([

    tf.keras.Input(
        shape=(2,)
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        3,
        activation="softmax"
    )
])


optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

gamma = 0.95

episodes = 1000


# -----------------------------------------
# REINFORCE Training
# -----------------------------------------

env = SmartHomeEnv()

for episode in range(episodes):

    state = env.reset()

    states = []
    actions = []
    rewards = []

    for step in range(
        env.max_steps
    ):

        state_tensor = tf.convert_to_tensor(
            [state],
            dtype=tf.float32
        )

        probabilities = model(
            state_tensor
        )[0]

        action = np.random.choice(
            3,
            p=probabilities.numpy()
        )

        (
            next_state,
            reward,
            terminated,
            truncated
        ) = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state

        if terminated or truncated:
            break

    # -------------------------------------
    # Discounted Returns
    # -------------------------------------

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = (
            reward +
            gamma * G
        )

        returns.insert(
            0,
            G
        )

    returns = np.array(
        returns,
        dtype=np.float32
    )

    # Normalize returns

    if np.std(returns) > 0:

        returns = (
            returns -
            np.mean(returns)
        ) / (
            np.std(returns) +
            1e-8
        )

    # -------------------------------------
    # Policy Update
    # -------------------------------------

    with tf.GradientTape() as tape:

        loss = 0

        for state, action, G in zip(
            states,
            actions,
            returns
        ):

            state_tensor = tf.convert_to_tensor(
                [state],
                dtype=tf.float32
            )

            probabilities = model(
                state_tensor
            )[0]

            log_probability = tf.math.log(
                probabilities[action] +
                1e-8
            )

            loss += (
                -log_probability * G
            )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    if (
        episode + 1
    ) % 200 == 0:

        print(
            "Episode:",
            episode + 1,
            "| Average Reward:",
            round(
                np.mean(rewards),
                2
            )
        )


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

print("\nSmart Home Evaluation\n")

for step in range(30):

    state_tensor = tf.convert_to_tensor(
        [state],
        dtype=tf.float32
    )

    probabilities = model(
        state_tensor
    )[0]

    action = np.argmax(
        probabilities.numpy()
    )

    (
        next_state,
        reward,
        terminated,
        truncated
    ) = env.step(action)

    total_reward += reward

    action_names = [
        "Cooling",
        "No Change",
        "Heating"
    ]

    print(
        "Step:", step + 1,
        "| Temperature:",
        round(env.temperature, 2),
        "| Action:",
        action_names[action],
        "| Reward:",
        round(reward, 2)
    )

    state = next_state

    if terminated or truncated:
        break


print(
    "\nFinal Temperature:",
    round(
        env.temperature,
        2
    )
)

print(
    "Target Temperature:",
    env.target_temperature
)

print(
    "Total Reward:",
    round(
        total_reward,
        2
    )
)