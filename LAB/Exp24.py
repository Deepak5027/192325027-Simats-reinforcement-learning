import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------------------
# Generate Market Data
# -----------------------------------------

np.random.seed(10)

prices = [100]

for i in range(200):
    change = np.random.normal(0, 1.5)
    prices.append(max(1, prices[-1] + change))

prices = np.array(prices)

# -----------------------------------------
# Policy Network
# -----------------------------------------

model = Sequential([
    tf.keras.Input(shape=(3,)),
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

# Actions:
# 0 -> Hold
# 1 -> Buy
# 2 -> Sell

gamma = 0.95
episodes = 500


# -----------------------------------------
# REINFORCE Training
# -----------------------------------------

for episode in range(episodes):

    state = np.array([
        prices[0] / 200,
        0.0,
        1000 / 2000
    ], dtype=np.float32)

    position = 0
    cash = 1000

    states = []
    actions = []
    rewards = []

    for t in range(len(prices) - 1):

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

        old_price = prices[t]
        new_price = prices[t + 1]

        # ---------------------------------
        # Trading Actions
        # ---------------------------------

        if action == 1 and position == 0:

            position = 1
            cash -= old_price

        elif action == 2 and position == 1:

            position = 0
            cash += old_price

        # Portfolio value
        portfolio_before = (
            cash +
            position * old_price
        )

        portfolio_after = (
            cash +
            position * new_price
        )

        reward = (
            portfolio_after -
            portfolio_before
        )

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = np.array([
            new_price / 200,
            position,
            portfolio_after / 2000
        ], dtype=np.float32)

    # -------------------------------------
    # Calculate Discounted Returns
    # -------------------------------------

    returns = []

    G = 0

    for reward in reversed(rewards):

        G = reward + gamma * G

        returns.insert(0, G)

    returns = np.array(returns)

    if np.std(returns) > 0:

        returns = (
            returns - np.mean(returns)
        ) / (
            np.std(returns) + 1e-8
        )

    # -------------------------------------
    # Policy Gradient Update
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
                probabilities[action] + 1e-8
            )

            loss += -log_probability * G

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

    if (episode + 1) % 100 == 0:

        print(
            "Episode:",
            episode + 1,
            "Average Reward:",
            round(np.mean(rewards), 3)
        )


# -----------------------------------------
# Evaluation
# -----------------------------------------

cash = 1000
position = 0

for t in range(len(prices) - 1):

    state = np.array([
        prices[t] / 200,
        position,
        (
            cash +
            position * prices[t]
        ) / 2000
    ], dtype=np.float32)

    probabilities = model.predict(
        np.array([state]),
        verbose=0
    )[0]

    action = np.argmax(probabilities)

    if action == 1 and position == 0:

        position = 1
        cash -= prices[t]

    elif action == 2 and position == 1:

        position = 0
        cash += prices[t]


final_value = (
    cash +
    position * prices[-1]
)

profit = final_value - 1000

print("\nFinal Portfolio Value:",
      round(final_value, 2))

print("Initial Capital: 1000")

print("Profit:",
      round(profit, 2))

if profit >= 0:
    print("Trading Strategy Profitable")
else:
    print("Trading Strategy Produced a Loss")