import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Environment settings
max_position = 5
max_battery = 10

state_size = 2
action_size = 2

# DQN model
model = Sequential([
    tf.keras.Input(shape=(state_size,)),
    Dense(24, activation="relu"),
    Dense(24, activation="relu"),
    Dense(action_size, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 300

for episode in range(episodes):

    position = 0
    battery = max_battery

    done = False

    while not done:

        state = np.array(
            [[position / max_position,
              battery / max_battery]]
        )

        if random.random() < epsilon:

            action = random.randint(
                0,
                action_size - 1
            )

        else:

            q_values = model.predict(
                state,
                verbose=0
            )

            action = np.argmax(q_values[0])

        # Action 0 = move one step
        # Action 1 = move two steps

        if action == 0:
            next_position = min(
                position + 1,
                max_position
            )
            battery_cost = 1

        else:
            next_position = min(
                position + 2,
                max_position
            )
            battery_cost = 2

        next_battery = (
            battery - battery_cost
        )

        if next_position == max_position:

            reward = 20
            done = True

        elif next_battery <= 0:

            reward = -20
            done = True

        else:

            reward = -battery_cost

        next_state = np.array(
            [[next_position / max_position,
              max(next_battery, 0) /
              max_battery]]
        )

        target = model.predict(
            state,
            verbose=0
        )

        if done:

            target[0][action] = reward

        else:

            next_q = model.predict(
                next_state,
                verbose=0
            )

            target[0][action] = (
                reward +
                gamma * np.max(next_q[0])
            )

        model.fit(
            state,
            target,
            epochs=1,
            verbose=0
        )

        position = next_position
        battery = next_battery

    if epsilon > epsilon_min:

        epsilon *= epsilon_decay


# Testing the trained drone

position = 0
battery = max_battery
route = [position]

while position < max_position and battery > 0:

    state = np.array(
        [[position / max_position,
          battery / max_battery]]
    )

    q_values = model.predict(
        state,
        verbose=0
    )

    action = np.argmax(q_values[0])

    if action == 0:

        position = min(
            position + 1,
            max_position
        )

        battery -= 1

    else:

        position = min(
            position + 2,
            max_position
        )

        battery -= 2

    route.append(position)

print("Optimal Drone Route:")
print(" -> ".join(map(str, route)))

print("Remaining Battery:", battery)

if position == max_position:
    print("Delivery Successful")
else:
    print("Delivery Failed")