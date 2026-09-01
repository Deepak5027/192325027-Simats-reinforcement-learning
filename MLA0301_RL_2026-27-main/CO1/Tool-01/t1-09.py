import numpy as np
import gymnasium as gym
from tensorflow import keras
from tensorflow.keras import layers

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

model = keras.Sequential([
    keras.Input(shape=(state_size,)),
    layers.Dense(24, activation="relu"),
    layers.Dense(24, activation="relu"),
    layers.Dense(action_size, activation="linear"),
])
model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")
model.summary()

state, info = env.reset(seed=0)
action = env.action_space.sample()
next_state, reward, terminated, truncated, info = env.step(action)
state = state.reshape(1, -1)
next_state = next_state.reshape(1, -1)

q_values = model.predict(state, verbose=0)
print("\nInitial Q-values:", np.round(q_values, 4))

gamma = 0.95
target = q_values.copy()
target[0][action] = reward + gamma * np.max(model.predict(next_state, verbose=0))

loss = model.fit(state, target, epochs=1, verbose=0).history["loss"][0]
print(f"Loss after one step: {loss:.6f}")
print("Updated Q-values:", np.round(model.predict(state, verbose=0), 4))
env.close()
