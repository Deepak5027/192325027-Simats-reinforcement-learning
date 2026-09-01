import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.model = keras.Sequential([
            keras.Input(shape=(state_size,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(action_size, activation="linear"),
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.model.predict(state, verbose=0)[0]))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = np.vstack([b[0] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        q = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)
        for i, (_, action, reward, _, done) in enumerate(batch):
            q[i][action] = reward if done else reward + self.gamma * np.max(q_next[i])
        self.model.fit(states, q, epochs=1, verbose=0)
        self.epsilon = max(0.01, self.epsilon * 0.995)


env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
agent = DQNAgent(state_size, env.action_space.n)
scores = []

for episode in range(100):
    state, info = env.reset()
    state = state.reshape(1, -1)
    done, total = False, 0
    while not done:
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state = next_state.reshape(1, -1)
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        total += reward
    agent.replay()
    scores.append(total)
    avg = np.mean(scores[-20:])
    print(f"Episode {episode + 1:>3} | reward = {total:>5.0f} | avg(20) = {avg:6.1f} | eps = {agent.epsilon:.3f}")
    if avg >= 195 and len(scores) >= 20:
        print(f"\nSolved! Average {avg:.1f} >= 195")
        break

env.close()
print(f"\nBest reward: {max(scores):.0f} | Success rate (>=195): {np.mean(np.array(scores) >= 195) * 100:.1f}%")

plt.figure(figsize=(10, 6))
plt.plot(scores, label="Episode Reward", alpha=0.7)
if len(scores) >= 20:
    moving = np.convolve(scores, np.ones(20) / 20, mode="valid")
    plt.plot(range(19, len(scores)), moving, color="red", label="20-episode avg")
plt.axhline(y=195, color="green", linestyle="--", label="Target (195)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("CartPole DQN - Learning Performance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("t1-10_cartpole_dqn.png", dpi=120)
plt.show()
