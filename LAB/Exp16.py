import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# -----------------------------------------
# Lane Keeping Environment
# -----------------------------------------

class LaneKeepingEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        self.max_steps = 200

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.position = np.random.uniform(-0.1, 0.1)
        self.velocity = 0
        self.steps = 0

        return np.array(
            [self.position, self.velocity],
            dtype=np.float32
        ), {}

    def step(self, action):

        self.steps += 1

        # 0 = Left
        # 1 = Straight
        # 2 = Right

        steering = [-0.05, 0, 0.05][action]

        self.velocity += steering
        self.position += self.velocity

        reward = 1 - abs(self.position)

        terminated = abs(self.position) > 1
        truncated = self.steps >= self.max_steps

        observation = np.array(
            [self.position, self.velocity],
            dtype=np.float32
        )

        return observation, reward, terminated, truncated, {}

# -----------------------------------------
# Train PPO
# -----------------------------------------

env = LaneKeepingEnv()

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=20000)

# -----------------------------------------
# Evaluate
# -----------------------------------------

obs, _ = env.reset()

total_reward = 0

for _ in range(200):

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print("\nEvaluation Reward =", round(total_reward, 2))

print("Final Lane Position =", round(obs[0], 3))