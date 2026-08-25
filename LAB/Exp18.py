import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# -------------------------------------------------
# Manufacturing Robot Environment
# -------------------------------------------------

class ManufacturingRobotEnv(gym.Env):

    def __init__(self, target=5):

        super().__init__()

        self.target = target

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=10,
            shape=(1,),
            dtype=np.float32
        )

        self.max_steps = 30

        self.reset()

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.position = 0

        self.steps = 0

        return np.array([self.position], dtype=np.float32), {}

    def step(self, action):

        self.steps += 1

        # Actions
        # 0 -> Move Left
        # 1 -> Stay
        # 2 -> Move Right

        if action == 0:
            self.position -= 1

        elif action == 2:
            self.position += 1

        reward = -1

        if self.position == self.target:
            reward = 20

        terminated = self.position == self.target
        truncated = self.steps >= self.max_steps

        observation = np.array(
            [self.position],
            dtype=np.float32
        )

        return observation, reward, terminated, truncated, {}

# -------------------------------------------------
# Task 1 Training
# -------------------------------------------------

env1 = ManufacturingRobotEnv(target=5)

model = PPO(
    "MlpPolicy",
    env1,
    verbose=0
)

model.learn(total_timesteps=10000)

# -------------------------------------------------
# Meta Adaptation
# New Manufacturing Task
# -------------------------------------------------

env2 = ManufacturingRobotEnv(target=8)

model.set_env(env2)

model.learn(total_timesteps=5000)

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

obs, _ = env2.reset()

total_reward = 0

for _ in range(30):

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, _ = env2.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print("Target Position :", env2.target)
print("Final Position :", int(obs[0]))
print("Total Reward :", total_reward)

if terminated:
    print("Task Adaptation Successful")
else:
    print("Task Not Completed")