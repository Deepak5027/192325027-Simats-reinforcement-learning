import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO


# -----------------------------------------
# Autonomous Highway Environment
# -----------------------------------------

class HighwayEnv(gym.Env):

    def __init__(self):

        super().__init__()

        # Actions:
        # 0 = Stay in lane
        # 1 = Change left
        # 2 = Change right
        # 3 = Accelerate
        # 4 = Brake

        self.action_space = gym.spaces.Discrete(5)

        # State:
        # Current lane
        # Vehicle speed
        # Front vehicle distance
        # Front vehicle speed

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [0, 0, 0, 0],
                dtype=np.float32
            ),
            high=np.array(
                [2, 30, 50, 30],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        self.max_steps = 100

    # -------------------------------------
    # Reset Environment
    # -------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.lane = 1

        self.speed = 15.0

        self.front_distance = 25.0

        self.front_speed = 12.0

        self.steps = 0

        return self._get_obs(), {}

    # -------------------------------------
    # Observation
    # -------------------------------------

    def _get_obs(self):

        return np.array(
            [
                self.lane,
                self.speed,
                self.front_distance,
                self.front_speed
            ],
            dtype=np.float32
        )

    # -------------------------------------
    # Environment Step
    # -------------------------------------

    def step(self, action):

        self.steps += 1

        reward = 0

        # ---------------------------------
        # Lane Change
        # ---------------------------------

        if action == 1:

            if self.lane > 0:
                self.lane -= 1
                reward += 1
            else:
                reward -= 3

        elif action == 2:

            if self.lane < 2:
                self.lane += 1
                reward += 1
            else:
                reward -= 3

        # ---------------------------------
        # Acceleration
        # ---------------------------------

        elif action == 3:

            self.speed += 2

            self.speed = min(
                self.speed,
                30
            )

        # ---------------------------------
        # Braking
        # ---------------------------------

        elif action == 4:

            self.speed -= 3

            self.speed = max(
                self.speed,
                0
            )

        # ---------------------------------
        # Vehicle Movement
        # ---------------------------------

        self.front_distance -= (
            self.speed - self.front_speed
        ) * 0.1

        # New traffic position
        if self.front_distance < 5:

            self.front_distance = np.random.uniform(
                10,
                30
            )

            self.front_speed = np.random.uniform(
                10,
                25
            )

        # ---------------------------------
        # Reward for progress
        # ---------------------------------

        reward += self.speed * 0.1

        # Maintain safe distance
        if self.front_distance < 5:

            reward -= 10

        # Penalize excessive speed
        if self.speed > 25:

            reward -= 2

        # ---------------------------------
        # Termination
        # ---------------------------------

        terminated = False

        truncated = (
            self.steps >= self.max_steps
        )

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {}
        )


# -----------------------------------------
# Create Environment
# -----------------------------------------

env = HighwayEnv()


# -----------------------------------------
# PPO Agent
# -----------------------------------------

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=256,
    batch_size=64,
    gamma=0.99,
    verbose=1
)


# -----------------------------------------
# Training
# -----------------------------------------

model.learn(
    total_timesteps=30000
)


# -----------------------------------------
# Evaluation
# -----------------------------------------

obs, _ = env.reset()

total_reward = 0

lane_changes = 0

print("\nAutonomous Highway Evaluation\n")

for step in range(100):

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    if action == 1 or action == 2:
        lane_changes += 1

    obs, reward, terminated, truncated, _ = env.step(
        action
    )

    total_reward += reward

    print(
        "Step:", step + 1,
        "| Lane:", int(obs[0]),
        "| Speed:", round(obs[1], 2),
        "| Distance:", round(obs[2], 2),
        "| Action:", int(action)
    )

    if terminated or truncated:
        break


print("\nTotal Reward:",
      round(total_reward, 2))

print("Number of Lane Changes:",
      lane_changes)

print("Final Lane:",
      int(obs[0]))

print("Final Speed:",
      round(obs[1], 2))