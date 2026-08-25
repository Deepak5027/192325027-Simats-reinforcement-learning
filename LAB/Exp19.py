from pettingzoo.mpe import simple_spread_v3
import supersuit as ss
from stable_baselines3 import PPO

# --------------------------------------
# Multi-Agent Warehouse Environment
# --------------------------------------

env = simple_spread_v3.parallel_env(
    N=3,
    local_ratio=0.5,
    max_cycles=50
)

# Convert to Stable-Baselines3 format
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1)

# --------------------------------------
# Train PPO Agent
# --------------------------------------

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=20000)

# --------------------------------------
# Evaluation
# --------------------------------------

obs = env.reset()

total_reward = 0

for _ in range(50):

    action, _ = model.predict(obs)

    obs, reward, done, info = env.step(action)

    total_reward += reward.mean()

    if done.any():
        break

print("\nTraining Completed Successfully")
print("Average Team Reward :", round(total_reward,2))