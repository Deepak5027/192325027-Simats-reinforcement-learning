import gymnasium as gym


def explore(env_name, steps=10, **kwargs):
    print(f"\n===== {env_name} =====")
    env = gym.make(env_name, **kwargs)
    print(f"Observation space : {env.observation_space}")
    print(f"Action space      : {env.action_space}")

    state, info = env.reset(seed=0)
    print(f"Initial state     : {state}")
    total = 0
    for i in range(steps):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"Step {i + 1:>2} | action={action} | state={state} | reward={reward} | done={terminated or truncated}")
        if terminated or truncated:
            state, info = env.reset()
    print(f"Total reward: {total}")
    env.close()


explore("FrozenLake-v1", is_slippery=False)
explore("CartPole-v1")
explore("MountainCar-v0")
