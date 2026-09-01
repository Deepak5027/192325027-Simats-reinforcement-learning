import sys
import platform

libraries = [
    ("NumPy", "numpy"),
    ("Matplotlib", "matplotlib"),
    ("Gymnasium", "gymnasium"),
    ("TensorFlow", "tensorflow"),
    ("Keras", "keras"),
]

print(f"Python : {platform.python_version()}  |  {platform.system()} {platform.release()}")
print(f"Path   : {sys.executable}\n")

all_ok = True
for name, mod in libraries:
    try:
        version = getattr(__import__(mod), "__version__", "unknown")
        print(f"[OK]   {name:<12} {version}")
    except ImportError:
        print(f"[FAIL] {name:<12} not installed")
        all_ok = False

print("\nAll libraries installed!" if all_ok else "\nSome libraries missing.")

if all_ok:
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    obs, info = env.reset(seed=42)
    total = 0
    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        if terminated or truncated:
            break
    env.close()
    print(f"\nRandom agent on CartPole-v1: {total} reward in {step + 1} steps.")
    print("RL environment is ready!")
