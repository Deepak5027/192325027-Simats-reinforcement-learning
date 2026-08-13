# Challenge 1: Agent-Environment interaction & cumulative reward
# A 1D corridor: states 0..5, goal at 5. Actions: 0=left, 1=right.
# Reward -1 per step, +10 on reaching the goal. We compare two policies
# to show how cumulative reward (the return) drives which policy is better.
import numpy as np
GOAL = 5

def step(s, a):
    s2 = min(GOAL, s + 1) if a == 1 else max(0, s - 1)
    reward = 10 if s2 == GOAL else -1
    done = s2 == GOAL
    return s2, reward, done

def run(policy, name, trace=False):
    s, total, steps = 0, 0, 0
    if trace: print(f"\n{name} policy trace:")
    while True:
        a = policy(s)
        s2, r, done = step(s, a)
        total += r; steps += 1
        if trace: print(f"  state {s} --action {'right' if a else 'left'}--> state {s2} | reward {r:+d}")
        s = s2
        if done or steps > 30: break
    print(f"{name:<16} cumulative reward (return) = {total:>4} in {steps} steps")
    return total

np.random.seed(0)
run(lambda s: 1, "Always-right", trace=True)          # optimal
run(lambda s: np.random.randint(2), "Random")          # weaker
print("\nThe agent prefers the policy with the higher cumulative reward.")
