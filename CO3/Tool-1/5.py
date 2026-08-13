# Challenge 5: Reward shaping accelerates learning and prevents poor policies.
# 1D corridor length 12, goal at the end, sparse reward (+1 only at goal).
# Potential-based shaping adds gamma*phi(s')-phi(s) using distance-to-goal.
# We compare how many episodes reach the goal (out of 400) and the first hit.
import numpy as np
np.random.seed(0)
N, GOAL, gamma = 12, 11, 0.99

def phi(s): return -(GOAL - s)          # closer to goal -> higher potential

def train(shaped):
    Q = np.zeros((N, 2)); eps = 0.2
    successes, first = 0, None
    for ep in range(400):
        s = 0
        for t in range(200):
            a = np.random.randint(2) if np.random.random() < eps else int(np.argmax(Q[s]))
            s2 = min(GOAL, s+1) if a == 1 else max(0, s-1)
            base = 1.0 if s2 == GOAL else 0.0
            r = base + (gamma*phi(s2) - phi(s) if shaped else 0.0)
            Q[s, a] += 0.1 * (r + gamma*np.max(Q[s2]) - Q[s, a])
            s = s2
            if s == GOAL:
                successes += 1
                if first is None: first = ep
                break
    return successes, first

for shaped, name in [(False, "Sparse reward only"), (True, "With reward shaping")]:
    succ, first = train(shaped)
    hit = f"first reached at episode {first}" if first is not None else "goal never reached"
    print(f"{name:<20}: {succ:>3}/400 episodes reached goal | {hit}")
print("\nShaping supplies a dense learning signal, so the agent solves the task")
print("almost immediately, while sparse reward leaves it wandering for long.")
