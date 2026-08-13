# Challenge 3: Policy & value functions via the Bellman equation.
# 4x4 grid world, terminal corners, reward -1 per step, uniform-random policy.
# Iterative policy evaluation gives the state-value V(s) (Bellman expectation).
# We also derive an action-value Q(s,a) for one state.
import numpy as np
SIZE, N = 4, 16
TERM = {0, 15}
ACTIONS = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}  # U D L R

def nxt(s, a):
    if s in TERM: return s
    r, c = divmod(s, SIZE); dr, dc = ACTIONS[a]
    r = min(SIZE-1, max(0, r+dr)); c = min(SIZE-1, max(0, c+dc))
    return r*SIZE + c

def evaluate(gamma=1.0, theta=1e-4):
    V = np.zeros(N)
    while True:
        delta = 0
        for s in range(N):
            if s in TERM: continue
            v = V[s]
            V[s] = sum(0.25 * (-1 + gamma * V[nxt(s, a)]) for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
        if delta < theta: break
    return V

V = evaluate()
print("State-value V(s) under uniform-random policy (Bellman expectation):")
print(np.round(V.reshape(SIZE, SIZE), 2))
s = 5
print(f"\nAction-value Q(s={s}, a) = -1 + V(next):")
for a, name in zip(ACTIONS, ["Up","Down","Left","Right"]):
    print(f"  {name:<5}: {-1 + V[nxt(s, a)]:.2f}")
print("Best action from Bellman optimality:", ["Up","Down","Left","Right"][int(np.argmax([-1+V[nxt(s,a)] for a in ACTIONS]))])
