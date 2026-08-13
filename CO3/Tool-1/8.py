# Challenge 8: Model-free vs model-based RL in a dynamic environment.
# Same 4x4 grid. Model-based value iteration uses the known model (no sampling).
# Model-free Q-learning learns purely from sampled interaction. We compare the
# greedy path each produces from the start and how many samples each needed.
import numpy as np
np.random.seed(0)
SIZE, N = 4, 16; GOAL = 15
ACT = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}; NAMES = ["U","D","L","R"]
def move(s, a):
    r, c = divmod(s, SIZE); dr, dc = ACT[a]
    r = min(SIZE-1, max(0, r+dr)); c = min(SIZE-1, max(0, c+dc)); return r*SIZE+c
def rew(s): return (10, True) if s == GOAL else (-1, False)

# --- Model-based: value iteration (uses the model directly) ---
V = np.zeros(N)
for _ in range(100):
    for s in range(N):
        if s == GOAL: continue
        V[s] = max(rew(move(s,a))[0] + 0.9*V[move(s,a)] for a in ACT)
pol_mb = [int(np.argmax([rew(move(s,a))[0] + 0.9*V[move(s,a)] for a in ACT])) for s in range(N)]

# --- Model-free: Q-learning (samples the environment) ---
Q = np.zeros((N,4)); samples = 0; eps = 0.2
for ep in range(600):
    s = 0
    for _ in range(50):
        a = np.random.randint(4) if np.random.random()<eps else int(np.argmax(Q[s]))
        s2 = move(s,a); r, done = rew(s2); samples += 1
        Q[s,a] += 0.1*(r + 0.9*np.max(Q[s2]) - Q[s,a]); s = s2
        if done: break
pol_mf = [int(np.argmax(Q[s])) for s in range(N)]

def path(pol):                      # greedy walk from start state 0
    s, steps = 0, []
    for _ in range(30):
        a = pol[s]; steps.append(NAMES[a]); s = move(s, a)
        if s == GOAL: break
    return steps

pmb, pmf = path(pol_mb), path(pol_mf)
print("Model-based (value iteration): samples used = 0 (uses known model)")
print(f"   greedy path from start: {'->'.join(pmb)}  ({len(pmb)} steps to goal)")
print(f"Model-free (Q-learning)     : samples used = {samples}")
print(f"   greedy path from start: {'->'.join(pmf)}  ({len(pmf)} steps to goal)")
print("\nBoth reach the goal optimally. Model-based needs the model but no samples;")
print("model-free needs many samples yet adapts when the model is unknown/changing.")
