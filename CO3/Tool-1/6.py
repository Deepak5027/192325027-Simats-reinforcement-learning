# Challenge 6: Challenges of RL in robotics - safety, sample efficiency,
# and environment variability. We train Q-learning on a grid, then test the
# learned policy under increasing transition noise (slip probability) and
# count unsafe-cell entries, showing how variability degrades performance.
import numpy as np
np.random.seed(0)
SIZE, N = 5, 25
GOAL, HAZARD = 24, {12, 17}          # hazard = unsafe cells (safety concern)
ACT = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}

def move(s, a):
    r, c = divmod(s, SIZE); dr, dc = ACT[a]
    r = min(SIZE-1, max(0, r+dr)); c = min(SIZE-1, max(0, c+dc))
    return r*SIZE + c

def reward(s):
    if s == GOAL: return 20, True
    if s in HAZARD: return -20, False
    return -1, False

def train(episodes):
    Q = np.zeros((N, 4)); eps = 0.2
    for _ in range(episodes):
        s = 0
        for _ in range(60):
            a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[s]))
            s2 = move(s, a); r, done = reward(s2)
            Q[s, a] += 0.1 * (r + 0.95*np.max(Q[s2]) - Q[s, a]); s = s2
            if done or s in HAZARD: break
    return Q

def test(Q, slip):
    succ = viol = 0
    for _ in range(500):
        s = 0
        for _ in range(60):
            a = int(np.argmax(Q[s]))
            if np.random.random() < slip: a = np.random.randint(4)   # variability
            s = move(s, a)
            if s in HAZARD: viol += 1; break
            if s == GOAL: succ += 1; break
    return succ/500, viol/500

for samples in [200, 2000]:
    Q = train(samples)
    print(f"\nTrained on {samples} episodes (sample efficiency):")
    for slip in [0.0, 0.1, 0.3]:
        sr, vr = test(Q, slip)
        print(f"  noise={slip:>3}:  success {sr:5.1%} | unsafe-entry rate {vr:5.1%}")
print("\nMore samples improve the policy; higher environment noise lowers success")
print("and raises unsafe entries - the core robotics challenges.")
