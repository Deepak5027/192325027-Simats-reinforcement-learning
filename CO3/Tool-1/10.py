# Challenge 10: On-policy (SARSA) vs off-policy (Q-learning).
# Cliff-walking style grid: a bottom-row "cliff" gives -100 and resets.
# Q-learning learns the optimal (risky, cliff-edge) path; SARSA learns a
# safer path because it accounts for exploratory moves. We compare returns.
import numpy as np
np.random.seed(0)
ROWS, COLS = 4, 12
START, GOAL = (3,0), (3,11)
CLIFF = {(3,c) for c in range(1,11)}
ACT = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}

def step(s, a):
    r, c = s; dr, dc = ACT[a]
    r = min(ROWS-1, max(0, r+dr)); c = min(COLS-1, max(0, c+dc)); ns = (r,c)
    if ns in CLIFF: return START, -100, False
    if ns == GOAL:  return ns, 0, True
    return ns, -1, False

def egreedy(Q, s, eps):
    return np.random.randint(4) if np.random.random()<eps else int(np.argmax(Q[s]))

def train(method, episodes=500, alpha=0.5, gamma=1.0, eps=0.1):
    Q = np.zeros((ROWS,COLS,4))
    for _ in range(episodes):
        s = START; a = egreedy(Q, s, eps)
        for _ in range(200):
            s2, r, done = step(s, a); a2 = egreedy(Q, s2, eps)
            if method == "sarsa":
                target = r + gamma*Q[s2][a2]
            else:  # q-learning
                target = r + gamma*np.max(Q[s2])
            Q[s][a] += alpha*(target - Q[s][a]); s, a = s2, a2
            if done: break
    return Q

def greedy_return(Q):
    s = START; total = 0
    for _ in range(200):
        a = int(np.argmax(Q[s])); s, r, done = step(s, a); total += r
        if done: break
    return total

for m in ["sarsa", "q-learning"]:
    Q = train(m)
    print(f"{m:<11}: greedy-path return = {greedy_return(Q)}")
print("\nQ-learning (off-policy) finds the optimal cliff-edge path (higher return");
print("but risky). SARSA (on-policy) prefers a safer path away from the cliff.")
