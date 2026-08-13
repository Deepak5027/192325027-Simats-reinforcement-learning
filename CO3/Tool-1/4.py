# Challenge 4: Exploration-exploitation trade-off.
# 10-armed bandit. Compare epsilon-greedy and softmax action selection.
import numpy as np
np.random.seed(1)
K, STEPS = 10, 2000
q_true = np.random.normal(0, 1, K)

def eps_greedy(eps):
    Q = np.zeros(K); Nc = np.zeros(K); total = 0
    for _ in range(STEPS):
        a = np.random.randint(K) if np.random.random() < eps else int(np.argmax(Q))
        r = np.random.normal(q_true[a], 1); Nc[a]+=1; Q[a]+=(r-Q[a])/Nc[a]; total+=r
    return total/STEPS, int(np.argmax(Q))

def softmax(tau):
    Q = np.zeros(K); Nc = np.zeros(K); total = 0
    for _ in range(STEPS):
        p = np.exp(Q/tau); p/=p.sum()
        a = np.random.choice(K, p=p)
        r = np.random.normal(q_true[a], 1); Nc[a]+=1; Q[a]+=(r-Q[a])/Nc[a]; total+=r
    return total/STEPS, int(np.argmax(Q))

print(f"Optimal arm = {int(np.argmax(q_true))}  (true value {q_true.max():.2f})\n")
print("epsilon-greedy:")
for e in [0.0, 0.1, 0.3]:
    avg, best = eps_greedy(e); print(f"  eps={e}: avg reward {avg:.3f}, arm chosen {best}")
print("softmax:")
for t in [0.1, 0.5, 1.0]:
    avg, best = softmax(t); print(f"  tau={t}: avg reward {avg:.3f}, arm chosen {best}")
