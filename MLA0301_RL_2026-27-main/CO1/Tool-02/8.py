# Qn 8) Fraud detection - state/action/reward with IMBALANCED data, delayed
# feedback, and asymmetric risk of wrong decisions.
# State  : transaction risk-score bucket.  Action: Approve / Flag.
import numpy as np
np.random.seed(7)

BUCKETS, ACTIONS = 8, 2           # 0=Approve, 1=Flag
alpha, gamma, epsilon = 0.1, 0.9, 0.1
Q = np.zeros((BUCKETS, ACTIONS))

def sample_txn():
    fraud = np.random.random() < 0.02            # imbalanced: 2% fraud
    score = np.random.randint(4, BUCKETS) if fraud else np.random.randint(0, 5)
    return score, fraud

def reward(action, fraud):
    # asymmetric costs: missing fraud is far worse than a false alarm
    if fraud and action == 1:   return 20        # caught fraud
    if fraud and action == 0:   return -50       # missed fraud (chargeback + risk)
    if not fraud and action == 1: return -3      # false alarm annoys customer
    return 1                                      # correctly approved

for t in range(80000):
    s, fraud = sample_txn()
    a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[s]))
    r = reward(a, fraud)
    Q[s, a] += alpha * (r + gamma * np.max(Q[s]) - Q[s, a])

tp = fp = fn = tn = 0
for _ in range(50000):
    s, fraud = sample_txn()
    a = int(np.argmax(Q[s]))
    if fraud and a == 1: tp += 1
    elif fraud and a == 0: fn += 1
    elif not fraud and a == 1: fp += 1
    else: tn += 1
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
print(f"Fraud caught (TP): {tp}   Missed (FN): {fn}")
print(f"False alarms (FP): {fp}   Correct approvals (TN): {tn}")
print(f"Precision: {prec:.2f}   Recall: {rec:.2f}")
print("Asymmetric reward makes the agent flag aggressively on high-risk scores,")
print("trading some precision for high recall - the right call when misses cost most.")
