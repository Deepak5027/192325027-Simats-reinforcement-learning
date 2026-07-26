# Qn 4) Cybersecurity intrusion detection - states/environment, actions, reward,
# with SPARSE rewards (attacks are rare) and real-time decisions.
# State  : discretised traffic feature (connection-rate bucket).
# Action : Allow / Block.  Reward: correct decisions rewarded, misses punished hard.
import numpy as np
np.random.seed(3)

BUCKETS, ACTIONS = 6, 2          # action 0=Allow, 1=Block
alpha, gamma, epsilon = 0.1, 0.9, 0.1
Q = np.zeros((BUCKETS, ACTIONS))

def sample_packet():
    attack = np.random.random() < 0.05          # sparse: only 5% are attacks
    # attacks skew toward high connection-rate buckets
    bucket = np.random.randint(3, BUCKETS) if attack else np.random.randint(0, 4)
    return bucket, attack

def reward(action, attack):
    if attack and action == 1:   return 10       # true positive (rare, sparse)
    if attack and action == 0:   return -20      # missed attack: worst case
    if not attack and action == 1: return -2     # false positive: blocks good traffic
    return 1                                      # true negative

tp = fp = fn = tn = 0
for t in range(60000):
    b, attack = sample_packet()
    a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[b]))
    r = reward(a, attack)
    Q[b, a] += alpha * (r + gamma * np.max(Q[b]) - Q[b, a])

# Evaluate greedy detector
for _ in range(20000):
    b, attack = sample_packet()
    a = int(np.argmax(Q[b]))
    if attack and a == 1: tp += 1
    elif attack and a == 0: fn += 1
    elif not attack and a == 1: fp += 1
    else: tn += 1
prec = tp / (tp + fp) if tp + fp else 0
rec = tp / (tp + fn) if tp + fn else 0
print(f"True Positives : {tp}   False Negatives (missed): {fn}")
print(f"False Positives: {fp}   True Negatives          : {tn}")
print(f"Precision      : {prec:.2f}")
print(f"Recall (detection rate): {rec:.2f}")
print("Despite sparse attack signal, RL learns to block high-rate traffic.")
