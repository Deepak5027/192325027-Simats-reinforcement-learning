# Qn 5) Ride-sharing driver allocation + dynamic pricing under uncertainty.
# State  : demand level (Low/Med/High) vs available supply.
# Action : price multiplier 1.0x / 1.5x / 2.0x.  Reward: expected revenue,
#          with a fairness penalty that discourages aggressive surge.
import numpy as np
np.random.seed(4)

DEMAND = ["Low", "Med", "High"]
MULT = [1.0, 1.5, 2.0]
alpha, gamma, epsilon = 0.1, 0.9, 0.15
Q = np.zeros((len(DEMAND), len(MULT)))

def env(d_idx, m_idx):
    base_riders = [20, 50, 90][d_idx]
    price = MULT[m_idx]
    # higher price -> fewer riders accept (uncertain elasticity)
    accept = base_riders * max(0.1, 1.0 - 0.35 * (price - 1.0)) * np.random.uniform(0.8, 1.2)
    revenue = accept * price
    fairness_penalty = 15 * (price - 1.0) if d_idx < 2 else 5 * (price - 1.0)
    return revenue - fairness_penalty

for t in range(40000):
    d = np.random.randint(len(DEMAND))
    a = np.random.randint(len(MULT)) if np.random.random() < epsilon else int(np.argmax(Q[d]))
    r = env(d, a)
    Q[d, a] += alpha * (r + gamma * np.max(Q[d]) - Q[d, a])

print("Learned dynamic-pricing policy under demand uncertainty:")
for d in range(len(DEMAND)):
    best = int(np.argmax(Q[d]))
    print(f"  Demand {DEMAND[d]:<4} -> price {MULT[best]}x   (Q={Q[d, best]:.0f})")
print("\nEnvironmental uncertainty (random rider acceptance) is absorbed by")
print("averaging returns; fairness penalty curbs surge when demand is low.")
