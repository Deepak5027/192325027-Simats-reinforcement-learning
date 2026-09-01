# Qn 10) Smart grid - MDP for electricity distribution & peak-load reduction.
# State  : (time-of-day block, stored/deferrable load bucket).
# Action : how much deferrable load to serve now (0/1/2 units).
# Reward : meet demand (reliability) while penalising peak-hour consumption.
import numpy as np
np.random.seed(9)

BLOCKS, LOADQ, ACTIONS = 4, 4, 3     # 4 time blocks, backlog 0..3, serve 0..2
alpha, gamma, epsilon = 0.15, 0.95, 0.15
Q = np.zeros((BLOCKS, LOADQ, ACTIONS))
PEAK = 2                              # block index 2 is peak demand

def train():
    for _ in range(40000):
        block, backlog = 0, np.random.randint(LOADQ)
        for _ in range(BLOCKS):
            a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[block, backlog]))
            served = min(a, backlog)
            new_demand = np.random.randint(0, 2)
            nbacklog = int(np.clip(backlog - served + new_demand, 0, LOADQ - 1))
            reliability = served                          # serving load is good
            peak_cost = 2 * served if block == PEAK else 0.2 * served
            unmet = 0.5 * nbacklog                         # backlog risks blackout
            r = reliability - peak_cost - unmet
            nblock = (block + 1) % BLOCKS
            Q[block, backlog, a] += alpha * (r + gamma * np.max(Q[nblock, nbacklog]) - Q[block, backlog, a])
            block, backlog = nblock, nbacklog

train()
print("Learned load-shifting policy (units served) by time block & backlog:")
label = ["Night", "Morning", "PEAK", "Evening"]
for b in range(BLOCKS):
    row = [int(np.argmax(Q[b, q])) for q in range(LOADQ)]
    print(f"  {label[b]:<8} backlog0..3 -> serve {row}")
print("\nThe agent defers load away from the PEAK block and catches up in")
print("off-peak blocks - flattening peak demand while keeping backlog low.")
