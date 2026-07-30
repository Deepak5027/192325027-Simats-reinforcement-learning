# Qn 1) Warehouse robot - MDP + reward balancing speed, accuracy, and safety.
# State  : (row, col) grid cell.  Action: Up/Down/Left/Right + Pick.
# Reward : -1 per step (speed), -10 hitting an obstacle (safety),
#          +5 correct pick at item, +20 delivery at goal (accuracy).
import numpy as np
import random

random.seed(0); np.random.seed(0)

SIZE = 5
GOAL = (4, 4)          # packing station
ITEM = (0, 4)          # item to pick
OBST = {(2, 2), (1, 3), (3, 1)}
ACTIONS = ["Up", "Down", "Left", "Right", "Pick"]
alpha, gamma, epsilon = 0.1, 0.9, 0.2

# Q[row, col, has_item, action]
Q = np.zeros((SIZE, SIZE, 2, len(ACTIONS)))

def step(pos, has_item, a):
    r, c = pos
    reward, done = -1, False            # speed: every move costs time
    if ACTIONS[a] == "Pick":
        if pos == ITEM and not has_item:
            return pos, 1, 5, False      # accuracy: correct pick
        return pos, has_item, -2, False  # wrong pick wastes time
    if a == 0: r -= 1
    elif a == 1: r += 1
    elif a == 2: c -= 1
    elif a == 3: c += 1
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        return pos, has_item, -5, False  # bumping a wall
    if (r, c) in OBST:
        return pos, has_item, -10, False # safety: collision, stay put
    npos = (r, c)
    if npos == GOAL and has_item:
        reward += 20                     # accuracy: successful delivery
        done = True
    return npos, has_item, reward, done

for ep in range(4000):
    pos, has_item = (0, 0), 0
    for _ in range(60):
        s = (pos[0], pos[1], has_item)
        a = random.randrange(len(ACTIONS)) if random.random() < epsilon else int(np.argmax(Q[s]))
        npos, nhas, rwd, done = step(pos, has_item, a)
        ns = (npos[0], npos[1], nhas)
        Q[s][a] += alpha * (rwd + gamma * np.max(Q[ns]) - Q[s][a])
        pos, has_item = npos, nhas
        if done: break

# Greedy roll-out of the learned policy
pos, has_item, total, path = (0, 0), 0, 0, [(0, 0)]
for _ in range(40):
    s = (pos[0], pos[1], has_item)
    a = int(np.argmax(Q[s]))
    pos, has_item, rwd, done = step(pos, has_item, a)
    total += rwd
    path.append(pos if ACTIONS[a] != "Pick" else f"PICK@{pos}")
    if done: break

print("Learned delivery path:", path)
print(f"Steps taken     : {len(path) - 1}")
print(f"Total reward    : {total}")
print(f"Item collected  : {bool(has_item)}   Delivered: {pos == GOAL and has_item}")
