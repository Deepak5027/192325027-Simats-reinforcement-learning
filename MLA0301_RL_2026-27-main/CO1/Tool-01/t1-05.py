import numpy as np

SIZE = 4
N = SIZE * SIZE
ACTIONS = [0, 1, 2, 3]  # Up, Down, Left, Right
TERMINAL = [0, N - 1]
REWARD = -1


def move(state, action):
    if state in TERMINAL:
        return state
    row, col = divmod(state, SIZE)
    if action == 0:
        row = max(row - 1, 0)
    elif action == 1:
        row = min(row + 1, SIZE - 1)
    elif action == 2:
        col = max(col - 1, 0)
    elif action == 3:
        col = min(col + 1, SIZE - 1)
    return row * SIZE + col


def policy_evaluation(gamma=1.0, theta=1e-4):
    V = np.zeros(N)
    iterations = 0
    while True:
        delta = 0
        for s in range(N):
            if s in TERMINAL:
                continue
            v = V[s]
            V[s] = np.mean([REWARD + gamma * V[move(s, a)] for a in ACTIONS])
            delta = max(delta, abs(v - V[s]))
        iterations += 1
        if delta < theta:
            break
    print(f"Policy Evaluation converged in {iterations} iterations.")
    return V


def value_iteration(gamma=1.0, theta=1e-4):
    V = np.zeros(N)
    iterations = 0
    while True:
        delta = 0
        for s in range(N):
            if s in TERMINAL:
                continue
            v = V[s]
            V[s] = max(REWARD + gamma * V[move(s, a)] for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
        iterations += 1
        if delta < theta:
            break
    print(f"Value Iteration converged in {iterations} iterations.")
    return V


V_pi = policy_evaluation()
print("V(s) for random policy:")
print(np.round(V_pi.reshape(SIZE, SIZE), 2), "\n")

V_star = value_iteration()
print("Optimal V*(s):")
print(np.round(V_star.reshape(SIZE, SIZE), 2), "\n")

symbols = {0: "U", 1: "D", 2: "L", 3: "R"}
print("Optimal policy:")
for s in range(N):
    a = int(np.argmax([REWARD + V_star[move(s, a)] for a in ACTIONS]))
    end = "\n" if s % SIZE == SIZE - 1 else "  "
    print("*" if s in TERMINAL else symbols[a], end=end)
