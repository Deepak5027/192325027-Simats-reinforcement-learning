# Challenge 2: Model a decision problem as a Markov Decision Process (MDP).
# Components: state space S, action space A, transition probability P,
# reward function R, discount factor gamma. We evaluate a fixed policy by
# solving the linear system V = R + gamma * P @ V  ->  V = (I - gamma P)^-1 R.
import numpy as np
np.random.seed(0)

S = ["Sunny", "Rainy"]           # state space
A = ["Walk", "Shop"]             # action space
gamma = 0.9                       # discount factor

# Transition probabilities P[s][a] -> distribution over next states
P = {
 "Sunny": {"Walk": [0.8, 0.2], "Shop": [0.6, 0.4]},
 "Rainy": {"Walk": [0.4, 0.6], "Shop": [0.5, 0.5]},
}
# Reward function R[s][a]
R = {
 "Sunny": {"Walk": 5, "Shop": 2},
 "Rainy": {"Walk": -2, "Shop": 3},
}
policy = {"Sunny": "Walk", "Rainy": "Shop"}   # fixed policy to evaluate

Pmat = np.array([P[s][policy[s]] for s in S])
Rvec = np.array([R[s][policy[s]] for s in S], dtype=float)
V = np.linalg.solve(np.eye(len(S)) - gamma * Pmat, Rvec)

print("Action space :", A)
print("Discount gamma:", gamma)
print("Transition matrix under policy:\n", Pmat)
print("Reward vector under policy:", Rvec)
for s, v in zip(S, V):
    print(f"V({s}) = {v:.3f}")
