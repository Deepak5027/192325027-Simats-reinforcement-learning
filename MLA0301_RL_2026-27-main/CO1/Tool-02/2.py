# Qn 2) Streaming recommender - state, action space, reward + exploration/exploitation.
# State  : user's recently preferred genre (context).  Action: genre to recommend.
# Reward : watch-time proxy (1 = watched fully) drawn from hidden user affinity.
import numpy as np
np.random.seed(1)

GENRES = ["Action", "Comedy", "Drama", "SciFi", "Docu"]
n = len(GENRES)
# Hidden affinity of each context-user toward each recommended genre.
affinity = np.array([
    [0.8, 0.2, 0.3, 0.6, 0.1],
    [0.2, 0.9, 0.4, 0.1, 0.2],
    [0.3, 0.4, 0.8, 0.3, 0.5],
    [0.6, 0.1, 0.2, 0.9, 0.2],
    [0.1, 0.3, 0.6, 0.2, 0.8],
])

def engagement(ctx, act):
    return 1 if np.random.random() < affinity[ctx, act] else 0

def run(epsilon, steps=5000):
    Q = np.zeros((n, n)); N = np.zeros((n, n)); total = 0
    for _ in range(steps):
        ctx = np.random.randint(n)                     # current user context
        act = np.random.randint(n) if np.random.random() < epsilon else int(np.argmax(Q[ctx]))
        r = engagement(ctx, act)
        N[ctx, act] += 1
        Q[ctx, act] += (r - Q[ctx, act]) / N[ctx, act]
        total += r
    # policy: best genre learned for each context
    policy = [GENRES[int(np.argmax(Q[c]))] for c in range(n)]
    return total / steps, policy

print("Exploration vs exploitation (avg engagement per recommendation):")
for eps in [0.0, 0.1, 0.3, 0.5]:
    rate, policy = run(eps)
    tag = "pure exploit" if eps == 0 else f"eps={eps}"
    print(f"  {tag:<13} -> {rate:.3f}")
_, policy = run(0.1)
print("\nLearned policy (context -> recommended genre):")
for c, g in zip(GENRES, policy):
    print(f"  liked {c:<7} -> recommend {g}")
