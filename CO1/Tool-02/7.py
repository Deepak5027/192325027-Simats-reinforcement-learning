# Qn 7) Strategy-game agent - model as MDP, reward for LONG-TERM strategy,
# and effect of exploration. Game: Nim (21 sticks, take 1-3, taker of last loses).
import numpy as np
np.random.seed(6)

alpha, gamma = 0.2, 0.95
def train(epsilon, games=30000):
    Q = np.zeros((22, 4))          # Q[sticks_left, take(1..3)]
    for _ in range(games):
        sticks, history = 21, []
        while sticks > 0:
            # agent move
            valid = [a for a in (1, 2, 3) if a <= sticks]
            a = np.random.choice(valid) if np.random.random() < epsilon else \
                max(valid, key=lambda x: Q[sticks, x])
            history.append((sticks, a))
            sticks -= a
            if sticks == 0:                      # agent took last -> agent loses
                for i, (s, act) in enumerate(reversed(history)):
                    tgt = -1 if i == 0 else gamma * np.max(Q[s - act])
                    Q[s, act] += alpha * (tgt - Q[s, act])
                break
            # opponent (random) move
            ov = [a for a in (1, 2, 3) if a <= sticks]
            sticks -= np.random.choice(ov)
            if sticks == 0:                      # opponent took last -> agent wins
                for i, (s, act) in enumerate(reversed(history)):
                    tgt = 1 if i == 0 else gamma * np.max(Q[s - act])
                    Q[s, act] += alpha * (tgt - Q[s, act])
                break
    return Q

def win_rate(Q, games=5000):
    wins = 0
    for _ in range(games):
        sticks = 21
        while sticks > 0:
            valid = [a for a in (1, 2, 3) if a <= sticks]
            sticks -= max(valid, key=lambda x: Q[sticks, x])
            if sticks == 0: break
            ov = [a for a in (1, 2, 3) if a <= sticks]
            sticks -= np.random.choice(ov)
            if sticks == 0: wins += 1; break
    return wins / games

print("Effect of exploration on long-term strategy (win rate vs random opponent):")
for eps in [0.05, 0.2, 0.5]:
    print(f"  exploration eps={eps} -> win rate {win_rate(train(eps)):.2%}")
print("Exploration is essential to discover the winning 'leave 4k+1 sticks'")
print("positions; once found, a low residual eps exploits it best, while very")
print("high eps keeps making random (losing) moves during play.")
