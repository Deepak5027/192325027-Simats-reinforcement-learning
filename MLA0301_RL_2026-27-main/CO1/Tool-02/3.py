# Qn 3) Agricultural irrigation/fertilizer - RL framework with DELAYED reward.
# State  : soil-moisture bucket (0=dry .. 4=saturated) on each day of season.
# Action : water amount 0/1/2.  Reward is sparse: paid only at harvest (delayed),
#          proportional to accumulated crop health minus water cost.
import numpy as np
np.random.seed(2)

DAYS, LEVELS, ACTIONS = 12, 5, 3
alpha, gamma, epsilon = 0.2, 0.95, 0.2
Q = np.zeros((DAYS, LEVELS, ACTIONS))

def transition(level, water):
    dry = np.random.choice([0, 1])                 # weather evaporation
    level = int(np.clip(level + water - dry, 0, LEVELS - 1))
    ideal = 2                                        # crops thrive near mid moisture
    health = 1.0 - abs(level - ideal) / 2.0         # 1 best, lower when too dry/wet
    return level, health, water

for ep in range(20000):
    level = np.random.randint(LEVELS)
    health_sum, water_sum, traj = 0, 0, []
    for d in range(DAYS):
        a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[d, level]))
        nlevel, health, water = transition(level, a)
        traj.append((d, level, a))
        health_sum += health; water_sum += water
        level = nlevel
    yield_reward = health_sum - 0.3 * water_sum      # DELAYED: only known at harvest
    # propagate the single terminal reward back through the season
    for i, (d, lv, a) in enumerate(reversed(traj)):
        target = yield_reward if i == 0 else gamma * np.max(Q[d + 1, traj[len(traj) - i][1]])
        Q[d, lv, a] += alpha * (target - Q[d, lv, a])

# Evaluate greedy policy
level, health_sum, water_sum = 2, 0, 0
for d in range(DAYS):
    a = int(np.argmax(Q[d, level]))
    level, health, water = transition(level, a)
    health_sum += health; water_sum += water
print(f"Season length         : {DAYS} days")
print(f"Accumulated crop health: {health_sum:.2f}")
print(f"Total water used       : {water_sum}")
print(f"Harvest reward (yield) : {health_sum - 0.3 * water_sum:.2f}")
print("Note: reward is delayed to harvest; Q-learning credits early-season watering.")
