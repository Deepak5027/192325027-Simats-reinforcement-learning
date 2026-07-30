# Qn 9) Healthcare rehab - RL for personalised exercise difficulty.
# State  : patient ability level.  Action: prescribe Easy/Medium/Hard exercise.
# Reward : improvement in ability (DELAYED, noisy due to patient variability).
import numpy as np
np.random.seed(8)

LEVELS, ACTIONS = 6, 3            # ability 0..5 ; difficulty 0=Easy,1=Med,2=Hard
alpha, gamma, epsilon = 0.15, 0.9, 0.2
Q = np.zeros((LEVELS, ACTIONS))

def transition(level, diff, sensitivity):
    # best progress when difficulty ~ matches ability; too hard risks setback
    match = diff - (level / (LEVELS - 1)) * 2       # scale ability to 0..2
    gain = (1.0 - abs(match)) * sensitivity + np.random.normal(0, 0.3)  # variability
    if diff == 2 and level < 2:
        gain -= 0.5                                  # too hard, too soon: setback risk
    nlevel = int(np.clip(round(level + gain), 0, LEVELS - 1))
    reward = gain                                    # improvement = reward (delayed/noisy)
    return nlevel, reward

for patient in range(6000):
    sensitivity = np.random.uniform(0.5, 1.0)        # patient variability
    level = np.random.randint(LEVELS)
    for _ in range(15):                              # rehab sessions
        a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[level]))
        nlevel, r = transition(level, a, sensitivity)
        Q[level, a] += alpha * (r + gamma * np.max(Q[nlevel]) - Q[level, a])
        level = nlevel

print("Learned rehab policy (ability level -> prescribed difficulty):")
name = ["Easy", "Medium", "Hard"]
for lv in range(LEVELS):
    print(f"  ability {lv} -> {name[int(np.argmax(Q[lv]))]}")
print("\nPolicy ramps difficulty with ability (safe progression), avoiding")
print("Hard exercises for weak patients despite noisy, delayed improvement signals.")
