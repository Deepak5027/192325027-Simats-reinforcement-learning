# Qn 6) Smart home HVAC - RL framework with CONFLICTING objectives (comfort vs
# energy) handled through REWARD SHAPING.
# State  : (temp deviation from setpoint bucket, occupancy).  Action: Heat/Cool/Off.
import numpy as np
np.random.seed(5)

TEMPS, OCC, ACTIONS = 5, 2, 3     # temp buckets 0..4 (2=comfortable), occ 0/1
alpha, gamma, epsilon = 0.15, 0.9, 0.15

def transition(temp, act):
    if act == 0: temp += 1        # heat
    elif act == 1: temp -= 1      # cool
    temp += np.random.choice([-1, 0, 1])   # outside weather drift
    return int(np.clip(temp, 0, TEMPS - 1))

def train(w_energy):
    Q = np.zeros((TEMPS, OCC, ACTIONS))
    for _ in range(30000):
        temp, occ = np.random.randint(TEMPS), np.random.randint(OCC)
        a = np.random.randint(ACTIONS) if np.random.random() < epsilon else int(np.argmax(Q[temp, occ]))
        ntemp = transition(temp, a)
        comfort = -abs(ntemp - 2) * occ          # discomfort only matters if occupied
        energy = -1 if a != 2 else 0             # running HVAC costs energy
        r = comfort + w_energy * energy          # SHAPED trade-off
        Q[temp, occ, a] += alpha * (r + gamma * np.max(Q[ntemp, occ]) - Q[temp, occ, a])
    return Q

for w in [0.2, 1.0, 3.0]:
    Q = train(w)
    # policy when occupied & too cold (temp=0) vs empty house (occ=0)
    act = ["Heat", "Cool", "Off"]
    occ_cold = act[int(np.argmax(Q[0, 1]))]
    empty = act[int(np.argmax(Q[0, 0]))]
    print(f"energy weight {w:<3} -> occupied&cold: {occ_cold:<4} | empty house: {empty}")
print("\nLow energy weight favours comfort (always heats);")
print("high energy weight favours saving (stays Off when the house is empty).")
