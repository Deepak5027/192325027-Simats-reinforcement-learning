# Challenge 7: Influence of the discount factor (gamma) on short- vs
# long-term decisions. A 2-action choice: a small immediate reward now, or
# a path that yields a large reward later. Value iteration for several gamma.
import numpy as np
# States: 0=start, 1..3 = path to big reward, 4=terminal-big, 5=terminal-small
# Action A ("greedy now"): start->small terminal, reward +2 immediately.
# Action B ("patient")   : start->1->2->3->big terminal, reward +10 at the end.
def value_of_patient(gamma):   # +10 received after 4 steps
    return gamma**4 * 10
def value_of_greedy(gamma):    # +2 received after 1 step
    return gamma * 2

print("gamma | greedy-now value | patient value | optimal choice")
for g in [0.1, 0.5, 0.9, 0.99]:
    vg, vp = value_of_greedy(g), value_of_patient(g)
    choice = "Patient (long-term)" if vp > vg else "Greedy (short-term)"
    print(f" {g:<4} |      {vg:6.3f}      |   {vp:6.3f}     | {choice}")

print("\nReturn of a fixed reward sequence [1,1,1,1,1] under different gamma (episodic):")
rews = [1,1,1,1,1]
for g in [0.1, 0.5, 0.9, 1.0]:
    G = sum((g**t)*r for t, r in enumerate(rews))
    print(f"  gamma={g}: return = {G:.3f}")
print("Low gamma favours immediate reward; high gamma values long-term outcomes.")
