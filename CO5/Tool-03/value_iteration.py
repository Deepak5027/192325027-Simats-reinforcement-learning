# Dynamic Programming (Value Iteration) for a warehouse robot's shortest path.
# The warehouse is an 8x8 grid. Shaded cells are shelves (obstacles). The robot
# must reach the GOAL (packing station) in the fewest moves. We model this as an
# MDP with reward -1 per move and a terminal GOAL, and solve it with Value
# Iteration using the Bellman optimality equation:  V(s) = max_a [ R + gamma*V(s') ].
import numpy as np
SIZE=8; GAMMA=1.0
SHELVES={(1,1),(1,2),(1,3),(3,3),(3,4),(3,5),(5,1),(5,2),(5,5),(5,6),(2,6),(6,4)}
GOAL=(7,7); START=(0,0)
MOVES={0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}      # N,S,W,E
ARROW={0:"↑",1:"↓",2:"←",3:"→"}

def nxt(s,a):
    r,c=s[0]+MOVES[a][0], s[1]+MOVES[a][1]
    if 0<=r<SIZE and 0<=c<SIZE and (r,c) not in SHELVES: return (r,c)
    return s

def value_iteration(theta=1e-6):
    V=np.zeros((SIZE,SIZE)); deltas=[]; snap1=None
    it=0
    while True:
        delta=0; newV=V.copy()
        for r in range(SIZE):
            for c in range(SIZE):
                if (r,c)==GOAL or (r,c) in SHELVES: continue
                q=[-1 + GAMMA*V[nxt((r,c),a)] for a in MOVES]
                newV[r,c]=max(q); delta=max(delta,abs(newV[r,c]-V[r,c]))
        V=newV; deltas.append(delta); it+=1
        if it==1: snap1=V.copy()
        if delta<theta: break
    # greedy policy
    pol={}
    for r in range(SIZE):
        for c in range(SIZE):
            if (r,c)==GOAL or (r,c) in SHELVES: continue
            q=[-1+GAMMA*V[nxt((r,c),a)] for a in MOVES]
            pol[(r,c)]=int(np.argmax(q))
    return V,pol,deltas,snap1

V,pol,deltas,snap1=value_iteration()
np.save("V.npy",V); import json; json.dump({str(k):v for k,v in pol.items()},open("pol.json","w"))
np.save("deltas.npy",np.array(deltas))
print(f"Value Iteration converged in {len(deltas)} sweeps (theta=1e-6).")
print(f"V*(START={START}) = {V[START]:.0f}  ->  shortest path = {int(-V[START])} moves to goal.")
print(f"V*(goal-neighbour (7,6)) = {V[7,6]:.0f}")
print("\nAfter sweep 1, V of cells one step from the goal:")
for s in [(7,6),(6,7)]:
    print(f"  V1{ s } = {snap1[s]:.0f}   (= -1 + V0(goal) = -1 + 0)")
print("\nConverged V*(s) grid  (blank = shelf, 0 = goal):")
for r in range(SIZE):
    row=""
    for c in range(SIZE):
        if (r,c) in SHELVES: row+="   ##"
        else: row+=f"{V[r,c]:5.0f}"
    print(row)
