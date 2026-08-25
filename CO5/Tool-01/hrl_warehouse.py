# Hierarchical Reinforcement Learning (HRL) for autonomous warehouse navigation.
# The robot must PICK a package at a shelf, then DELIVER it to the packing station,
# navigating a grid with shelf obstacles. We use the OPTIONS framework (2 levels):
#   Manager (high level) : chooses a sub-goal option  {GO-TO-PICKUP, GO-TO-DROPOFF}
#   Worker  (low level)  : a goal-conditioned navigation policy that reaches a target
# and compare it against FLAT Q-learning (primitive actions only) under a sparse reward.
import numpy as np
np.random.seed(0)

SIZE = 10
SHELVES = {(2,2),(2,3),(2,4),(2,5),(5,5),(5,6),(5,7),(7,2),(7,3),(3,7),(4,7)}
START, PICKUP, DROPOFF = (0,0), (2,6), (8,8)
MOVES = [(-1,0),(1,0),(0,-1),(0,1)]           # N,S,W,E

def step_pos(pos, a):
    r,c = pos[0]+MOVES[a][0], pos[1]+MOVES[a][1]
    if 0<=r<SIZE and 0<=c<SIZE and (r,c) not in SHELVES:
        return (r,c)
    return pos                                  # blocked -> stay

# ---------------- FLAT Q-LEARNING ----------------
# state = (row, col, carrying) ; actions = 4 moves + pickup(4) + dropoff(5)
def train_flat(episodes=1500, alpha=0.2, gamma=0.97):
    Q = {}
    def q(s):
        return Q.setdefault(s, np.zeros(6))
    eps=1.0; curve=[]
    for ep in range(episodes):
        pos, carry = START, 0; total=0
        for t in range(200):
            s=(pos[0],pos[1],carry)
            a = np.random.randint(6) if np.random.random()<eps else int(np.argmax(q(s)))
            r=-1; done=False
            if a<4:
                pos=step_pos(pos,a)
            elif a==4:                          # pickup
                if pos==PICKUP and carry==0: carry=1; r=-1
                else: r=-5
            else:                               # dropoff
                if pos==DROPOFF and carry==1: r=20; done=True
                else: r=-5
            ns=(pos[0],pos[1],carry)
            q(s)[a]+=alpha*(r+gamma*np.max(q(ns))-q(s)[a]); total+=r
            if done: break
        eps=max(0.05,eps*0.997); curve.append(total)
    return np.array(curve)

# ---------------- HRL (OPTIONS) ----------------
# Worker: goal-conditioned navigation Q[target][ (r,c) ] over 4 moves, reward -1/step,
# option terminates on reaching the target cell.
def train_hrl(episodes=1500, alpha=0.2, gamma=0.97):
    targets=[PICKUP, DROPOFF]
    QW={t:{} for t in targets}
    def qw(t,pos): return QW[t].setdefault(pos, np.zeros(4))
    eps=1.0; curve=[]
    def run_option(target, pos, learn):
        steps=0
        while pos!=target and steps<80:
            a=np.random.randint(4) if (learn and np.random.random()<eps) else int(np.argmax(qw(target,pos)))
            npos=step_pos(pos,a)
            r=0.0 if npos==target else -1.0
            if learn:
                tgt = 0.0 if npos==target else gamma*np.max(qw(target,npos))
                qw(target,pos)[a]+=alpha*(r+tgt-qw(target,pos)[a])
            pos=npos; steps+=1
        return pos, steps
    for ep in range(episodes):
        # Manager policy is trivial (2 states): not carrying -> GO-TO-PICKUP, else GO-TO-DROPOFF
        pos=START; total=0
        pos,s1=run_option(PICKUP,pos,True); total+= -s1
        carry=1 if pos==PICKUP else 0
        if carry: total+= -1                    # pickup action
        pos,s2=run_option(DROPOFF,pos,True); total+= -s2
        if pos==DROPOFF and carry: total+=20    # successful dropoff
        eps=max(0.05,eps*0.997); curve.append(total)
    return np.array(curve), QW

def greedy_path(QW):
    targets=[PICKUP,DROPOFF]; path=[START]; pos=START
    for t in targets:
        steps=0
        while pos!=t and steps<80:
            a=int(np.argmax(QW[t].setdefault(pos,np.zeros(4)))); pos=step_pos(pos,a); path.append(pos); steps+=1
    return path

flat=train_flat()
hrl,QW=train_hrl()
print(f"FLAT Q-learning : first-100-ep avg return = {flat[:100].mean():7.2f} | last-100 avg = {flat[-100:].mean():7.2f}")
print(f"HRL (options)   : first-100-ep avg return = {hrl[:100].mean():7.2f} | last-100 avg = {hrl[-100:].mean():7.2f}")
path=greedy_path(QW)
print(f"Learned HRL delivery path length = {len(path)-1} steps  (start -> pickup -> dropoff)")
np.save("flat_curve.npy",flat); np.save("hrl_curve.npy",hrl)
import json; json.dump(path, open("hrl_path.json","w"))
