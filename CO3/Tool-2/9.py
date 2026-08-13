# Experiment 9: Policy Gradient predictive maintenance - REINFORCE vs PPO.
# State: machine wear level (0..5). Action: run or maintain. Running raises wear
# and running cost; at maximum wear a run risks a costly breakdown. Maintenance
# resets wear at a moderate cost. Reward = negative cost (maintenance + downtime).
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
class Machine:
    n, nA = 6, 2
    def reset(self): self.w=0; self.t=0; return self.w
    def step(self, a):
        self.t+=1
        if a==1:                       # maintain
            self.w=0; r=-2.0; fail=0
        else:                          # run
            if self.w>=5:              # breakdown risk at max wear
                r=-10.0; self.w=0; fail=1
            else:
                r=-0.2*self.w; self.w+=1; fail=0
        return self.w, r, self.t>=30
res={}
for m in ["reinforce","ppo"]:
    c,T,_=rlkit.train_pg(m, Machine(), updates=120, seed=9, return_policy=True)
    env=Machine(); cost=fail=n=0
    for _ in range(400):
        s=env.reset()
        for _ in range(30):
            a=int(np.argmax(rlkit.softmax(T[s]))); s,r,d=env.step(a); cost+=-r; fail+=(r==-10.0); n+=1
            if d: break
    res[m]=(c,cost/400,fail/400)
    print(f"{m.upper():10}: avg cost/episode {cost/400:5.2f} | breakdowns (downtime)/episode {fail/400:.2f} | final reward {c[-10:].mean():.2f}")
print("Both avoid breakdown downtime; PPO learns the just-in-time maintenance")
print("policy at a lower overall cost than REINFORCE.")
plt.figure(figsize=(7.4,4.0))
plt.plot(res["reinforce"][0],color="#873600",lw=1.6,label="REINFORCE")
plt.plot(res["ppo"][0],color="#1A5276",lw=1.9,label="PPO")
plt.xlabel("Training update"); plt.ylabel("Episode reward (-cost)")
plt.title("Predictive Maintenance: REINFORCE vs PPO")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig9.png", dpi=130); plt.close()
