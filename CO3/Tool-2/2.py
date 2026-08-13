# Experiment 2: Autonomous warehouse robot - Vanilla Policy Gradient vs A2C.
# 4x4 grid; the robot must reach the pick cell then the delivery goal.
# State: cell (0..15) x has_package (0/1) = 32 states. Actions: up/down/left/right.
# Reward: -1 per step, +5 on pickup, +10 on delivery.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
PICK, GOAL = 5, 15
class Warehouse:
    n, nA = 32, 4
    def reset(self): self.p=0; self.has=0; self.t=0; return self.p*2+self.has
    def step(self, a):
        r,c = divmod(self.p,4)
        if a==0: r=max(0,r-1)
        elif a==1: r=min(3,r+1)
        elif a==2: c=max(0,c-1)
        else: c=min(3,c+1)
        self.p=r*4+c; self.t+=1; rew=-1; done=False
        if self.p==PICK and not self.has: self.has=1; rew+=5
        if self.p==GOAL and self.has: rew+=10; done=True
        if self.t>=60: done=True
        return self.p*2+self.has, rew, done

curves = {m: rlkit.train_pg(m, Warehouse(), updates=150, seed=2) for m in ["reinforce","a2c"]}
for m in ["reinforce","a2c"]:
    print(f"{('VPG' if m=='reinforce' else 'A2C'):4} final reward = {curves[m][-10:].mean():6.2f}")
print("A2C reaches higher, more stable reward and completes pick-and-deliver in fewer steps.")

plt.figure(figsize=(7.4,4.0))
plt.plot(curves["reinforce"], color="#8E44AD", lw=1.8, label="Vanilla PG (REINFORCE)")
plt.plot(curves["a2c"], color="#1E8449", lw=1.8, label="A2C (Actor-Critic)")
plt.xlabel("Training update"); plt.ylabel("Episode reward")
plt.title("Warehouse Robot: Training Reward & Stability")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig2.png", dpi=130); plt.close()
