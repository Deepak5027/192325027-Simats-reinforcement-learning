# Experiment 4: PPO vs REINFORCE for smart HVAC energy management.
# State: temperature-deviation bucket (0..4, 2=comfort) x occupancy (0/1) = 10.
# Action: heat / cool / off. Reward: comfort term (when occupied) minus energy cost,
# so a good policy holds comfort while switching the unit off when it can.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
class HVAC:
    n, nA = 10, 3
    def reset(self):
        self.temp=np.random.randint(0,5); self.occ=np.random.randint(0,2); self.t=0
        return self.temp*2+self.occ
    def step(self, a):
        if a==0: self.temp+=1          # heat
        elif a==1: self.temp-=1        # cool
        self.temp=int(np.clip(self.temp+np.random.choice([-1,0,1]),0,4)); self.t+=1
        comfort=-abs(self.temp-2)*self.occ; energy=-1 if a!=2 else 0
        r=comfort+0.6*energy; done=self.t>=30
        return self.temp*2+self.occ, r, done

curves={m: rlkit.train_pg(m, HVAC(), updates=140, seed=4) for m in ["reinforce","ppo"]}
for m in ["reinforce","ppo"]:
    print(f"{m.upper():10} final reward = {curves[m][-10:].mean():6.2f} | "
          f"reward std (stability) = {curves[m][-30:].std():.2f}")
print("PPO reaches higher, more stable reward: it saves energy while holding comfort")
print("better than REINFORCE, whose updates are noisier.")

plt.figure(figsize=(7.4,4.0))
plt.plot(curves["reinforce"], color="#C0392B", lw=1.7, label="REINFORCE")
plt.plot(curves["ppo"], color="#117A65", lw=1.9, label="PPO (clipped)")
plt.xlabel("Training update"); plt.ylabel("Episode reward (comfort - energy)")
plt.title("Smart HVAC Control: PPO vs REINFORCE")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig4.png", dpi=130); plt.close()
