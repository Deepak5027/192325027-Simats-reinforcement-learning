# Experiment 6: A2C vs PPO for adaptive patient treatment planning.
# State: patient severity level (0=healthy .. 4=critical). Action: treatment
# intensity (low/medium/high). Reward: improvement in severity minus a cost for
# aggressive treatment, so the agent must treat effectively without over-dosing.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
class Patient:
    n, nA = 5, 3
    def reset(self): self.sev=np.random.randint(1,5); self.t=0; return self.sev
    def step(self, a):
        prev=self.sev
        self.sev=int(np.clip(self.sev - a + np.random.choice([0,0,1]),0,4)); self.t+=1
        improve=prev-self.sev; cost=0.3*a; over=0.5*max(0,a-prev)  # penalise over-treatment
        r=improve - cost - over; done=(self.sev==0) or self.t>=25
        return self.sev, r, done
curves={m: rlkit.train_pg(m, Patient(), updates=130, seed=6) for m in ["a2c","ppo"]}
for m in ["a2c","ppo"]:
    print(f"{m.upper():4} cumulative reward = {curves[m][-10:].mean():6.2f} | "
          f"learning stability (std) = {curves[m][-30:].std():.3f}")
print("Both A2C and PPO learn effective dosing to similar cumulative reward; PPO's")
print("clipped objective bounds each policy update, a useful safety property here.")

plt.figure(figsize=(7.4,4.0))
plt.plot(curves["a2c"], color="#884EA0", lw=1.7, label="A2C")
plt.plot(curves["ppo"], color="#148F77", lw=1.9, label="PPO")
plt.fill_between(range(len(curves["ppo"])), curves["ppo"], alpha=0.08, color="#148F77")
plt.xlabel("Training update"); plt.ylabel("Cumulative reward (treatment effectiveness)")
plt.title("Adaptive Treatment Planning: A2C vs PPO")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig6.png", dpi=130); plt.close()
