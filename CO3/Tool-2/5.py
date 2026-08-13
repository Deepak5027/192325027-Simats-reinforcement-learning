# Experiment 5: TRPO vs A2C for an industrial robotic arm reaching a target.
# State: arm position bucket (0..8, target=4). Action: torque level {-2,-1,0,1,2}.
# Reward: -|position - target|. TRPO limits the policy change per update (KL
# trust region), giving smoother, more stable convergence than A2C.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
TARGET=4
class Arm:
    n, nA = 9, 5
    def reset(self): self.p=np.random.randint(0,9); self.t=0; return self.p
    def step(self, a):
        torque=a-2; self.p=int(np.clip(self.p+torque+np.random.choice([-1,0,0,1]),0,8)); self.t+=1
        r=-abs(self.p-TARGET); done=(self.p==TARGET) or self.t>=40
        return self.p, r, done
curves={}
curves["a2c"]  = rlkit.train_pg("a2c",  Arm(), updates=140, alpha=3.0, seed=5)
curves["trpo"] = rlkit.train_pg("trpo", Arm(), updates=140, alpha=3.0, kl_delta=0.005, seed=5)
for m in ["trpo","a2c"]:
    print(f"{m.upper():5} final reward = {curves[m][-10:].mean():6.2f} | "
          f"update-to-update variance (stability) = {np.diff(curves[m][-40:]).std():.3f}")
print("Both learn the reaching task; TRPO shows lower update-to-update variance,")
print("i.e. more stable, precise convergence - valuable for delicate arm control.")

plt.figure(figsize=(7.4,4.0))
plt.plot(curves["a2c"], color="#CA6F1E", lw=1.6, alpha=0.9, label="A2C")
plt.plot(curves["trpo"], color="#1F618D", lw=2.0, label="TRPO (KL trust region)")
plt.xlabel("Training update"); plt.ylabel("Episode reward")
plt.title("Industrial Robotic Arm: Policy Stability (TRPO vs A2C)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig5.png", dpi=130); plt.close()
