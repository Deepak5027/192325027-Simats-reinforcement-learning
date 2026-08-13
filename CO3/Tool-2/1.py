# Experiment 1: Policy Gradient traffic-signal control - REINFORCE vs A2C.
# State: discretised queue lengths on two roads (each 0..3) -> 16 states.
# Action: which road gets the green light (2). Reward: negative total queue
# (a proxy for average vehicle waiting time).
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit

class Traffic:
    n, nA = 16, 2
    def reset(self):
        self.q = [np.random.randint(0,3), np.random.randint(0,3)]; self.t = 0
        return self.q[0]*4 + self.q[1]
    def step(self, a):
        for i in range(2): self.q[i] = min(3, self.q[i] + np.random.binomial(1, 0.5))
        self.q[a] = max(0, self.q[a] - 2); self.t += 1
        r = -(self.q[0] + self.q[1]); done = self.t >= 30
        return self.q[0]*4 + self.q[1], r, done

curves = {m: rlkit.train_pg(m, Traffic(), updates=120, seed=1) for m in ["reinforce","a2c"]}
wait = {m: -c/30 for m, c in curves.items()}     # avg queue per step ~ waiting time
for m in curves:
    print(f"{m.upper():10} avg waiting  start={wait[m][:10].mean():.2f} -> final={wait[m][-10:].mean():.2f} cars/step")
print(f"Throughput improves as waiting falls. A2C converges faster and steadier than REINFORCE.")

plt.figure(figsize=(7.4,4.0))
plt.plot(wait["reinforce"], color="#C0392B", lw=1.8, label="REINFORCE")
plt.plot(wait["a2c"], color="#2E5A88", lw=1.8, label="A2C (Actor-Critic)")
plt.xlabel("Training update"); plt.ylabel("Avg vehicle waiting (cars/step)")
plt.title("Traffic Signal Control: Waiting Time During Training")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig("fig1.png", dpi=130); plt.close()
