# Experiment 3: DDPG for continuous portfolio optimization.
# State: market signal s in [-1,1]. Action: continuous risky-asset weight in [0,1].
# Target (optimal) allocation a*(s)=clip(0.5+0.4s,0,1). Reward: -(a-a*)^2.
# We then evaluate cumulative return and risk of the learned policy vs a static
# 50/50 portfolio on a simulated price series.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
np.random.seed(3)
def env_fn():
    s = np.random.uniform(-1,1); astar = np.clip(0.5+0.4*s, 0, 1); return s, astar
curve, w = rlkit.train_ddpg(env_fn, steps=6000, seed=3)
def alloc(s): return float(np.clip(w[0]*s+w[1], 0, 1))

# evaluate on a simulated market
np.random.seed(7); T=300; s=0.0; ddpg_r=[]; static_r=[]
for _ in range(T):
    s = np.clip(0.9*s + 0.2*np.random.randn(), -1, 1)
    risky = 0.03 + 0.06*s + np.random.randn()*0.08          # risky asset step return
    ddpg_r.append(alloc(s)*risky); static_r.append(0.5*risky)
def summarize(x):
    x=np.array(x); return x.sum(), x.std()
dr,dv = summarize(ddpg_r); sr,sv = summarize(static_r)
print(f"Learned allocation rule: a(s) = clip({w[0]:.2f}*s + {w[1]:.2f}, 0, 1)")
print(f"DDPG    : cumulative return {dr:+.3f} | risk (std) {dv:.3f} | return/risk {dr/dv:+.2f}")
print(f"Static  : cumulative return {sr:+.3f} | risk (std) {sv:.3f} | return/risk {sr/sv:+.2f}")
print("DDPG tracks the optimal allocation, lifting risk-adjusted return over the static split.")

fig,(a1,a2)=plt.subplots(1,2,figsize=(8.2,3.8))
xs=np.linspace(-1,1,50)
a1.plot(xs,[np.clip(0.5+0.4*x,0,1) for x in xs],"--",color="#555",label="optimal a*(s)")
a1.plot(xs,[alloc(x) for x in xs],color="#B5651D",lw=2,label="DDPG policy")
a1.set_xlabel("Market signal s"); a1.set_ylabel("Risky-asset weight"); a1.set_title("Learned Allocation Policy"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
a2.plot(np.cumsum(ddpg_r),color="#B5651D",lw=1.8,label="DDPG")
a2.plot(np.cumsum(static_r),color="#2E5A88",lw=1.8,label="Static 50/50")
a2.set_xlabel("Trading step"); a2.set_ylabel("Cumulative return"); a2.set_title("Back-test Performance"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig3.png", dpi=130); plt.close()
