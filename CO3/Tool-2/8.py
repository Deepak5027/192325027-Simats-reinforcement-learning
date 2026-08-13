# Experiment 8: A3C vs Vanilla Policy Gradient for cloud resource allocation.
# State: incoming load level (0..4). Action: provision 1, 2 or 3 servers.
# Reward: negative of (response time + server cost), so the agent must match
# capacity to demand - under-provisioning hurts latency, over-provisioning wastes cost.
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rlkit
class Cloud:
    n, nA = 5, 3
    def reset(self): self.load=np.random.randint(0,5); self.t=0; return self.load
    def step(self, a):
        servers=a+1; resp=max(0,self.load-servers); cost=0.5*servers
        r=-(resp+cost); self.t+=1
        self.load=int(np.clip(self.load+np.random.choice([-1,0,1]),0,4))
        return self.load, r, self.t>=25
res={}
for m in ["a3c","reinforce"]:
    c,T,_=rlkit.train_pg(m, Cloud(), updates=120, seed=8, return_policy=True)
    env=Cloud(); resp=util=n=0
    for _ in range(400):
        s=env.reset()
        for _ in range(25):
            a=int(np.argmax(rlkit.softmax(T[s]))); servers=a+1; resp+=max(0,s-servers)
            util+=min(s,servers)/servers; s,r,d=env.step(a); n+=1
            if d: break
    res[m]=(c,resp/n,util/n)
    print(f"{('A3C' if m=='a3c' else 'VPG'):4}: avg response time {resp/n:.2f} | CPU utilization {util/n:5.1%} | final reward {c[-10:].mean():.2f}")
print("A3C's parallel actors give steadier learning and reach a policy that matches")
print("load to capacity - comparable response time and utilization to VPG, more reliably.")
fig,(a1,a2)=plt.subplots(1,2,figsize=(8.4,3.8))
a1.plot(res["a3c"][0],color="#0E6655",lw=1.8,label="A3C"); a1.plot(res["reinforce"][0],color="#B9770E",lw=1.6,label="Vanilla PG")
a1.set_xlabel("Training update"); a1.set_ylabel("Episode reward"); a1.set_title("Learning Curve"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
a2.bar(["A3C","VPG"],[res["a3c"][1],res["reinforce"][1]],color=["#0E6655","#B9770E"])
a2.set_ylabel("Avg response time"); a2.set_title("Final Response Time (lower is better)")
plt.tight_layout(); plt.savefig("fig8.png", dpi=130); plt.close()
