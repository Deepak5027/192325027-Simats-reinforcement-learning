# rlkit.py - a compact NumPy policy-gradient toolkit shared by all experiments.
# Implements REINFORCE (Vanilla PG), A2C/A3C (actor-critic), PPO (clipped) and
# TRPO (KL-constrained) over a tabular softmax policy, plus a DDPG routine for
# continuous control. Small environments let every experiment train in seconds.
import numpy as np

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def rollout(env, T, maxT=200):
    S, A, R, P = [], [], [], []
    s = env.reset()
    for _ in range(maxT):
        p = softmax(T[s]); a = int(np.random.choice(env.nA, p=p))
        s2, r, done = env.step(a)
        S.append(s); A.append(a); R.append(r); P.append(p[a]); s = s2
        if done: break
    return S, A, R, P

def discounted(R, gamma):
    G = np.zeros(len(R)); acc = 0.0
    for i in reversed(range(len(R))): acc = R[i] + gamma * acc; G[i] = acc
    return G

def kl(Told, Tnew, states):
    tot = 0.0
    for s in states:
        po, pn = softmax(Told[s]), softmax(Tnew[s])
        tot += np.sum(po * (np.log(po + 1e-9) - np.log(pn + 1e-9)))
    return tot / max(1, len(states))

def train_pg(method, env, updates=150, batch=8, alpha=1.5, gamma=0.99,
             clip=0.2, kl_delta=0.02, seed=0, return_policy=False):
    """Returns the per-update mean episode reward curve. method in
    {reinforce, a2c, a3c, ppo, trpo}."""
    rng = np.random.RandomState(seed); np.random.seed(seed)
    T = np.zeros((env.n, env.nA)); w = np.zeros(env.n); curve = []
    workers = 16 if method == "a3c" else batch          # A3C = more parallel actors
    for u in range(updates):
        BS, BA, BG, BP, ep = [], [], [], [], []
        for _ in range(workers):
            S, A, R, P = rollout(env, T)
            BS += S; BA += A; BG += list(discounted(R, gamma)); BP += P; ep.append(sum(R))
        BS, BA, BG, BP = map(np.array, (BS, BA, BG, BP)); curve.append(np.mean(ep))
        if method in ("a2c", "a3c", "ppo", "trpo"):
            adv = BG - w[BS]
            for s in np.unique(BS): w[s] += 0.1 * (BG[BS == s] - w[s]).mean()
        else:
            adv = BG.copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        epochs = 4 if method == "ppo" else 1
        Told = T.copy()
        for _ in range(epochs):
            grad = np.zeros_like(T)
            for i in range(len(BS)):
                s, a = BS[i], BA[i]; p = softmax(T[s]); gl = -p.copy(); gl[a] += 1
                if method == "ppo":
                    ratio = p[a] / (BP[i] + 1e-8)
                    if ratio * adv[i] <= np.clip(ratio, 1-clip, 1+clip) * adv[i]:
                        grad[s] += adv[i] * ratio * gl
                else:
                    grad[s] += adv[i] * gl
            grad /= len(BS)
            if method == "trpo":                       # KL-constrained line search
                step = alpha
                for _ in range(8):
                    cand = T + step * grad
                    if kl(Told, cand, np.unique(BS)) <= kl_delta:
                        T = cand; break
                    step *= 0.5
            else:
                T += alpha * grad
            T = np.clip(T, -30, 30)
    return (np.array(curve), T, w) if return_policy else np.array(curve)

def train_ddpg(env_fn, steps=4000, la=0.02, lc=0.05, noise=0.3, seed=0):
    """Deterministic actor a=w0*s+w1, quadratic critic; contextual continuous
    control. env_fn() -> (state s, optimal action a*(s)). Returns reward curve."""
    np.random.seed(seed); w = np.zeros(2); c = np.zeros(3); curve = []
    for t in range(steps):
        s, astar = env_fn()
        a_mean = w[0]*s + w[1]
        a = a_mean + np.random.randn() * noise * max(0.05, 1 - t/steps)
        r = -(a - astar)**2
        peak = c[0]*s + c[1]; err = r - (-(a-peak)**2 + c[2])
        c[0] += lc*err*2*(a-peak)*s; c[1] += lc*err*2*(a-peak); c[2] += lc*err
        ga = -2*(a_mean - (c[0]*s + c[1]))
        w[0] += la*ga*s; w[1] += la*ga; curve.append(r)
    return np.array(curve), w
