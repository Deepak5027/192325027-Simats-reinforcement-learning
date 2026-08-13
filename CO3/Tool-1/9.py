# Challenge 9: Q-learning update of the action-value function and convergence.
# Small 1x6 corridor, goal at the end. We show one worked update, then track
# the largest Q change per block of episodes to demonstrate convergence.
import numpy as np
np.random.seed(0)
N, GOAL = 6, 5
def move(s,a): return (min(GOAL,s+1) if a==1 else max(0,s-1))
def rew(s): return (10,True) if s==GOAL else (-1,False)
Q = np.zeros((N,2)); alpha, gamma, eps = 0.5, 0.9, 0.2

# one worked update from state 4 taking action 'right'
s, a = 4, 1; s2 = move(s,a); r,_ = rew(s2)
old = Q[s,a]; Q[s,a] += alpha*(r + gamma*np.max(Q[s2]) - Q[s,a])
print(f"Worked update: Q({s},right) = {old:.2f} + {alpha}*({r} + {gamma}*{np.max(Q[s2]):.2f} - {old:.2f}) = {Q[s,a]:.2f}\n")

Q = np.zeros((N,2))
print("Convergence (max |Q change| per 50 episodes):")
for block in range(6):
    maxd = 0
    for _ in range(50):
        s = 0
        for _ in range(30):
            a = np.random.randint(2) if np.random.random()<eps else int(np.argmax(Q[s]))
            s2 = move(s,a); r, done = rew(s2)
            d = alpha*(r + gamma*np.max(Q[s2]) - Q[s,a]); Q[s,a]+=d
            maxd = max(maxd, abs(d)); s = s2
            if done: break
    print(f"  episodes {block*50+1:>3}-{block*50+50}: max change = {maxd:.4f}")
print("\nFinal Q-table (rows=state 0..5, cols=[left,right]):")
print(np.round(Q,2))
