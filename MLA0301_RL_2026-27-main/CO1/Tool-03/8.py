import random

arms = [0.3,0.6,0.8]

random_reward = 0

for i in range(500):
    arm = random.randint(0,2)
    random_reward += 1 if random.random()<arms[arm] else 0

print("Random Reward:", random_reward)

epsilon = 0.1

Q=[0,0,0]
N=[0,0,0]
reward_sum=0

for i in range(500):

    if random.random()<epsilon:
        arm=random.randint(0,2)
    else:
        arm=Q.index(max(Q))

    reward=1 if random.random()<arms[arm] else 0

    N[arm]+=1
    Q[arm]+= (reward-Q[arm])/N[arm]

    reward_sum+=reward

print("Epsilon Greedy Reward:", reward_sum)