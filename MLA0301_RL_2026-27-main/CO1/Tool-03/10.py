import random

threshold = 80

usage = 0

for hour in range(24):

    consume = random.randint(2,6)

    if usage + consume <= threshold:
        usage += consume
        reward = 10
    else:
        reward = -10

    print("Hour:", hour+1, "Usage:", usage, "Reward:", reward)

print("Total Energy Used:", usage)