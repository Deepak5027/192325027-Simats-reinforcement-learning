import random

bandwidth = 100

for packet in range(10):

    usage = random.randint(5,20)

    if bandwidth >= usage:
        bandwidth -= usage
        reward = 10
        print("Packet Sent", reward)
    else:
        reward = -5
        print("Dropped", reward)

print("Remaining Bandwidth:", bandwidth)