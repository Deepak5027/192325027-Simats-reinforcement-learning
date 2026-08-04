import random

actions = [
    "Move Forward",
    "Turn Left",
    "Turn Right",
    "Reverse",
    "Park"
]

print("Autonomous Parking\n")

reward = 0

for action in actions:
    print(action)
    reward += random.randint(2,5)

print("\nParking Successful")
print("Total Reward =", reward)