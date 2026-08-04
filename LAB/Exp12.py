import numpy as np

positions = ["Home", "Pick", "Lift", "Move", "Place"]

reward = 0

print("Robot Arm Operations\n")

for i in range(len(positions)-1):
    print(positions[i], "->", positions[i+1])
    reward += 5

print("\nTotal Reward =", reward)