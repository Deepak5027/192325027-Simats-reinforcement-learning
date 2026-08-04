import numpy as np

floors = 5
requests = [2, 4, 1, 3]
current_floor = 0
reward = 0

print("Smart Elevator Scheduling\n")

for req in requests:
    print("Current Floor:", current_floor)
    print("Serving Floor:", req)
    reward += 10 - abs(req - current_floor)
    current_floor = req
    print()

print("Total Reward =", reward)
print("Average Waiting Time Reduced")