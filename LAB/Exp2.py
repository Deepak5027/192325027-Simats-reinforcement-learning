import random

position = 0
goal = 5
reward = 0

print("Robot Navigation")

while position < goal:
    move = random.choice([1, 1, 1])
    position += move
    reward += 1
    print("Robot moved to", position)

print("\nGoal Reached")
print("Total Reward =", reward)