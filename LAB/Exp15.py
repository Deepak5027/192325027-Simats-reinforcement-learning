import random

steps = 20
balance = 100

print("Humanoid Robot Walking\n")

for i in range(steps):

    loss = random.randint(0, 3)
    balance -= loss

    print("Step", i + 1,
          "Balance =", balance)

print("\nFinal Balance =", balance)

if balance > 70:
    print("Stable Walking Achieved")
else:
    print("Robot Lost Balance")