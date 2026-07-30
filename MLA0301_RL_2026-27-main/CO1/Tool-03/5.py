battery = 30
deliveries = 0

while battery >= 5:
    deliveries += 1
    battery -= 5
    print("Delivery", deliveries)

print("Maximum Deliveries =", deliveries)