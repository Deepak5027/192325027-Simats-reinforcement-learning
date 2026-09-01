battery = 100

no_fly = [3,5]

for location in range(1,8):

    if location in no_fly:
        print("Avoid Location", location)
        continue

    if battery < 15:
        print("Battery Low")
        break

    battery -= 15

    print("Delivered at", location)

print("Battery Left:", battery)