signals = ["North", "East", "South", "West"]

emergency = "East"

for signal in signals:
    if signal == emergency:
        print(signal, "GREEN")
    else:
        print(signal, "RED")