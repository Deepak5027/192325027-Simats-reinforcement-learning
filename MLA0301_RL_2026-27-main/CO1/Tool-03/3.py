states = ["Start", "Road", "Destination"]
actions = ["Move", "Wait"]

transition = {
    "Start":{"Move":"Road"},
    "Road":{"Move":"Destination"},
    "Destination":{"Move":"Destination"}
}

reward = {
    "Start":-2,
    "Road":-1,
    "Destination":100
}

energy = 20

state = "Start"

while state != "Destination" and energy > 0:
    action = "Move"
    state = transition[state][action]
    energy -= 5
    print(state, reward[state], energy)