import numpy as np

np.random.seed(1)

# transitions[state][action] = list of (probability, next_state, reward)
transitions = {
    "Sleep": {"work": [(1.0, "Study", -1)], "rest": [(1.0, "Sleep", -2)]},
    "Study": {"work": [(0.8, "Pass", 10), (0.2, "Play", -1)], "rest": [(1.0, "Sleep", -1)]},
    "Play":  {"work": [(0.6, "Study", 0), (0.4, "Play", -1)], "rest": [(1.0, "Sleep", -2)]},
}
policy = {"Sleep": "work", "Study": "work", "Play": "work"}


def step(state, action):
    outcomes = transitions[state][action]
    i = np.random.choice(len(outcomes), p=[o[0] for o in outcomes])
    return outcomes[i][1], outcomes[i][2]


def run_episode(start="Sleep", max_steps=20):
    state, total = start, 0
    for _ in range(max_steps):
        action = policy[state]
        next_state, reward = step(state, action)
        total += reward
        print(f"  {state:<6} --({action})--> {next_state:<6} | reward = {reward}")
        state = next_state
        if state == "Pass":
            break
    print(f"  Total reward: {total}")
    return total


rewards = []
for ep in range(5):
    print(f"Episode {ep + 1}:")
    rewards.append(run_episode())
print(f"\nAverage reward over {len(rewards)} episodes: {np.mean(rewards):.2f}")
