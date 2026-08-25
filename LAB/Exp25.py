import numpy as np
import random
import math

# -----------------------------------------
# Advertisement Environment
# -----------------------------------------

np.random.seed(10)

true_ctr = [
    0.05,
    0.10,
    0.20,
    0.15,
    0.08
]

n_ads = len(true_ctr)
trials = 5000


# -----------------------------------------
# Function to Generate Click
# -----------------------------------------

def get_reward(ad):

    if random.random() < true_ctr[ad]:
        return 1

    return 0


# -----------------------------------------
# Epsilon-Greedy
# -----------------------------------------

def epsilon_greedy():

    counts = np.zeros(n_ads)
    rewards = np.zeros(n_ads)

    epsilon = 0.1

    total_clicks = 0

    for t in range(trials):

        if random.random() < epsilon:

            ad = random.randint(
                0,
                n_ads - 1
            )

        else:

            ad = np.argmax(
                rewards
            )

        reward = get_reward(ad)

        counts[ad] += 1
        total_clicks += reward

        rewards[ad] += (
            reward - rewards[ad]
        ) / counts[ad]

    return total_clicks, rewards


# -----------------------------------------
# Upper Confidence Bound
# -----------------------------------------

def ucb():

    counts = np.zeros(n_ads)
    rewards = np.zeros(n_ads)

    total_clicks = 0

    # Try every advertisement once

    for ad in range(n_ads):

        reward = get_reward(ad)

        counts[ad] += 1
        total_clicks += reward

        rewards[ad] = reward

    for t in range(n_ads, trials):

        confidence = np.zeros(n_ads)

        for ad in range(n_ads):

            confidence[ad] = (
                rewards[ad]
                +
                math.sqrt(
                    2 * math.log(t + 1)
                    / counts[ad]
                )
            )

        ad = np.argmax(confidence)

        reward = get_reward(ad)

        counts[ad] += 1
        total_clicks += reward

        rewards[ad] += (
            reward - rewards[ad]
        ) / counts[ad]

    return total_clicks, rewards


# -----------------------------------------
# Thompson Sampling
# -----------------------------------------

def thompson_sampling():

    successes = np.ones(n_ads)
    failures = np.ones(n_ads)

    total_clicks = 0

    for _ in range(trials):

        samples = np.random.beta(
            successes,
            failures
        )

        ad = np.argmax(samples)

        reward = get_reward(ad)

        total_clicks += reward

        if reward == 1:

            successes[ad] += 1

        else:

            failures[ad] += 1

    estimated_ctr = (
        successes /
        (
            successes +
            failures
        )
    )

    return total_clicks, estimated_ctr


# -----------------------------------------
# Run Algorithms
# -----------------------------------------

eg_clicks, eg_rates = (
    epsilon_greedy()
)

ucb_clicks, ucb_rates = (
    ucb()
)

ts_clicks, ts_rates = (
    thompson_sampling()
)


# -----------------------------------------
# Calculate CTR
# -----------------------------------------

eg_ctr = eg_clicks / trials
ucb_ctr = ucb_clicks / trials
ts_ctr = ts_clicks / trials


print("Advertisement Bandit Comparison\n")

print(
    "Epsilon-Greedy:"
)
print(
    "Clicks =", eg_clicks,
    "CTR =", round(eg_ctr, 4)
)

print()

print(
    "UCB:"
)
print(
    "Clicks =", ucb_clicks,
    "CTR =", round(ucb_ctr, 4)
)

print()

print(
    "Thompson Sampling:"
)
print(
    "Clicks =", ts_clicks,
    "CTR =", round(ts_ctr, 4)
)


# -----------------------------------------
# Find Best Algorithm
# -----------------------------------------

ctrs = [
    eg_ctr,
    ucb_ctr,
    ts_ctr
]

names = [
    "Epsilon-Greedy",
    "UCB",
    "Thompson Sampling"
]

best = np.argmax(ctrs)

print(
    "\nBest Algorithm:",
    names[best]
)

print(
    "Highest CTR:",
    round(ctrs[best], 4)
)