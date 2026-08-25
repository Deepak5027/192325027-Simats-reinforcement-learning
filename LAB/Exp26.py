import numpy as np
import random
import math

# -----------------------------------------
# Dynamic Pricing Environment
# -----------------------------------------

prices = np.array([
    50,
    75,
    100,
    125,
    150
])

# Probability that a customer buys
# at each corresponding price

buy_probability = np.array([
    0.90,
    0.75,
    0.60,
    0.45,
    0.30
])

n_prices = len(prices)
trials = 5000


# -----------------------------------------
# Generate Customer Purchase
# -----------------------------------------

def get_reward(price_index):

    if random.random() < buy_probability[price_index]:

        return prices[price_index]

    return 0


# -----------------------------------------
# Epsilon-Greedy
# -----------------------------------------

def epsilon_greedy():

    counts = np.zeros(n_prices)
    values = np.zeros(n_prices)

    total_revenue = 0

    epsilon = 0.1

    for t in range(trials):

        if random.random() < epsilon:

            price = random.randint(
                0,
                n_prices - 1
            )

        else:

            price = np.argmax(values)

        revenue = get_reward(price)

        counts[price] += 1

        total_revenue += revenue

        values[price] += (
            revenue - values[price]
        ) / counts[price]

    return total_revenue, values


# -----------------------------------------
# Upper Confidence Bound
# -----------------------------------------

def ucb():

    counts = np.zeros(n_prices)
    values = np.zeros(n_prices)

    total_revenue = 0

    # Try every price once

    for price in range(n_prices):

        revenue = get_reward(price)

        counts[price] += 1

        total_revenue += revenue

        values[price] = revenue

    # UCB selection

    for t in range(n_prices, trials):

        confidence = np.zeros(n_prices)

        for price in range(n_prices):

            confidence[price] = (
                values[price]
                +
                math.sqrt(
                    2 * math.log(t + 1)
                    / counts[price]
                )
            )

        selected = np.argmax(confidence)

        revenue = get_reward(selected)

        counts[selected] += 1

        total_revenue += revenue

        values[selected] += (
            revenue - values[selected]
        ) / counts[selected]

    return total_revenue, values


# -----------------------------------------
# Thompson Sampling
# -----------------------------------------

def thompson_sampling():

    # Beta distribution parameters
    success = np.ones(n_prices)
    failure = np.ones(n_prices)

    total_revenue = 0

    for _ in range(trials):

        samples = np.random.beta(
            success,
            failure
        )

        selected = np.argmax(samples)

        purchase = (
            random.random()
            < buy_probability[selected]
        )

        if purchase:

            revenue = prices[selected]

            success[selected] += 1

        else:

            revenue = 0

            failure[selected] += 1

        total_revenue += revenue

    estimated_probability = (
        success /
        (success + failure)
    )

    return (
        total_revenue,
        estimated_probability
    )


# -----------------------------------------
# Run Algorithms
# -----------------------------------------

eg_revenue, eg_values = (
    epsilon_greedy()
)

ucb_revenue, ucb_values = (
    ucb()
)

ts_revenue, ts_values = (
    thompson_sampling()
)


# -----------------------------------------
# Calculate Average Revenue
# -----------------------------------------

eg_average = eg_revenue / trials
ucb_average = ucb_revenue / trials
ts_average = ts_revenue / trials


print("Dynamic Pricing using Bandit Algorithms\n")

print(
    "Epsilon-Greedy"
)

print(
    "Total Revenue =",
    round(eg_revenue, 2)
)

print(
    "Average Revenue =",
    round(eg_average, 2)
)

print()


print(
    "UCB"
)

print(
    "Total Revenue =",
    round(ucb_revenue, 2)
)

print(
    "Average Revenue =",
    round(ucb_average, 2)
)

print()


print(
    "Thompson Sampling"
)

print(
    "Total Revenue =",
    round(ts_revenue, 2)
)

print(
    "Average Revenue =",
    round(ts_average, 2)
)


# -----------------------------------------
# Determine Best Strategy
# -----------------------------------------

revenues = [
    eg_revenue,
    ucb_revenue,
    ts_revenue
]

names = [
    "Epsilon-Greedy",
    "UCB",
    "Thompson Sampling"
]

best = np.argmax(revenues)

print(
    "\nBest Pricing Strategy:",
    names[best]
)

print(
    "Maximum Revenue =",
    round(revenues[best], 2)
)