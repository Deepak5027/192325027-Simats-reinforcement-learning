import numpy as np
from sklearn.ensemble import RandomForestRegressor

# -----------------------------------------
# Generate Historical Portfolio Data
# -----------------------------------------

np.random.seed(10)

samples = 1000

# Features:
# 1 -> Expected Return
# 2 -> Risk
# 3 -> Investment Horizon

expected_return = np.random.uniform(
    0.02,
    0.15,
    samples
)

risk = np.random.uniform(
    0.01,
    0.30,
    samples
)

horizon = np.random.randint(
    1,
    11,
    samples
)

# -----------------------------------------
# Long-Term Portfolio Value
# -----------------------------------------

future_value = (
    expected_return * horizon
    -
    0.5 * risk
)

# Add random market variation

future_value += np.random.normal(
    0,
    0.01,
    samples
)

X = np.column_stack([
    expected_return,
    risk,
    horizon
])

y = future_value


# -----------------------------------------
# Train Value Prediction Model
# -----------------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=10
)

model.fit(
    X,
    y
)


# -----------------------------------------
# Portfolio Strategies
# -----------------------------------------

portfolios = {

    "Conservative": [
        0.05,
        0.05,
        5
    ],

    "Balanced": [
        0.09,
        0.12,
        5
    ],

    "Growth": [
        0.13,
        0.20,
        5
    ],

    "Aggressive": [
        0.15,
        0.28,
        5
    ]
}


# -----------------------------------------
# Predict Long-Term Value
# -----------------------------------------

print(
    "Portfolio Value Predictions\n"
)

predictions = {}

for name, values in portfolios.items():

    prediction = model.predict(
        [values]
    )[0]

    predictions[name] = prediction

    print(
        name,
        "-> Predicted Value:",
        round(
            prediction,
            4
        )
    )


# -----------------------------------------
# Select Best Portfolio
# -----------------------------------------

best_portfolio = max(
    predictions,
    key=predictions.get
)

print(
    "\nBest Portfolio:",
    best_portfolio
)

print(
    "Predicted Long-Term Value:",
    round(
        predictions[best_portfolio],
        4
    )
)