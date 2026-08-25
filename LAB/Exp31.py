import numpy as np
import random


# -----------------------------------------
# Tic-Tac-Toe Environment
# -----------------------------------------

class TicTacToe:

    def __init__(self):

        self.reset()

    def reset(self):

        self.board = [0] * 9

        return self.get_state()

    def get_state(self):

        return tuple(self.board)

    # -------------------------------------
    # Available Actions
    # -------------------------------------

    def available_actions(self):

        return [
            i for i in range(9)
            if self.board[i] == 0
        ]

    # -------------------------------------
    # Check Winner
    # -------------------------------------

    def winner(self):

        lines = [

            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),

            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),

            (0, 4, 8),
            (2, 4, 6)
        ]

        for a, b, c in lines:

            if (
                self.board[a] != 0
                and
                self.board[a]
                == self.board[b]
                == self.board[c]
            ):

                return self.board[a]

        return 0

    # -------------------------------------
    # Game Step
    # -------------------------------------

    def step(self, action):

        self.board[action] = 1

        winner = self.winner()

        # Agent wins

        if winner == 1:

            return (
                self.get_state(),
                10,
                True
            )

        # Draw

        if len(
            self.available_actions()
        ) == 0:

            return (
                self.get_state(),
                0,
                True
            )

        # Random opponent

        opponent_action = random.choice(
            self.available_actions()
        )

        self.board[
            opponent_action
        ] = -1

        winner = self.winner()

        # Opponent wins

        if winner == -1:

            return (
                self.get_state(),
                -10,
                True
            )

        # Game continues

        return (
            self.get_state(),
            -1,
            False
        )


# -----------------------------------------
# SARSA Agent
# -----------------------------------------

Q = {}

alpha = 0.1
gamma = 0.9
epsilon = 0.2


# -----------------------------------------
# Get Q Values
# -----------------------------------------

def get_q(state, action):

    key = (state, action)

    if key not in Q:

        Q[key] = 0.0

    return Q[key]


# -----------------------------------------
# Epsilon-Greedy Action
# -----------------------------------------

def choose_action(state, actions):

    if random.random() < epsilon:

        return random.choice(actions)

    values = [
        get_q(state, action)
        for action in actions
    ]

    maximum = max(values)

    best = [
        action
        for action, value in zip(
            actions,
            values
        )
        if value == maximum
    ]

    return random.choice(best)


# -----------------------------------------
# SARSA Training
# -----------------------------------------

env = TicTacToe()

episodes = 20000

for episode in range(episodes):

    state = env.reset()

    actions = env.available_actions()

    action = choose_action(
        state,
        actions
    )

    done = False

    while not done:

        (
            next_state,
            reward,
            done
        ) = env.step(action)

        if done:

            target = reward

            Q[
                (state, action)
            ] = Q.get(
                (state, action),
                0
            ) + alpha * (
                target
                -
                Q.get(
                    (state, action),
                    0
                )
            )

            break

        next_actions = (
            env.available_actions()
        )

        next_action = choose_action(
            next_state,
            next_actions
        )

        current_q = Q.get(
            (state, action),
            0
        )

        next_q = Q.get(
            (next_state, next_action),
            0
        )

        target = (
            reward
            + gamma * next_q
        )

        Q[
            (state, action)
        ] = (
            current_q
            + alpha *
            (
                target
                - current_q
            )
        )

        state = next_state
        action = next_action

    if (
        episode + 1
    ) % 5000 == 0:

        print(
            "Training Episode:",
            episode + 1,
            "| Q-States:",
            len(Q)
        )


# -----------------------------------------
# Evaluation
# -----------------------------------------

epsilon = 0

games = 100

wins = 0
losses = 0
draws = 0

for game in range(games):

    state = env.reset()

    done = False

    while not done:

        actions = env.available_actions()

        if not actions:

            draws += 1
            break

        action = choose_action(
            state,
            actions
        )

        (
            next_state,
            reward,
            done
        ) = env.step(action)

        state = next_state

        if done:

            if reward == 10:

                wins += 1

            elif reward == -10:

                losses += 1

            else:

                draws += 1


# -----------------------------------------
# Performance
# -----------------------------------------

win_rate = (
    wins / games
) * 100

print("\nEvaluation Results")

print("Games:", games)

print("Wins:", wins)

print("Losses:", losses)

print("Draws:", draws)

print(
    "Win Rate:",
    round(win_rate, 2),
    "%"
)