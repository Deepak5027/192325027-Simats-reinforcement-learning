import numpy as np
import tensorflow as tf

from collections import deque
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# -----------------------------------------
# Strategy Game Environment
# -----------------------------------------

class StrategyEnv:

    def __init__(self):

        self.state_size = 4
        self.action_size = 2

        # Action 0 -> Resource allocation
        # Action 1 -> Unit production

        self.max_steps = 50

    def reset(self):

        self.resources = 20.0
        self.units = 1.0
        self.enemy = 20.0

        self.steps = 0

        return self.get_state()

    def get_state(self):

        return np.array([
            self.resources / 100,
            self.units / 20,
            self.enemy / 20,
            self.steps / self.max_steps
        ], dtype=np.float32)

    def step(self, action):

        self.steps += 1

        resource_action = np.clip(
            action[0],
            0,
            1
        )

        unit_action = np.clip(
            action[1],
            0,
            1
        )

        # ---------------------------------
        # Gather Resources
        # ---------------------------------

        gathered = (
            resource_action * 5
        )

        self.resources += gathered

        # ---------------------------------
        # Build Units
        # ---------------------------------

        cost = unit_action * 3

        if self.resources >= cost:

            self.resources -= cost

            self.units += unit_action * 2

        # ---------------------------------
        # Attack Enemy
        # ---------------------------------

        damage = (
            self.units *
            unit_action *
            0.5
        )

        self.enemy -= damage

        # ---------------------------------
        # Reward
        # ---------------------------------

        reward = (
            gathered
            +
            damage
            -
            cost * 0.2
        )

        # Victory

        if self.enemy <= 0:

            reward += 50

            done = True

        else:

            done = False

        # Resource shortage penalty

        if self.resources < 1:

            reward -= 5

        truncated = (
            self.steps >= self.max_steps
        )

        return (
            self.get_state(),
            reward,
            done,
            truncated
        )


# -----------------------------------------
# Actor Network
# -----------------------------------------

def build_actor():

    inputs = tf.keras.Input(
        shape=(4,)
    )

    x = Dense(
        64,
        activation="relu"
    )(inputs)

    x = Dense(
        64,
        activation="relu"
    )(x)

    outputs = Dense(
        2,
        activation="sigmoid"
    )(x)

    return Model(
        inputs,
        outputs
    )


# -----------------------------------------
# Critic Network
# -----------------------------------------

def build_critic():

    state_input = tf.keras.Input(
        shape=(4,)
    )

    action_input = tf.keras.Input(
        shape=(2,)
    )

    x = tf.keras.layers.Concatenate()(
        [state_input, action_input]
    )

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    output = Dense(
        1,
        activation="linear"
    )(x)

    return Model(
        [state_input, action_input],
        output
    )


# -----------------------------------------
# DDPG Agent
# -----------------------------------------

class DDPG:

    def __init__(self):

        self.gamma = 0.99
        self.tau = 0.005

        self.actor = build_actor()
        self.target_actor = build_actor()

        self.critic = build_critic()
        self.target_critic = build_critic()

        self.target_actor.set_weights(
            self.actor.get_weights()
        )

        self.target_critic.set_weights(
            self.critic.get_weights()
        )

        self.actor_optimizer = (
            tf.keras.optimizers.Adam(
                0.001
            )
        )

        self.critic_optimizer = (
            tf.keras.optimizers.Adam(
                0.002
            )
        )

        self.memory = deque(
            maxlen=10000
        )

    # -------------------------------------
    # Select Action
    # -------------------------------------

    def act(
        self,
        state,
        noise=0.1
    ):

        state = tf.convert_to_tensor(
            state.reshape(1, -1),
            dtype=tf.float32
        )

        action = self.actor(
            state
        )[0].numpy()

        action += np.random.normal(
            0,
            noise,
            size=2
        )

        return np.clip(
            action,
            0,
            1
        )

    # -------------------------------------
    # Store Experience
    # -------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    # -------------------------------------
    # Training
    # -------------------------------------

    def train(
        self,
        batch_size=32
    ):

        if len(self.memory) < batch_size:

            return

        batch = [
            self.memory[i]
            for i in np.random.choice(
                len(self.memory),
                batch_size,
                replace=False
            )
        ]

        states = np.array([
            x[0] for x in batch
        ])

        actions = np.array([
            x[1] for x in batch
        ])

        rewards = np.array([
            x[2] for x in batch
        ], dtype=np.float32)

        next_states = np.array([
            x[3] for x in batch
        ])

        dones = np.array([
            x[4] for x in batch
        ], dtype=np.float32)

        states = tf.convert_to_tensor(
            states,
            dtype=tf.float32
        )

        actions = tf.convert_to_tensor(
            actions,
            dtype=tf.float32
        )

        rewards = tf.convert_to_tensor(
            rewards,
            dtype=tf.float32
        )

        next_states = tf.convert_to_tensor(
            next_states,
            dtype=tf.float32
        )

        # ---------------------------------
        # Critic Update
        # ---------------------------------

        with tf.GradientTape() as tape:

            next_actions = (
                self.target_actor(
                    next_states
                )
            )

            next_q = (
                self.target_critic(
                    [
                        next_states,
                        next_actions
                    ]
                )
            )

            target = (
                rewards
                +
                self.gamma
                *
                (1 - dones)
                *
                tf.squeeze(next_q)
            )

            current_q = tf.squeeze(
                self.critic(
                    [
                        states,
                        actions
                    ]
                )
            )

            critic_loss = tf.reduce_mean(
                tf.square(
                    target - current_q
                )
            )

        critic_gradients = tape.gradient(
            critic_loss,
            self.critic.trainable_variables
        )

        self.critic_optimizer.apply_gradients(
            zip(
                critic_gradients,
                self.critic.trainable_variables
            )
        )

        # ---------------------------------
        # Actor Update
        # ---------------------------------

        with tf.GradientTape() as tape:

            new_actions = self.actor(
                states
            )

            actor_loss = -tf.reduce_mean(
                self.critic(
                    [
                        states,
                        new_actions
                    ]
                )
            )

        actor_gradients = tape.gradient(
            actor_loss,
            self.actor.trainable_variables
        )

        self.actor_optimizer.apply_gradients(
            zip(
                actor_gradients,
                self.actor.trainable_variables
            )
        )

        # ---------------------------------
        # Soft Target Update
        # ---------------------------------

        for target, source in zip(
            self.target_actor.variables,
            self.actor.variables
        ):

            target.assign(
                self.tau * source
                +
                (1 - self.tau) * target
            )

        for target, source in zip(
            self.target_critic.variables,
            self.critic.variables
        ):

            target.assign(
                self.tau * source
                +
                (1 - self.tau) * target
            )


# -----------------------------------------
# Training
# -----------------------------------------

env = StrategyEnv()

agent = DDPG()

episodes = 300

for episode in range(episodes):

    state = env.reset()

    total_reward = 0

    for step in range(
        env.max_steps
    ):

        action = agent.act(
            state,
            noise=0.2
        )

        (
            next_state,
            reward,
            done,
            truncated
        ) = env.step(action)

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done or truncated
        )

        agent.train()

        state = next_state

        total_reward += reward

        if done or truncated:

            break

    if (
        episode + 1
    ) % 25 == 0:

        print(
            "Episode:",
            episode + 1,
            "| Reward:",
            round(
                total_reward,
                2
            )
        )


# -----------------------------------------
# Evaluation
# -----------------------------------------

state = env.reset()

total_reward = 0

for step in range(
    env.max_steps
):

    action = agent.act(
        state,
        noise=0
    )

    (
        next_state,
        reward,
        done,
        truncated
    ) = env.step(action)

    total_reward += reward

    state = next_state

    if done or truncated:

        break


print("\nEvaluation Results")

print(
    "Total Reward:",
    round(
        total_reward,
        2
    )
)

print(
    "Resources:",
    round(
        env.resources,
        2
    )
)

print(
    "Units:",
    round(
        env.units,
        2
    )
)

print(
    "Enemy Strength:",
    round(
        env.enemy,
        2
    )
)

if env.enemy <= 0:

    print(
        "Opponent Defeated"
    )

else:

    print(
        "Opponent Not Yet Defeated"
    )