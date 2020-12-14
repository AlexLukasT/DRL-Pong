import gym
import tensorflow as tf
import numpy as np

layers = tf.keras.layers


LEARNING_RATE = 1e-3
GAMMA = 0.99

BACKGROUND_COLOR = (144, 72, 17)
LEFT_PADDLE_COLOR = (213, 130, 74)
RIGHT_PADDLE_COLOR = (92, 186, 92)
BALL_COLOR = (236, 236, 236)


def main():
    model = get_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    env = gym.make("Pong-v0")
    observation = env.reset()

    num_games = 0
    states = []
    rewards = []
    actions = []

    while True:
        # used to feed difference between two images into the network
        prev_state = np.zeros((1, 160, 160, 1))
        cur_state = preprocess(observation)

        dif_state = cur_state - prev_state
        states.append(dif_state)

        prob = np.squeeze(model.predict(dif_state))
        # sample from the probability of going up
        action = 2 if np.random.uniform() < prob[0] else 5

        corrected_action = 0 if action == 2 else 1  # label "up" as 1 and "down" as 0
        actions.append(corrected_action)

        prev_state = cur_state

        env.render()
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            sum_reward = 0
            discount_rewards = []
            for r in reversed(rewards):
                sum_reward = r + GAMMA * sum_reward
                discount_rewards.append(sum_reward)
            discount_rewards.reverse()

            for state, reward, action in zip(states, discount_rewards, actions):
                with tf.GradientTape() as tape:
                    prob = tf.squeeze(model(state, training=True))
                    log_prob = tf.math.log(prob[action])
                    loss = tf.math.negative(log_prob * reward)
                gradients = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            observation = env.reset()

    env.close()


def preprocess(x):
    # raw image has shape (210,160,3) and dtype uint8
    x = x[34:194, :]  # cut away top and bottom stuff
    x[x == BACKGROUND_COLOR] = 0  # set background to 0
    x[x != 0] = 1  # set everything else to 1
    x = x[:, :, 0]  # (r,g,b) -> grey scale
    x = x[np.newaxis, :, :, np.newaxis]  # add batch and extra dimension for Keras
    return x.astype(np.float32)  # convert to float32


def get_model():
    inp = layers.Input((160, 160, 1))
    x = layers.Conv2D(16, (5, 5), strides=(3, 3), padding="same", activation="relu")(inp)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    return model


if __name__ == "__main__":
    main()
