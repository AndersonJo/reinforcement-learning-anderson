import os
from argparse import ArgumentParser

import gym
import keras.backend as K
import numpy as np
import tensorflow as tf
from gym import Env
from keras import Model, Input, Sequential
from keras.engine.saving import load_model
from keras.layers import Dense, Activation, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('mode', default='train', type=str)
    parser.add_argument('--checkpoint', default=None)
    parser.set_defaults(mode='train', checkpoint=None)
    args = parser.parse_args()
    return args


def create_model(input_size: int, n_actions: int, seed: int = 0):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    model = Sequential()
    model.add(Dense(256, activation='relu', input_dim=input_size))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_actions, activation='softmax'))

    return model


class REINFORCE(object):

    def __init__(self, env: Env, model: Model):
        self.env = env
        self.model = model

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_rewards = []
        self.buffer_qvalues = []
        self.cur_rewards = []

        self._build_loss()

    def _build_loss(self):
        # Compile Model
        self.model.compile(loss=categorical_crossentropy, optimizer='adam')

        # Build loss
        softmax_outputs = self.model.output
        onehot_actions = K.placeholder(shape=(None, int(self.env.action_space.n)), name='actions')
        discouted_rewards = K.placeholder(shape=(None,), name='discounted_rewards_or_q_values')  # Q values

        action_probs = K.sum(onehot_actions * softmax_outputs, axis=1)
        score = - discouted_rewards * K.log(action_probs)
        score = K.mean(score)

        # Build Training Function
        adam = Adam(lr=0.001)
        updates = adam.get_updates(params=self.model.trainable_weights,
                                   loss=score)
        self.train_fn = K.function(inputs=[self.model.input,
                                           onehot_actions,
                                           discouted_rewards],
                                   outputs=[score],
                                   updates=updates)

    def append(self, state: np.ndarray, action: int, reward: float):
        onehot_action = np.zeros([self.env.action_space.n], dtype=np.float32)
        onehot_action[action] = 1

        self.buffer_states.append(state)
        self.buffer_actions.append(onehot_action)
        self.buffer_rewards.append(reward)
        self.cur_rewards.append(reward)

    def clear(self):
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_rewards.clear()
        self.buffer_qvalues.clear()
        self.cur_rewards.clear()

    def action(self, state) -> int:
        action_probs = self.model.predict(np.array([state]))
        action = np.random.choice(range(action_probs.shape[1]), p=action_probs.ravel())
        return action

    def train(self, n_epochs=10, batch_episodes=1, gamma=0.99, episode_step_increment: int = 100,
              limit_episode_step: int = 10000):
        env = self.env
        best_reward = 0

        for epoch in range(n_epochs):
            current_state = env.reset()

            while True:
                # env.render()
                action = self.action(current_state)
                next_state, reward, done, info = env.step(action)

                if done:
                    break

                self.append(current_state, action, reward)
                current_state = next_state

            # Calculate Q Values
            q_values = self.calculate_q_values(self.cur_rewards, gamma=gamma)
            self.buffer_qvalues.extend(q_values)
            self.cur_rewards.clear()

            # Check Batch Episodes
            if epoch % batch_episodes != 0:
                continue

            # Update Gradients
            score = self.train_fn([self.buffer_states, self.buffer_actions, self.buffer_qvalues])[0]
            _score = round(float(score) / batch_episodes, 2)
            _sum_qval = round(float(np.sum(q_values)) / batch_episodes, 2)
            _sum_reward = round(float(np.sum(self.buffer_rewards)) / batch_episodes, 2)

            # Increase Maximum Episode Steps
            # _episode_step_increased = False
            # if env._max_episode_steps < limit_episode_step and _sum_reward >= env._max_episode_steps - 1:
            #     env._max_episode_steps += episode_step_increment
            #     if env._max_episode_steps > limit_episode_step:
            #         env._max_episode_steps = limit_episode_step
            #     _episode_step_increased = True
            if epoch > 2000:
                env._max_episode_steps = epoch

            # Save Model
            is_saved = False
            if best_reward <= _sum_reward:
                self.model.save('checkpoints/reinforce_keras.h5')
                best_reward = _sum_reward
                is_saved = True

            # Visualization
            print(f'[{epoch}] score: {_score:6} | q-values: {_sum_qval:8} | reward: {_sum_reward:6} | '
                  f'save:{is_saved}')
            self.clear()

    def play(self, n=100):
        env = self.env
        env._max_episode_steps = 10000

        for epoch in range(n):
            current_state = env.reset()
            rewards = []
            step = 0
            while True:
                step += 1
                env.render()
                action = self.action(current_state)
                next_state, reward, done, info = env.step(action)
                rewards.append(reward)

                if done:
                    break

                current_state = next_state

            # Visualization
            _sum_reward = round(float(np.sum(rewards)), 2)
            print(f'[{epoch:2}] reward:{_sum_reward}')

    @staticmethod
    def calculate_q_values(rewards: list, gamma: float):
        res = []
        sum_rewards = 0.0
        for r in rewards[::-1]:
            sum_rewards *= gamma
            sum_rewards += r
            res.append(sum_rewards)
        return np.log(list(reversed(res)))


def main():
    np.random.seed(0)
    args = get_args()

    env = gym.make("CartPole-v0")

    env.reset()
    print('mode             :', args.mode)
    print('observation space:', env.observation_space.shape)
    print('action space     :', env.action_space.n)

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print(f'Load existing model - {args.checkpoint}')
        model = load_model(args.checkpoint)
    else:
        print('Craeted a new model')
        model = create_model(env.observation_space.shape[0], env.action_space.n)

    # Plot the model
    plot_model(model, 'reinforce.png', show_shapes=True)

    reinforce = REINFORCE(env, model)
    if args.mode == 'play':
        print('MODE: PLAY!')
        reinforce.play()
    elif args.mode == 'train':
        print('MODE: TRAIN!')
        reinforce.train(n_epochs=5000, batch_episodes=2)


if __name__ == '__main__':
    main()
