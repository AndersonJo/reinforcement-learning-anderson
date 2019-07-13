from multiprocessing.connection import Connection
from typing import List

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.multiprocessing import Process, Pipe


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        print(type(input))
        print(input.size())
        return input.view(input.size(0), -1)


class A2CModel(nn.Module):
    def __init__(self, output_size: int):
        super(A2CModel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(in_features=7 * 7 * 64,
                      out_features=512),
            nn.LeakyReLU())

        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        self.activation3 = nn.LeakyReLU()
        self.flatten = Flatten()
        self.linear = nn.Linear(in_features=7 * 7 * 64,
                                out_features=512)
        self.activation4 = nn.LeakyReLU()

        self.actor = nn.Linear(in_features=512,
                               out_features=output_size)
        self.critic = nn.Linear(in_features=512,
                                out_features=1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.xavier_uniform_(p.weight)
                # nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            elif isinstance(p, nn.Linear):
                # torch.nn.init.xavier_uniform_(p.weight)
                nn.init.kaiming_uniform_(p.weight, a=1.)
                p.bias.data.zero_()

    def forward(self, states):
        # h = self.feature(state)
        h = self.conv1(states)
        h = self.activation1(h)
        print(h.size())

        h = self.conv2(h)
        h = self.activation2(h)
        print(h.size())

        h = self.conv3(h)
        h = self.activation3(h)
        print(h.size())

        h = self.flatten(h)
        print('flatten:', h.size())
        h = self.linear(h)
        print('linear:', h.size())
        h = self.activation4(h)
        print('after linear activation4:', h.size())

        policy = self.actor(h)
        value = self.critic(h)
        return policy, value


class MultiProcessEnv(Process):
    def __init__(self, process_idx: int, env: gym.Env, child_conn: Connection, render: bool = False,
                 replay_size: int = 4):
        super(MultiProcessEnv, self).__init__()
        self.process_idx = process_idx
        self.env = env
        self.child_conn = child_conn
        self.render = render

        # Game State Variables
        self.step = 0

        # Set Replay Memory
        self.replay_size = replay_size
        height, width, channel = env.observation_space.shape
        self.replay: np.ndarray = np.zeros([replay_size, height, width])

        self.reset()

    def run(self):
        super(MultiProcessEnv, self).run()
        self.reset()
        env = self.env
        render = self.render

        while True:
            action = self.child_conn.recv()
            # action = env.action_space.sample()

            if render:
                env.render()

            next_state, reward, done, info = env.step(action)

            self.memory_state(next_state)
            self.step += 1

            # Send information to Parent Processor
            self.child_conn.send([self.replay, reward, done, info])

            # Check
            if info['life'] <= 0:
                done = True

            if done:
                break

    @property
    def shape(self) -> tuple:
        return self.env.observation_space.shape

    def memory_state(self, state):
        self.replay[:3, :, :] = self.replay[1:, :, :]
        self.replay[3, :, :] = self.preprocess(state)

    def preprocess(self, state):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        return h

    def reset(self):
        self.step = 0

        # Initialize Replay Memory
        init_state = self.env.reset()
        for i in range(self.replay_size):
            self.replay[i, :, :] = self.preprocess(init_state)


class Agent(object):

    def __init__(self, model: nn.Module, cuda: bool = False):
        self.device: str = 'cuda' if cuda else 'cpu'
        self.model = model.to(self.device)

    def get_action(self, states: np.ndarray):
        states = torch.from_numpy(states).float().to(self.device)
        import ipdb
        ipdb.set_trace()
        policy, value = self.model(states)
        policy = F.softmax(policy, dim=-1).data.cpu().numpy()


class A2CAgent(Agent):
    pass


class A2CTrain(object):
    def __init__(self, game_id: str, model: nn.Module, n_processor: int, render: bool = False,
                 replay_size: int = 4, cuda=False):
        self.game_id = game_id
        self.model = model
        self.n_processor = n_processor
        self.render = render
        self.replay_size = replay_size

        # Initialize Agent
        self.agent: A2CAgent = A2CAgent(model, cuda=cuda)

        # Initialize Environments
        self.envs: List[MultiProcessEnv] = []
        self._init_envs()

    def _init_envs(self):
        for idx in range(self.n_processor):
            parent_conn, child_conn = Pipe()
            env = self.create_env()
            env_processor = MultiProcessEnv(idx, env, child_conn,
                                            replay_size=self.replay_size,
                                            render=self.render)
            env_processor.start()
            self.envs.append(env_processor)

    def create_env(self) -> gym.Env:
        env = gym.make(self.game_id)
        return env

    def train(self):
        agent = self.agent
        envs = self.envs

        # Prepare Training
        states = self.get_init_states()

        action = agent.get_action(states)
        print('action:', action)

    def get_init_states(self) -> np.ndarray:
        observation_shape = self.envs[0].shape
        states = np.zeros([self.n_processor, self.replay_size, observation_shape[0], observation_shape[1]])
        print('get_init_states:', states.shape)
        return states


class A2CMarioTrain(A2CTrain):

    def create_env(self) -> gym.Env:
        env = gym_super_mario_bros.make(self.game_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        print('env observation:', env.observation_space.shape)
        return env


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    output_size = env.action_space.n

    # Hyperparameters
    a2c_model = A2CModel(output_size)
    a2c_train = A2CMarioTrain('SuperMarioBros-v0', a2c_model, n_processor=1, render=True, cuda=True)
    a2c_train.train()


if __name__ == '__main__':
    main()
