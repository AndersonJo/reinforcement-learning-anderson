from collections import deque
from multiprocessing.connection import Connection
from typing import List, Tuple

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
        return input.view(input.size(0), -1)


class A2CModel(nn.Module):
    def __init__(self, input_shape, n_action: int, replay_size: int):
        super(A2CModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=8,
                      stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),
            nn.ReLU())
        conv_output_size = self._calculate_output_size(input_shape, replay_size)

        # Flatten
        self.flatten = Flatten()

        # Actor
        self.policy = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_action))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(in_features=conv_output_size, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1))

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.xavier_uniform_(p.weight)
                # nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            elif isinstance(p, nn.Linear):
                torch.nn.init.xavier_uniform_(p.weight)
                # nn.init.kaiming_uniform_(p.weight, a=1.)
                p.bias.data.zero_()

    def _calculate_output_size(self, input_shape: tuple, replay_size: int) -> int:
        o = self.conv(torch.zeros(1, replay_size, *input_shape))
        output_size = int(np.prod(o.size()))
        return output_size

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param states: [1, 4, 120, 128] Tensor
        """
        h = self.conv(states)  # [1, 512] Tensor
        h = self.flatten(h)

        policy = self.policy(h)  # [1, 7] Tensor
        value = self.critic(h)  # [1, 1] Tensor
        return policy, value


class MultiProcessEnv(Process):
    def __init__(self, process_idx: int, env: gym.Env, child_conn: Connection, input_size: Tuple[int, int],
                 render: bool = False, replay_size: int = 4):
        super(MultiProcessEnv, self).__init__()
        self.process_idx = process_idx
        self.env = env
        self.child_conn = child_conn
        self.input_size = input_size
        self.render = render

        # Game State Variables
        self.step = 0

        # Set Replay Memory
        self.replay_size = replay_size
        self.replay: np.ndarray = np.zeros([replay_size, *self.input_size])

        self.reset()

    def run(self):
        super(MultiProcessEnv, self).run()
        self.reset()
        env = self.env
        render = self.render

        while True:
            action = self.child_conn.recv()

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

    def preprocess(self, state: np.ndarray):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (self.input_size[1], self.input_size[0]))
        h = np.float32(h) / 255.
        return h

    def reset(self):
        self.step = 0

        # Initialize Replay Memory
        init_state = self.env.reset()
        for i in range(self.replay_size):
            self.replay[i, :, :] = self.preprocess(init_state)


class Agent(object):

    def __init__(self, model: nn.Module, n_action: int, cuda: bool = False):
        self.device: str = 'cuda' if cuda else 'cpu'
        self.n_action = n_action
        self.model = model.to(self.device)

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """
        :param states: preprocessed states
        :return: a list of actions -> [1, 0, 2, 3, 0, 1, ...]
        """
        states = torch.from_numpy(states).float().to(self.device)
        policy, value = self.model(states)
        softmax_policy = F.softmax(policy, dim=-1)
        actions = softmax_policy.multinomial(1).view(-1).cpu().numpy()  # 가중치에 따라서 action을 선택
        # action = np.random.choice(self.n_action, size=1, p=policy)
        return actions


class A2CAgent(Agent):
    pass


class A2CTrain(object):
    """
    Advantage Actor Critic Trainer with N-Step bootstrapping
    """

    def __init__(self, game_id: str, model: nn.Module, input_shape: Tuple[int, int], n_step: int, n_action: int,
                 n_processor: int = 1,
                 render: bool = False, replay_size: int = 4, cuda=False):
        self.game_id = game_id
        self.model = model
        self.input_shape = input_shape
        self.n_step = n_step
        self.n_action = n_action
        self.n_processor = n_processor
        self.render = render
        self.replay_size = replay_size
        self.device: str = 'cuda' if cuda else 'cpu'

        # Initialize Agent
        self.agent: A2CAgent = A2CAgent(model, n_action=n_action, cuda=cuda)

        # Initialize Environments
        self.envs: List[MultiProcessEnv] = []
        self.parent_conns: List[Connection] = []
        self.child_conns: List[Connection] = []
        self._init_envs()

        # N-Step storing variables
        self.dq_states = deque(maxlen=n_step)  # (n_step, n_processor, 4, 120, 128)
        self.dq_next_states = deque(maxlen=n_step)
        self.dq_rewards = deque(maxlen=n_step)
        self.dq_next_rewards = deque(maxlen=n_step)
        self.dq_dones = deque(maxlen=n_step)
        self.dq_actions = deque(maxlen=n_step)

    def _init_envs(self):
        for idx in range(self.n_processor):
            parent_conn, child_conn = Pipe()
            env = self.create_env()
            env_processor = MultiProcessEnv(idx, env, child_conn,
                                            input_size=self.input_shape,
                                            replay_size=self.replay_size,
                                            render=self.render)
            env_processor.start()
            self.envs.append(env_processor)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def create_env(self) -> gym.Env:
        env = gym.make(self.game_id)
        return env

    def train(self, gamma: float = 0.99):
        agent = self.agent

        # Prepare Training
        states = self.get_init_states()

        while True:
            # Get Action
            actions = agent.get_action(states)

            # Interact with environments
            self.send_actions(actions)
            next_states, rewards, dones, infos = self.receive_from_envs()

            # Train Policy and Value Networks
            self._build_training_data(states, actions, next_states, rewards, dones, infos, gamma=gamma)

    def get_init_states(self) -> np.ndarray:
        """
        :return: [n_envs, 4, 120, 128]
        """
        states = np.zeros([self.n_processor, self.replay_size, *self.input_shape])
        return states

    def send_actions(self, actions: np.ndarray):
        """
        Send actions to environments
        """
        [parent_conn.send(action) for action, parent_conn in zip(actions, self.parent_conns)]

    def receive_from_envs(self):
        """
        :return:
            - next_states: [n_envs, 4, 120, 128]
            - rewards: (n_envs, )
            - dones: (n_envs, )
            - infos: a list of dictionaries
        """
        next_states, rewards, dones, infos = [], [], [], []
        for parent_conn in self.parent_conns:
            next_state, reward, done, info = parent_conn.recv()
            next_states.append(next_state)  # [4, 120, 128]
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        next_states = np.stack(next_states)  # [n_envs, 4, 120, 128]
        rewards = np.stack(rewards)
        dones = np.stack(dones).astype(np.int16)

        return next_states, rewards, dones, infos

    def _build_training_data(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                             rewards: np.ndarray, dones: np.ndarray, infos: List[dict], gamma: float):
        """
         - policy: \pi(a_t, s_t)
         - value: V_v(s_t)
         - next_value: V(s_{t+1})
        """
        # Calculate current policy and value
        states_tensor = torch.from_numpy(states).to(self.device).float()
        policies, values = self.agent.model(states_tensor)

        # Calculate Next value
        next_states_tensor = torch.from_numpy(next_states).to(self.device).float()
        _, next_values = self.agent.model(next_states_tensor)

        # To Numpy
        policies = policies.view(-1).data.cpu().numpy()
        values = values.view(-1).data.cpu().numpy()
        next_values = next_values.view(-1).data.cpu().numpy()

        # Make target variables
        deltas = rewards + gamma * next_values * (1 - dones) - values

        # Add data to deque for N-Step training
        self.dq_actions.append(actions)
        self.dq_states.append(states)
        self.dq_next_states.append(next_states)
        self.dq_rewards.append(rewards)
        self.dq_dones.append(dones)


class A2CMarioTrain(A2CTrain):

    def create_env(self) -> gym.Env:
        env = gym_super_mario_bros.make(self.game_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        print('env observation:', env.observation_space.shape)
        return env


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    n_action = env.action_space.n
    resized_input_shape = (120, 128)
    replay_size = 4
    n_step = 5

    print('input  shape       :', env.observation_space.shape)
    print('output shape       :', env.action_space.n)
    print('resized input shape:', resized_input_shape)

    # Hyperparameters
    a2c_model = A2CModel(resized_input_shape, n_action, replay_size=replay_size)
    a2c_train = A2CMarioTrain('SuperMarioBros-v0', a2c_model, n_processor=1, render=True, cuda=True,
                              n_step=n_step, n_action=n_action, input_shape=resized_input_shape)
    a2c_train.train()


if __name__ == '__main__':
    main()
