import os
import shutil
from argparse import ArgumentParser
from collections import deque, defaultdict
from multiprocessing.connection import Connection
from time import sleep
from typing import List, Tuple, Callable

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.error import UnregisteredEnv
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

from nes_py.wrappers import JoypadSpace
from torch import optim
from torch.distributions import Categorical
from torch.multiprocessing import Process, Pipe


def get_args():
    parser = ArgumentParser()
    parser.add_argument('mode', default='test', help='train | test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mario.model')
    args = parser.parse_args()

    assert args.mode in ['test', 'train']
    return args


class A2CModel(nn.Module):
    def __init__(self, input_shape, n_action: int, n_history: int):
        super(A2CModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4,  # history frame
                      out_channels=32,
                      kernel_size=4,
                      stride=3,
                      padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ELU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=64,
                      out_channels=96,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ELU()
        )

        conv_output_size = self._calculate_output_size(input_shape, n_history)

        self.lstm = nn.LSTMCell(conv_output_size, 512)

        # Actor
        self.policy = nn.Linear(in_features=512, out_features=n_action)

        # Critic
        self.critic = nn.Linear(in_features=512, out_features=1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                print('CNN initialized')
                torch.nn.init.xavier_uniform_(p.weight)
                torch.nn.init.constant_(p.bias, 0)

            elif isinstance(p, nn.Linear):
                print('Linear initialized')
                torch.nn.init.xavier_uniform_(p.weight)
                torch.nn.init.constant_(p.bias, 0)

            elif isinstance(p, nn.LSTMCell):
                print('LSTM initialized')
                torch.nn.init.constant_(p.bias_ih, 0)
                torch.nn.init.constant_(p.bias_hh, 0)

    def _calculate_output_size(self, input_shape: tuple, n_history: int) -> int:
        o = self.conv(torch.zeros(1, n_history, *input_shape))
        output_size = int(np.prod(o.size()))
        return output_size

    def forward(self, states: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param hx: (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        :param cx: (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        :param states: [1, 4, 120, 128] Tensor
        """
        h = self.conv(states)  # [1, 512] Tensor
        h = h.view(h.size(0), -1)  # flatten
        next_hx, next_cx = self.lstm(h, (hx, cx))

        logits = self.policy(hx)  # [1, 7] Tensor
        value = self.critic(hx)  # [1, 1] Tensor
        return logits, value, next_hx, next_cx


class MultiProcessEnv(Process):
    def __init__(self, process_idx: int, env: gym.Env, child_conn: Connection, input_size: Tuple[int, int],
                 render: bool = False, n_history: int = 4):
        super(MultiProcessEnv, self).__init__()
        self.process_idx = process_idx
        self.env = env
        self.child_conn = child_conn
        self.input_size = input_size
        self.render = render
        self.skip = 4

        # Mario Specific Variables
        self.lives = 2

        # Game State Variables
        self.stage = 1
        self.step = 0
        self._reward = 0
        self._cur_score = 0
        self._score_dq = deque(maxlen=100)
        self._reward_dq = deque(maxlen=100)
        self._move_dq = deque(maxlen=500)
        self.actions = defaultdict(int)

        # Set Replay Memory
        self.n_history = n_history
        self.history: np.ndarray = np.zeros([n_history, *self.input_size])

        self.reset()

    def run(self):
        super(MultiProcessEnv, self).run()
        self.reset()
        env = self.env
        render = self.render
        episode = 0
        prev_x_pos = 0
        while True:
            action = self.child_conn.recv()
            done = False
            info = None
            move_var = 0
            total_reward = 0
            for _ in range(self.skip):
                self.actions[action] += 1

                if render:
                    env.render()

                next_state, reward, done, info = env.step(action)
                score = (info['score'] - self._cur_score) / 50.
                reward += score

                if score > 0:
                    self._score_dq.append(score)

                if done:
                    if info['flag_get']:
                        reward += 50.
                        self.stage = info['stage']
                    else:
                        reward -= 50

                self.memory_state(next_state)

                # Check No Move
                self._move_dq.append(info['x_pos'])
                move_var = np.var(self._move_dq)
                if (len(self._move_dq) >= self._move_dq.maxlen) and move_var < 3:
                    reward -= 50
                    done = True

                # Normalize reward
                reward /= 15
                self._reward += reward
                total_reward += reward
                self.step += 1

                if done:
                    break

            # Send information to Parent Processor
            self.child_conn.send([self.history, total_reward, done, info])
            self._reward_dq.append(self._reward)

            if done:
                episode += 1

                print(f'[{self.process_idx:2}] epi:{episode:5} | step:{self.step:<4} | '
                      f'acum:{int(self._reward):<4} | '
                      f'mean:{int(np.mean(self._reward_dq)):<4} | '
                      f'last:{round(reward, 1):<4} | '
                      f'score:{int(np.mean(self._score_dq)):<2} | '
                      f'move:{int(move_var):<6} | act:{dict(self.actions)}')
                self.reset()

            prev_x_pos = info['x_pos']
            self.stage = info['stage']
            self._cur_score = info['score']

    # def is_done(self, info):
    #     if info['life'] < self.lives or info['life'] < 0:
    #         self.lives = info['life']
    #         return True
    #     return False

    @property
    def shape(self) -> tuple:
        return self.env.observation_space.shape

    def memory_state(self, state):
        idx = self.n_history - 1
        self.history[:idx, :, :] = self.history[1:, :, :]
        self.history[idx, :, :] = self.preprocess(state)

    def preprocess(self, state: np.ndarray):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (self.input_size[1], self.input_size[0]))
        h = np.float32(h) / 255
        return h

    def reset(self):
        self.env.reset()
        self.step = 0
        self._cur_score = 0
        self._reward = 0
        self.lives = 2
        self.actions = defaultdict(int)
        self.stage = 1
        self._score_dq.clear()
        self._reward_dq.clear()
        self._move_dq.clear()

        self._score_dq.append(0)

        # Initialize Replay Memory
        init_state = self.env.reset()
        for i in range(self.n_history):
            self.history[i, :, :] = self.preprocess(init_state)


class Agent(object):

    def __init__(self, model: nn.Module, n_action: int, learning_rate=0.0002, cuda: bool = True):
        self.device: str = 'cuda' if cuda else 'cpu'
        self.n_action = n_action
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def get_action(self, states: np.ndarray, h_0: np.ndarray, c_0: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        :param h_0: hidden state for LSTM
        :param c_0: cell state for LSTM
        :param states: preprocessed states
        :return: a list of actions -> [1, 0, 2, 3, 0, 1, ...]
        """
        states = torch.from_numpy(states).float().to(self.device)
        h_0 = torch.from_numpy(h_0).float().to(self.device)
        c_0 = torch.from_numpy(c_0).float().to(self.device)

        logits, value, next_hidden, next_cell = self.model(states, h_0, c_0)
        policy = F.softmax(logits, dim=-1)
        actions = policy.multinomial(1).view(-1).cpu().numpy()  # 가중치에 따라서 action을 선택

        next_hidden = next_hidden.cpu().detach().numpy()
        next_cell = next_cell.cpu().detach().numpy()
        return actions, next_hidden, next_cell

    def predict_transition(self, states: np.ndarray, h_0: np.ndarray, c_0: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
         - policy: \pi(a_t, s_t)
         - value: V(s_t)
        """
        # Calculate current policy and value
        states_tensor = torch.from_numpy(states).to(self.device).float()
        h_0 = torch.from_numpy(h_0).float().to(self.device)
        c_0 = torch.from_numpy(c_0).float().to(self.device)

        pred_policies, pred_values, next_hidden, next_cell = self.model(states_tensor, h_0, c_0)
        pred_policies = pred_policies.view(-1).data.cpu().numpy()
        pred_values = pred_values.view(-1).data.cpu().numpy()
        next_hidden = next_hidden.data.cpu().numpy()
        next_cell = next_cell.data.cpu().numpy()
        return pred_policies, pred_values, next_hidden, next_cell

    def train(self, group_states, group_next_states, group_actions, group_critic_y, group_actor_y,
              group_hidden, group_cell, group_next_hidden, group_next_cell, entropy_coef=0.02):
        with torch.no_grad():  # It makes tensors set "requires_grad" to be false
            states = torch.FloatTensor(group_states).to(self.device)
            next_states = torch.FloatTensor(group_next_states).to(self.device)
            actions = torch.LongTensor(group_actions).to(self.device)
            critic_y = torch.FloatTensor(group_critic_y).to(self.device)
            actor_y = torch.FloatTensor(group_actor_y).to(self.device)
            group_hidden = torch.FloatTensor(group_hidden).to(self.device)
            group_cell = torch.FloatTensor(group_cell).to(self.device)

        pred_policy, pred_value, next_h, next_c = self.model(states, group_hidden, group_cell)
        m = Categorical(F.softmax(pred_policy, dim=-1))

        # Actor loss
        actor_loss = -m.log_prob(actions) * actor_y

        # Entorpy
        entropy = m.entropy()

        # Critic loss
        critic_loss = self.mse(pred_value.view(-1), critic_y)

        # Loss
        loss = actor_loss.mean() + 0.5 * critic_loss - entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()


class A2CAgent(Agent):
    pass


class A2C(object):
    """
    Advantage Actor Critic Trainer with N-Step bootstrapping
    """

    def __init__(self, game_id: str, model: nn.Module, input_shape: Tuple[int, int], n_step: int, n_action: int,
                 n_processor: int = 1, render: bool = False, n_history: int = 4, cuda=False,
                 checkpoint: str = None, process_reward: Callable = None):
        self.game_id = game_id
        self.model = model
        self.input_shape = input_shape
        self.n_step = n_step
        self.n_action = n_action
        self.n_processor = n_processor
        self.render = render
        self.n_history = n_history
        self.skip = 4
        self.device: str = 'cuda' if cuda else 'cpu'

        # Make checkpoint directory
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            checkpoint_path = os.path.dirname(checkpoint)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

        # Functions
        self.process_reward = process_reward

        # Initialize Agent
        self.agent: A2CAgent = A2CAgent(model, n_action=n_action, cuda=cuda)
        if self.checkpoint is not None and self.checkpoint.endswith('.model'):
            if os.path.exists(self.checkpoint):
                print(f'{self.checkpoint} has been loaded')
                self.agent.model.load_state_dict(torch.load(self.checkpoint))

        # Initialize Environments
        self.envs: List[MultiProcessEnv] = []
        self.parent_conns: List[Connection] = []
        self.child_conns: List[Connection] = []

        # N-Step storing variables
        self.dq_states = deque(maxlen=n_step)  # (n_step, n_processor, 4, 120, 128)
        self.dq_next_states = deque(maxlen=n_step)
        self.dq_rewards = deque(maxlen=n_step)
        self.dq_next_rewards = deque(maxlen=n_step)
        self.dq_dones = deque(maxlen=n_step)
        self.dq_actions = deque(maxlen=n_step)
        self.dq_infos = deque(maxlen=n_step)
        self.dq_hidden_state = deque(maxlen=n_step)
        self.dq_cell_state = deque(maxlen=n_step)
        self.dq_next_hidden_state = deque(maxlen=n_step)
        self.dq_next_cell_state = deque(maxlen=n_step)

    def _init_envs(self, n_processor: int):
        for idx in range(n_processor):
            parent_conn, child_conn = Pipe()
            env = self.create_env()
            render = False
            if idx == 0:
                render = True

            env_processor = MultiProcessEnv(idx, env, child_conn,
                                            input_size=self.input_shape,
                                            n_history=self.n_history,
                                            render=render)
            env_processor.start()
            self.envs.append(env_processor)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def create_env(self) -> gym.Env:
        env = gym.make(self.game_id)
        return env

    def initialize_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :return: [n_envs, 4, 120, 128]
        """
        states = np.zeros([self.n_processor, self.n_history, *self.input_shape])
        h_0 = np.zeros([self.n_processor, 512], dtype=np.float)  # initial hidden state for LSTM
        c_0 = np.zeros([self.n_processor, 512], dtype=np.float)  # initial cell state for LSTM
        return states, h_0, c_0

    def test(self):
        self._init_envs(1)
        agent = self.agent

        # Prepare Test
        step = -1
        states, h_0, c_0 = self.initialize_states()

        while True:
            action, h_0, c_0 = agent.get_action(states, h_0, c_0)

            # Interact with environments
            self.send_actions(action)
            next_states, rewards, dones, infos = self.receive_from_envs()

            # Update states <- next_states
            states = next_states[:, :, :, :]

            sleep(0.01)

    def train(self, gamma: float = 0.99, lambda_: float = 0.95):
        self._init_envs(self.n_processor)

        agent = self.agent

        # Prepare Training
        step = -1
        states, h_0, c_0 = self.initialize_states()

        while True:
            self.clean_queues()
            step += 1

            for _ in range(self.n_step):
                # Get Action
                actions, next_h, next_c = agent.get_action(states, h_0, c_0)

                # Interact with environments
                self.send_actions(actions)
                next_states, rewards, dones, infos = self.receive_from_envs()
                rewards = self.process_reward(states, actions, next_states, rewards, dones, infos)

                # Store environment data
                self._store_data(states, actions, next_states, rewards, dones, infos, h_0, c_0, next_h, next_c)

                # Update states <- next_states
                states = next_states[:, :, :, :]
                h_0 = next_h
                c_0 = next_c

            # Train Policy and Value Networks
            target_data = self._build_target_data(gamma=gamma, lambda_=lambda_)

            self.agent.train(*target_data)

            if step % 500 == 0:
                torch.save(agent.model.state_dict(), self.checkpoint)
                print('saved model')

    def _store_data(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                    dones: np.ndarray, infos: list,
                    h_0: np.ndarray, c_0: np.ndarray, next_h: np.ndarray, next_c: np.ndarray):
        self.dq_states.append(states)
        self.dq_actions.append(actions)
        self.dq_next_states.append(next_states)
        self.dq_rewards.append(rewards)
        self.dq_dones.append(dones)
        self.dq_infos.append(infos)
        self.dq_hidden_state.append(h_0)
        self.dq_cell_state.append(c_0)
        self.dq_next_hidden_state.append(next_h)
        self.dq_next_cell_state.append(next_c)

    def _retrieve_data(self) -> List[np.ndarray]:
        state_shape = [-1, self.n_history, *self.input_shape]
        lstm_shape = [-1, 512]

        group_states = np.array(self.dq_states)  # shape: (step, processor, history, h, w)
        group_states = group_states.transpose([1, 0, 2, 3, 4])  # shape: (processor, step, history, h, w)
        group_states = group_states.reshape(*state_shape)  # shape: (processor * step, history, h, w)

        group_next_states = np.array(self.dq_next_states)  # shape: (step, processor, history, h, w)
        group_next_states = group_next_states.transpose([1, 0, 2, 3, 4])  # shape: (processor, step, history, h, w)
        group_next_states = group_next_states.reshape(*state_shape)  # shape: (processor * step, history, h, w)

        group_rewards = np.array(self.dq_rewards).T.reshape(-1)
        group_actions = np.array(self.dq_actions).T.reshape(-1)
        group_dones = np.array(self.dq_dones).T.reshape(-1)

        group_hidden_states = np.array(self.dq_hidden_state).reshape(lstm_shape)
        group_cell_states = np.array(self.dq_cell_state).reshape(lstm_shape)
        group_next_hidden_states = np.array(self.dq_next_hidden_state).reshape(lstm_shape)
        group_next_cell_states = np.array(self.dq_next_cell_state).reshape(lstm_shape)

        return [group_states, group_next_states, group_rewards, group_actions, group_dones,
                group_hidden_states, group_cell_states, group_next_hidden_states, group_next_cell_states]

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
            next_states.append(next_state)  # [4*4, 84, 84]
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        next_states = np.stack(next_states)  # [n_envs, 4*4, 84, 84]
        rewards = np.stack(rewards)
        dones = np.stack(dones).astype(np.int16)

        return next_states, rewards, dones, infos

    def _build_target_data(self, gamma: float, lambda_: float):
        # Retrieve grouped environment data : N-Step = Group
        res = self._retrieve_data()
        group_states = res[0]
        group_next_states = res[1]
        group_rewards = res[2]
        group_actions = res[3]
        group_dones = res[4]
        group_hidden = res[5]
        group_cell = res[6]
        group_next_hidden = res[7]
        group_next_cell = res[8]

        # predict transitions
        pred_policies, pred_values, _, _ = self.agent.predict_transition(group_states, group_hidden, group_cell)
        _, pred_next_values, _, _ = self.agent.predict_transition(group_next_states, group_next_hidden, group_next_cell)

        # Build Target
        group_critic_y = []
        group_actor_y = []
        for idx in range(self.n_processor):
            # r_{t+1}, r_{t+2}, ... r_{t+4}
            _rewards = group_rewards[idx * self.n_step: (idx + 1) * self.n_step]

            # V(s_t), V(s_{t+1}), ..., V(s_{t+4})
            _pred_values = pred_values[idx * self.n_step: (idx + 1) * self.n_step]

            # V(s_{t+1}), V(s_{t+2}), ..., V(s_{t+5})
            _pred_next_values = pred_next_values[idx * self.n_step: (idx + 1) * self.n_step]

            # d_{t+1, d_{t+2}, ..., d_{t+5}
            _dones = group_dones[idx * self.n_step: (idx + 1) * self.n_step]

            critic_y, actor_y = self._build_targets(_rewards, _pred_values, _pred_next_values,
                                                    _dones, gamma=gamma, lambda_=lambda_)

            group_critic_y.append(critic_y)
            group_actor_y.append(actor_y)

        group_critic_y = np.hstack(group_critic_y)
        group_actor_y = np.hstack(group_actor_y)

        return [group_states, group_next_states, group_actions, group_critic_y, group_actor_y,
                group_hidden, group_cell, group_next_hidden, group_next_cell]

    def _build_targets(self, rewards: np.ndarray, pred_values: np.ndarray, pred_next_values: np.ndarray,
                       dones: np.ndarray, gamma: float, lambda_: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        It receives environment data of a single processor
        :param rewards: an array of the n-step rewards r_t from a single processor
        :param pred_values: an array of the n-step V(s_t) from a single processor
        :param pred_next_values: an array of the n-step V(s_{t+1}) from a single processor
        :param dones:
        """

        # N-Step Bootstrapping
        critic_y = np.zeros(self.n_step)  # discounted rewards
        _next_value = pred_next_values[-1]
        for t in range(self.n_step - 1, -1, -1):
            # 1-step TD: V(s_t) r_t + \gamma V(s_{t+1}) - V(s_t)
            _next_value = rewards[t] + gamma * _next_value * (1 - dones[t])
            critic_y[t] = _next_value
        actor_y = critic_y - pred_values

        return critic_y, actor_y

    def clean_queues(self):
        self.dq_actions.clear()
        self.dq_rewards.clear()
        self.dq_next_states.clear()
        self.dq_states.clear()
        self.dq_dones.clear()
        self.dq_infos.clear()


class A2CBreakout(A2C):
    _RUNNING_ENVS = {}

    def create_env(self) -> gym.Env:
        envs = [
            'SuperMarioBros-1-1-v0',
            'SuperMarioBros-1-2-v0',
            'SuperMarioBros-1-3-v0',
            'SuperMarioBros-1-4-v0',
            'SuperMarioBros-2-1-v0',
            'SuperMarioBros-2-3-v0',
            'SuperMarioBros-2-4-v0',
            'SuperMarioBros-3-1-v0',
            'SuperMarioBros-3-2-v0',
            'SuperMarioBros-3-3-v0',
            'SuperMarioBros-3-4-v0',
            'SuperMarioBros-4-1-v0',
            'SuperMarioBros-4-2-v0',
            'SuperMarioBros-4-3-v0',
            'SuperMarioBros-4-4-v0',
            'SuperMarioBros-5-1-v0',
            'SuperMarioBros-5-2-v0',
            'SuperMarioBros-5-3-v0',
            'SuperMarioBros-5-4-v0',
            'SuperMarioBros-6-1-v0',
            'SuperMarioBros-6-2-v0',
            'SuperMarioBros-6-3-v0',
            'SuperMarioBros-6-4-v0',
            'SuperMarioBros-7-1-v0',
            'SuperMarioBros-7-3-v0',
            'SuperMarioBros-7-4-v0',
            'SuperMarioBros-8-1-v0',
            'SuperMarioBros-8-2-v0',
            'SuperMarioBros-8-3-v0',
            'SuperMarioBros-8-4-v0',
        ]

        while True:
            try:
                env_id = np.random.choice(envs)
                if env_id in self._RUNNING_ENVS:
                    continue
                env = JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT)
            except UnregisteredEnv as e:
                continue
            break
        return env


def process_reward(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                   dones: np.ndarray, infos: list) -> np.ndarray:
    return rewards


def main():
    args = get_args()

    env_id = 'SuperMarioBros-v0'
    env = JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT)
    n_action = env.action_space.n
    resized_input_shape = (84, 84)
    n_processor = 18
    n_history = 4
    n_step = 5

    print('n_processor        :', n_processor)
    print('input  shape       :', env.observation_space.shape)
    print('output shape       :', env.action_space.n)
    print('resized input shape:', resized_input_shape)

    # Hyperparameters
    a2c_model = A2CModel(resized_input_shape, n_action, n_history=n_history)
    a2c_breakout = A2CBreakout('SuperMarioBros-v0', a2c_model, n_processor=n_processor, render=True, cuda=True,
                               n_step=n_step, n_action=n_action, input_shape=resized_input_shape,
                               process_reward=process_reward, checkpoint=args.checkpoint)

    if args.mode == 'train':
        a2c_breakout.train()
    elif args.mode == 'test':
        a2c_breakout.test()


if __name__ == '__main__':
    main()
