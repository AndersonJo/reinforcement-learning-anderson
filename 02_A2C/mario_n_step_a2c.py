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
            nn.Conv2d(in_channels=4,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=96,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96,
                      out_channels=96,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU())
        conv_output_size = self._calculate_output_size(input_shape, n_history)

        self.lstm = nn.LSTMCell(conv_output_size, 512)

        # Actor
        self.policy = nn.Linear(in_features=512, out_features=n_action)

        # Critic
        self.critic = nn.Linear(in_features=512, out_features=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LSTMCell):
                torch.nn.init.constant_(module.bias_ih, 0)
                torch.nn.init.constant_(module.bias_hh, 0)

    def _calculate_output_size(self, input_shape: tuple, n_history: int) -> int:
        o = self.conv(torch.zeros(1, n_history, *input_shape))
        output_size = int(np.prod(o.size()))
        return output_size

    def forward(self,
                states: torch.Tensor,
                hidden_state: torch.Tensor,
                cell_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param hidden_state: (batch, hidden_size) -> tensor containing the initial hidden state
        :param cell_state:   (batch, hidden_size) -> tensor containing the initial cell state
        :param states: [1, 4, 120, 128] Tensor
        """
        h = self.conv(states)  # [1, 512] Tensor
        h = h.view(h.size(0), -1)
        next_hidden, next_cell = self.lstm(h, (hidden_state, cell_state))

        logits = self.policy(next_hidden)  # [1, 7] Tensor
        value = self.critic(next_hidden)  # [1, 1] Tensor
        return logits, value, next_hidden, next_cell


class MultiProcessEnv(Process):
    def __init__(self, process_idx: int, env: gym.Env, child_conn: Connection, input_size: Tuple[int, int],
                 render: bool = False, n_history: int = 4, skip: int = 4):
        super(MultiProcessEnv, self).__init__()
        self.process_idx = process_idx
        self.env = env
        self.child_conn = child_conn
        self.input_size = input_size
        self.render = render
        self.skip = skip

        # Mario Specific Variables
        self.lives = 3

        # Game State Variables
        self.stage = 1
        self.step = 0
        self._cur_score = 0

        self._reward_dq = deque(maxlen=64)
        self.actions = defaultdict(int)
        self._move_dq = deque(maxlen=500)

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

        while True:
            action = self.child_conn.recv()
            self.actions[action] += 1

            total_reward = 0
            done = None
            info = None
            for _ in range(self.skip):
                next_state, reward, done, info = env.step(action)
                self._move_dq.append(info['x_pos'])
                self.memory_state(next_state)

                if render:
                    env.render()

                reward = self.process_reward(reward, done, info)
                total_reward += reward

                self._cur_score = info['score']
                self.step += 1

                if done:
                    break

            self._reward_dq.append(total_reward)

            # Check No Move
            move_var = np.var(self._move_dq)

            # Send information to Parent Processor
            self.child_conn.send([self.history, total_reward, done, info])

            if done:
                episode += 1
                print(f'[{self.process_idx:2}] epi:{episode:4} | step:{self.step:<5} | '
                      f'reward: {round(total_reward, 2):<4} | '
                      f'mean:{round(np.mean(self._reward_dq), 2):<4} | '
                      f'move_var: {int(move_var):<6} | actions: {dict(self.actions)}')
                self.reset()

    def process_reward(self, reward, done, info):
        reward += (info['score'] - self._cur_score) / 64.

        if done:
            if info['flag_get']:
                reward += 64
            else:
                reward -= 64

        reward /= 10
        return reward

    @property
    def shape(self) -> tuple:
        return self.env.observation_space.shape

    def memory_state(self, state):
        self.history[:3, :, :] = self.history[1:, :, :]
        self.history[3, :, :] = self.preprocess(state)

    def preprocess(self, state: np.ndarray):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (self.input_size[1], self.input_size[0]))
        h = np.float32(h) / 255.
        return h

    def reset(self):
        self.env.reset()
        self.step = 0
        self._cur_score = 0
        self.lives = 3
        self.actions = defaultdict(int)
        self.stage = 1
        self._reward_dq.clear()
        self._move_dq.clear()

        # Initialize Replay Memory
        init_state = self.env.reset()
        for i in range(self.n_history):
            self.history[i, :, :] = self.preprocess(init_state)


class Agent(object):

    def __init__(self, model: nn.Module, n_action: int, learning_rate=0.0001, cuda: bool = True):
        self.device: str = 'cuda' if cuda else 'cpu'
        self.n_action = n_action
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def get_action(self,
                   states: np.ndarray,
                   h_s: np.ndarray,
                   c_s: np.ndarray) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        """
        :param h_s: hidden state for LSTM
        :param c_s: cell state for LSTM
        :param states: preprocessed states
        """
        states = torch.from_numpy(states).float().to(self.device)
        hidden_tensor = torch.from_numpy(h_s).float().to(self.device)
        cell_tensor = torch.from_numpy(c_s).float().to(self.device)

        logits, value, h_next, c_next = self.model(states, hidden_tensor, cell_tensor)
        softmax_policy = F.softmax(logits, dim=-1)
        actions = softmax_policy.multinomial(1).cpu().numpy()  # 가중치에 따라서 action을 선택

        # h_next = h_next.view(-1, 512).cpu().detach().numpy()
        # c_next = c_next.view(-1, 512).cpu().detach().numpy()

        return actions, logits, value, h_next, c_next

    def predict_transition(self, states: np.ndarray, h_state, c_state) -> Tuple[np.ndarray, np.ndarray]:
        """
         - policy: \pi(a_t, s_t)
         - value: V(s_t)
        """
        # Calculate current policy and value
        states_tensor = torch.from_numpy(states).to(self.device).float()
        h_tensor = torch.from_numpy(h_state).to(self.device).float()
        c_tensor = torch.from_numpy(c_state).to(self.device).float()
        pred_logits, pred_values, h_next, c_next = self.model(states_tensor, h_tensor, c_tensor)
        # pred_policies = pred_policies.view(-1).data.cpu().numpy()
        # pred_values = pred_values.view(-1).data.cpu().numpy()

        return pred_logits, pred_values

    def predict_last_value(self, states: np.ndarray, h_state, c_state,
                           n_step, n_processor) -> Tuple[np.ndarray, np.ndarray]:
        last_indices = [n_step * i - 1 for i in range(1, n_processor + 1)]
        last_states = states[last_indices]
        last_h_states = h_state[last_indices]
        last_c_states = c_state[last_indices]

        return self.predict_transition(last_states, last_h_states, last_c_states)

    def train(self, group_states: np.ndarray, group_h: np.ndarray, group_c: np.ndarray,
              group_actions, group_critic_y, group_actor_y, entropy_coef=0.01):
        with torch.no_grad():  # It makes tensors set "requires_grad" to be false
            state_tensor = torch.FloatTensor(group_states).to(self.device)
            h_tensor = torch.FloatTensor(group_h).to(self.device)
            c_tensor = torch.FloatTensor(group_c).to(self.device)

            # next_states_tensor = torch.FloatTensor(group_next_states).to(self.device)
            # next_h_tensor = torch.FloatTensor(group_next_h).to(self.device)
            # next_c_tensor = torch.FloatTensor(group_next_c).to(self.device)

            actions = torch.LongTensor(group_actions).to(self.device)
            critic_y = torch.FloatTensor(group_critic_y).to(self.device)
            actor_y = torch.FloatTensor(group_actor_y).to(self.device)

        pred_logits, pred_value, next_h, next_c = self.model(state_tensor, h_tensor, c_tensor)
        softmax_policy = F.softmax(pred_logits, dim=-1)
        log_policy = F.log_softmax(pred_logits, dim=-1)
        m = Categorical(softmax_policy)

        # Actor loss
        actor_loss = -m.log_prob(actions) * actor_y

        # Entorpy
        # entropy = m.entropy()  # <- it's the same as the following code
        entropy = -(softmax_policy * log_policy).sum(-1, keepdim=True)

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
                 checkpoint: str = None):
        self.game_id = game_id
        self.model = model
        self.input_shape = input_shape
        self.n_step = n_step
        self.n_action = n_action
        self.n_processor = n_processor
        self.render = render
        self.n_history = n_history
        self.device: str = 'cuda' if cuda else 'cpu'

        # Make checkpoint directory
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            checkpoint_path = os.path.dirname(checkpoint)
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

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
        self.dq_h_states = deque(maxlen=n_step)
        self.dq_c_states = deque(maxlen=n_step)
        self.dq_rewards = deque(maxlen=n_step)
        self.dq_next_rewards = deque(maxlen=n_step)
        self.dq_next_h_states = deque(maxlen=n_step)
        self.dq_next_c_states = deque(maxlen=n_step)
        self.dq_dones = deque(maxlen=n_step)
        self.dq_actions = deque(maxlen=n_step)
        self.dq_infos = deque(maxlen=n_step)

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
        hidden_0 = np.zeros([self.n_processor, 512], dtype=np.float)
        cell_0 = np.zeros([self.n_processor, 512], dtype=np.float)
        return states, hidden_0, cell_0

    def test(self):
        self._init_envs(1)
        agent = self.agent

        # Prepare Test
        step = -1
        states, h_state, c_state = self.initialize_states()

        while True:
            action, h_state, c_state = agent.get_action(states, h_state, c_state)

            # Interact with environments
            self.send_actions(action)
            next_states, rewards, dones, infos = self.receive_from_envs()

            sleep(0.01)

    def train(self, gamma: float = 0.9, lambda_: float = 0.95):
        self._init_envs(self.n_processor)

        agent = self.agent

        # Prepare Training
        step = -1
        states, h_state, c_state = self.initialize_states()

        while True:
            self.clean_queues()
            step += 1

            for _ in range(self.n_step):
                # Get Action
                res = agent.get_action(states, h_state, c_state)
                actions = res[0]
                logits = res[1]
                value = res[2]
                h_next = res[3]
                c_next = res[4]

                # Interact with environments
                self.send_actions(actions)
                next_states, rewards, dones, infos = self.receive_from_envs()

                # Store environment data
                self._store_data(actions, states, logits, value, h_state, c_state,
                                 next_states, h_next, c_next,
                                 rewards, dones, infos)
                h_state = h_next
                c_state = c_next

                # Update states <- next_states
                states = next_states[:, :, :, :]

            # Train Policy and Value Networks
            target_data = self._build_target_data(gamma=gamma, lambda_=lambda_)

            self.agent.train(*target_data)

            if step % 10 == 0:
                print('Saved!')
                torch.save(agent.model.state_dict(), self.checkpoint)

    def _store_data(self, actions: np.ndarray, states: np.ndarray,
                    logits: torch.Tensor, value: torch.Tensor,
                    h_state: torch.Tensor, c_state: torch.Tensor,
                    next_states: np.ndarray, h_next: np.ndarray, c_next: np.ndarray,
                    rewards: np.ndarray, dones: np.ndarray, infos: list):

        policy = F.softmax(logits, dim=-1)
        log_policy = F.log_softmax(logits, dim=-1)
        entropy = (policy * log_policy).sum(1, keepdim=True)
        import ipdb
        ipdb.set_trace()

        self.dq_states.append(states)
        self.dq_h_states.append(h_state)
        self.dq_c_states.append(c_state)
        self.dq_actions.append(actions)
        self.dq_next_states.append(next_states)
        self.dq_next_h_states.append(h_next)
        self.dq_next_c_states.append(c_next)
        self.dq_rewards.append(rewards)
        self.dq_dones.append(dones)
        self.dq_infos.append(infos)

    def _retrieve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                      np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state_shape = [-1, self.n_history, *self.input_shape]

        group_states = np.array(self.dq_states)  # shape: (step, processor, history, h, w)
        group_states = group_states.transpose([1, 0, 2, 3, 4])  # shape: (processor, step, history, h, w)
        group_states = group_states.reshape(*state_shape)  # shape: (processor * step, history, h, w)

        group_next_states = np.array(self.dq_next_states)  # shape: (step, processor, history, h, w)
        group_next_states = group_next_states.transpose([1, 0, 2, 3, 4])  # shape: (processor, step, history, h, w)
        group_next_states = group_next_states.reshape(*state_shape)  # shape: (processor * step, history, h, w)

        group_rewards = np.array(self.dq_rewards).T.reshape(-1)
        group_actions = np.array(self.dq_actions).T.reshape(-1)
        group_dones = np.array(self.dq_dones).T.reshape(-1)

        group_h = np.array(self.dq_h_states)  # (history, processor, 512)
        group_h = group_h.transpose([1, 0, 2])  # (processor, history, 512)
        group_h = group_h.reshape([-1, 512])  # (processor * history, 512)

        group_c = np.array(self.dq_c_states)  # (history, processor, 512)
        group_c = group_c.transpose([1, 0, 2])  # (processor, history, 512)
        group_c = group_c.reshape([-1, 512])  # (processor * history, 512)

        group_next_h = np.array(self.dq_next_h_states)  # (history, processor, 512)
        group_next_h = group_next_h.transpose([1, 0, 2])  # (processor, history, 512)
        group_next_h = group_next_h.reshape([-1, 512])  # (processor * history, 512)

        group_next_c = np.array(self.dq_next_c_states)  # (history, processor, 512)
        group_next_c = group_next_c.transpose([1, 0, 2])  # (processor, history, 512)
        group_next_c = group_next_c.reshape([-1, 512])  # (processor * history, 512)

        return (group_states, group_h, group_c, group_next_states, group_next_h, group_next_c,
                group_rewards, group_actions, group_dones)

    def send_actions(self, actions: np.ndarray):
        """
        Send actions to environments
        """
        actions = actions.reshape(-1)
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

    def _build_target_data(self,
                           gamma: float,
                           lambda_: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        # Retrieve grouped environment data : N-Step = Group
        _res = self._retrieve_data()
        group_states = _res[0]
        group_h = _res[1]
        group_c = _res[2]
        group_next_states = _res[3]
        group_next_h = _res[4]
        group_next_c = _res[5]
        group_rewards = _res[6]
        group_actions = _res[7]
        group_dones = _res[8]

        # predict transitions
        pred_logits, pred_values = self.agent.predict_transition(group_states, group_h, group_c)
        # _, pred_next_values = self.agent.predict_transition(group_next_states, group_next_h, group_next_c)
        _, next_values = self.agent.predict_last_value(group_next_states, group_next_h, group_next_c, self.n_step,
                                                       self.n_processor)

        # Build Target
        group_critic_y = []
        group_actor_y = []
        for idx in range(self.n_processor):
            # r_{t+1}, r_{t+2}, ... r_{t+4}
            _rewards = group_rewards[idx * self.n_step: (idx + 1) * self.n_step]

            # V(s_t), V(s_{t+1}), ..., V(s_{t+4})
            _pred_values = pred_values[idx * self.n_step: (idx + 1) * self.n_step]

            # V(s_{t+1}), V(s_{t+2}), ..., V(s_{t+5})
            # _pred_next_values = pred_next_values[idx * self.n_step: (idx + 1) * self.n_step]
            next_value = next_values[idx]

            # d_{t+1, d_{t+2}, ..., d_{t+5}
            _dones = group_dones[idx * self.n_step: (idx + 1) * self.n_step]

            critic_y, actor_y = self._build_targets(_rewards, _pred_values, next_value,
                                                    _dones, gamma=gamma, lambda_=lambda_)

            group_critic_y.append(critic_y)
            group_actor_y.append(actor_y)

        group_critic_y = np.hstack(group_critic_y)
        group_actor_y = np.hstack(group_actor_y)

        return group_states, group_h, group_c, group_actions, group_critic_y, group_actor_y

    def _build_targets(self,
                       rewards: np.ndarray,
                       pred_values: torch.Tensor,
                       next_value: torch.Tensor,
                       dones: np.ndarray,
                       gamma: float, lambda_: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        It receives environment data of a single processor
        :param rewards: an array of the n-step rewards r_t from a single processor
        :param pred_values: an array of the n-step V(s_t) from a single processor
        :param pred_next_values: an array of the n-step V(s_{t+1}) from a single processor
        :param dones:
        """
        critic_y = np.zeros(self.n_step)  # discounted rewards

        # Generalized Advantage Function
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        for t in range(self.n_step - 1, -1, -1):
            gae = gae * gamma
            delta = rewards[t] + gamma * next_value.detach() * (1 - dones[t]) - pred_values[t]
            gae = gae * delta
            next_value = pred_values[t]

            actor_loss

            gae = gamma * lambda_ * gae * (1 - dones[t]) + delta
            critic_y[t] = gae + pred_values[t]

        # N-Step Bootstrapping
        # _next_value = pred_next_values[-1]
        # for t in range(self.n_step - 1, -1, -1):
        #     # 1-step TD: V(s_t) r_t + \gamma V(s_{t+1}) - V(s_t)
        #     _next_value = rewards[t] + gamma * _next_value * (1 - dones[t])
        #     critic_y[t] = _next_value

        actor_y = critic_y - pred_values

        return critic_y, actor_y

    def clean_queues(self):
        self.dq_actions.clear()
        self.dq_rewards.clear()
        self.dq_next_states.clear()
        self.dq_states.clear()
        self.dq_dones.clear()
        self.dq_infos.clear()


class A2CSuperMario(A2C):

    def create_env(self) -> gym.Env:
        while True:
            try:
                world = 1
                stage = np.random.randint(1, 4)

                env_id = f'SuperMarioBros-{world}-{stage}-v0'
                env = JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT)
            except UnregisteredEnv as e:
                continue
            break
        return env


def main():
    args = get_args()

    env_id = 'SuperMarioBros-v0'
    env = JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT)
    n_action = env.action_space.n
    resized_input_shape = (84, 84)
    n_processor = 18
    n_history = 4
    n_step = 64

    print('n_processor        :', n_processor)
    print('input  shape       :', env.observation_space.shape)
    print('output shape       :', env.action_space.n)
    print('resized input shape:', resized_input_shape)

    # Hyperparameters
    a2c_model = A2CModel(resized_input_shape, n_action, n_history=n_history)
    a2c_breakout = A2CSuperMario('SuperMarioBros-v0', a2c_model, n_processor=n_processor, render=True, cuda=True,
                                 n_step=n_step, n_action=n_action, input_shape=resized_input_shape,
                                 checkpoint=args.checkpoint)

    if args.mode == 'train':
        a2c_breakout.train()
    elif args.mode == 'test':
        a2c_breakout.test()


if __name__ == '__main__':
    main()
