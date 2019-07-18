from abc import abstractmethod
from collections import deque
from multiprocessing.connection import Connection
from typing import List, Tuple, Callable

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch import optim
from torch.distributions import Categorical
from torch.multiprocessing import Process, Pipe


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.view(input.size(0), -1)


class A2CModel(nn.Module):
    def __init__(self, input_shape, n_action: int, n_history: int):
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
        conv_output_size = self._calculate_output_size(input_shape, n_history)

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
                # torch.nn.init.xavier_uniform_(p.weight)
                nn.init.kaiming_uniform_(p.weight, a=1.)
                p.bias.data.zero_()

    def _calculate_output_size(self, input_shape: tuple, n_history: int) -> int:
        o = self.conv(torch.zeros(1, n_history, *input_shape))
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
                 render: bool = False, n_history: int = 4):
        super(MultiProcessEnv, self).__init__()
        self.process_idx = process_idx
        self.env = env
        self.child_conn = child_conn
        self.input_size = input_size
        self.render = render

        # Game State Variables
        self.step = 0
        self._reward = 0

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

            if render:
                env.render()

            next_state, reward, done, info = env.step(action)
            done = self.is_done(info)

            self.memory_state(next_state)
            self.step += 1
            self._reward += reward

            # Send information to Parent Processor
            self.child_conn.send([self.history, reward, done, info])

            # Check
            # if info['life'] <= 0:
            #     done = True

            if done:
                episode += 1
                print(f'[{self.process_idx}] episode:{episode} | step:{self.step} | reward: {self._reward}')
                self.reset()

    def is_done(self, info):
        if 'ale.lives' in info:
            if info['ale.lives'] <= 0:
                return True
        return False

    @property
    def shape(self) -> tuple:
        return self.env.observation_space.shape

    def memory_state(self, state):
        self.history[:3, :, :] = self.history[1:, :, :]
        self.history[3, :, :] = self.preprocess(state)

    def preprocess(self, state: np.ndarray):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (self.input_size[1], self.input_size[0]))
        h = np.float32(h) / 255
        # h = h[:, :, 0] * 0.299 + h[:, :, 1] * 0.587 + h[:, :, 2] * 0.114
        return h

    def reset(self):
        self.step = 0
        self._reward = 0

        # Initialize Replay Memory
        init_state = self.env.reset()
        for i in range(self.n_history):
            self.history[i, :, :] = self.preprocess(init_state)


class Agent(object):

    def __init__(self, model: nn.Module, n_action: int, learning_rate=0.0002, cuda: bool = False):
        self.device: str = 'cuda' if cuda else 'cpu'
        self.n_action = n_action
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def get_action(self, states: np.ndarray) -> np.ndarray:
        """
        :param states: preprocessed states
        :return: a list of actions -> [1, 0, 2, 3, 0, 1, ...]
        """
        states = torch.from_numpy(states).float().to(self.device)
        policy, value = self.model(states)
        softmax_policy = F.softmax(policy, dim=-1)

        actions = softmax_policy.multinomial(1).view(-1).cpu().numpy()  # 가중치에 따라서 action을 선택
        actions += 1
        print(actions)
        # action = np.random.choice(self.n_action, size=1, p=policy)
        return actions

    def predict_transition(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
         - policy: \pi(a_t, s_t)
         - value: V(s_t)
        """
        # Calculate current policy and value
        states_tensor = torch.from_numpy(states).to(self.device).float()
        pred_policies, pred_values = self.model(states_tensor)
        pred_policies = pred_policies.view(-1).data.cpu().numpy()
        pred_values = pred_values.view(-1).data.cpu().numpy()
        return pred_policies, pred_values

    def train(self, actions, states, next_states, discounted_returns, advantages, entropy_coef=0.02):
        with torch.no_grad():  # It makes tensors set "requires_grad" to be false
            state_batch = torch.FloatTensor(states).to(self.device)
            next_state_batch = torch.FloatTensor(next_states).to(self.device)
            target_batch = torch.FloatTensor(discounted_returns).to(self.device)
            advantage_batch = torch.FloatTensor(advantages).to(self.device)
            y_batch = torch.LongTensor(actions).to(self.device)

        # Standardization
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-18)

        pred_policy, pred_value = self.model(state_batch)
        policy_softmax = F.softmax(pred_policy, dim=-1)
        m = Categorical(policy_softmax)

        # Actor loss
        actor_loss = -m.log_prob(y_batch) * advantage_batch

        # Entorpy
        entropy = m.entropy()

        # Critic loss
        critic_loss = self.mse(pred_value.sum(1), target_batch)

        self.optimizer.zero_grad()

        # Loss
        loss = actor_loss.mean() * 0.5 * critic_loss - entropy_coef * entropy.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()


class A2CAgent(Agent):
    pass


class A2CTrain(object):
    """
    Advantage Actor Critic Trainer with N-Step bootstrapping
    """

    def __init__(self, game_id: str, model: nn.Module, input_shape: Tuple[int, int], n_step: int, n_action: int,
                 n_processor: int = 1, render: bool = False, n_history: int = 4, cuda=False,
                 process_reward: Callable = None):
        self.game_id = game_id
        self.model = model
        self.input_shape = input_shape
        self.n_step = n_step
        self.n_action = n_action
        self.n_processor = n_processor
        self.render = render
        self.n_history = n_history
        self.device: str = 'cuda' if cuda else 'cpu'

        # Functions
        self.process_reward = process_reward

        # Initialize Agent
        self.agent: A2CAgent = A2CAgent(model, n_action=n_action, cuda=cuda)
        self.agent.model.load_state_dict(torch.load('checkpoints/mario.model'))

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
        self.dq_infos = deque(maxlen=n_step)

    def _init_envs(self):
        for idx in range(self.n_processor):
            parent_conn, child_conn = Pipe()
            env = self.create_env()
            env_processor = MultiProcessEnv(idx, env, child_conn,
                                            input_size=self.input_shape,
                                            n_history=self.n_history,
                                            render=self.render)
            env_processor.start()
            self.envs.append(env_processor)
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)

    def create_env(self) -> gym.Env:
        env = gym.make(self.game_id)
        return env

    def initialize_states(self) -> np.ndarray:
        """
        :return: [n_envs, 4, 120, 128]
        """
        states = np.zeros([self.n_processor, self.n_history, *self.input_shape])
        return states

    def train(self, gamma: float = 0.99, lambda_: float = 0.95):
        agent = self.agent

        # Prepare Training
        states = self.initialize_states()

        step = -1
        while True:
            self.clean_queues()
            step += 1

            for _ in range(self.n_step):
                # Get Action
                actions = agent.get_action(states)

                # Interact with environments
                self.send_actions(actions)
                next_states, rewards, dones, infos = self.receive_from_envs()
                rewards = self.process_reward(states, actions, next_states, rewards, dones, infos)

                # Store environment data
                self._store_data(states, actions, next_states, rewards, dones, infos)

                # Update states <- next_states
                states = next_states

            # Train Policy and Value Networks
            target_data = self._build_target_data(gamma=gamma, lambda_=lambda_)
            group_states, group_next_states, group_discounted_returns, group_actions, group_advantages = target_data

            self.agent.train(group_actions,
                             group_states,
                             group_next_states,
                             group_discounted_returns,
                             group_advantages)

            if step % 500 == 0:
                print(f'saved | actions:{group_actions}')
                torch.save(agent.model.state_dict(), 'checkpoints/mario.model')

    def _store_data(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                    dones: np.ndarray, infos: list):
        self.dq_states.append(states)
        self.dq_actions.append(actions)
        self.dq_next_states.append(next_states)
        self.dq_rewards.append(rewards)
        self.dq_dones.append(dones)
        self.dq_infos.append(infos)

    def _retrieve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        return group_states, group_next_states, group_rewards, group_actions, group_dones

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

    def _build_target_data(self, gamma: float, lambda_: float):
        # Retrieve grouped environment data : N-Step = Group
        group_states, group_next_states, group_rewards, group_actions, group_dones = self._retrieve_data()

        # predict transitions
        pred_policies, pred_values = self.agent.predict_transition(group_states)
        _, pred_next_values = self.agent.predict_transition(group_next_states)

        # Build Target
        group_discounted_returns = []
        group_advantages = []
        for idx in range(self.n_processor):
            _rewards = group_rewards[idx * self.n_step: (idx + 1) * self.n_step]
            _pred_values = pred_values[idx * self.n_step: (idx + 1) * self.n_step]
            _pred_next_values = pred_next_values[idx * self.n_step: (idx + 1) * self.n_step]
            _dones = group_dones[idx * self.n_step: (idx + 1) * self.n_step]

            discounted_return, advantage = self.calculate_td_lambda(_rewards, _pred_values, _pred_next_values,
                                                                    _dones, gamma=gamma, lambda_=lambda_)

            group_discounted_returns.append(discounted_return)
            group_advantages.append(advantage)

        group_discounted_returns = np.hstack(group_discounted_returns)
        group_advantages = np.hstack(group_advantages)

        return group_states, group_next_states, group_discounted_returns, group_actions, group_advantages

    def calculate_td_lambda(self, rewards: np.ndarray, pred_values: np.ndarray, pred_next_values: np.ndarray,
                            dones: np.ndarray, gamma: float, lambda_: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        TD(\lambda)

        It receives environment data of a single processor
        :param rewards: an array of the n-step rewards r_t from a single processor
        :param pred_values: an array of the n-step V(s_t) from a single processor
        :param pred_next_values: an array of the n-step V(s_{t+1}) from a single processor
        :param dones:
        """
        discounted_return = np.zeros((self.n_step))

        z = 0
        for t in range(self.n_step - 1, -1, -1):
            # 1-step TD: V(s_t) r_t + \gamma V(s_{t+1}) - V(s_t)
            td_error = rewards[t] + gamma * pred_next_values[t] * (1 - dones[t]) - pred_values[t]

            # z_0 = 0
            # z_t = \gammma * \lambda * z_{t-1} + V(s_t)
            z = gamma * lambda_ * (1 - dones[t]) * z + td_error

            discounted_return[t] = z + pred_values[t]
        advantage = discounted_return - pred_values
        return discounted_return, advantage

    def clean_queues(self):
        self.dq_actions.clear()
        self.dq_rewards.clear()
        self.dq_next_states.clear()
        self.dq_states.clear()
        self.dq_dones.clear()
        self.dq_infos.clear()


class A2CMarioTrain(A2CTrain):

    def create_env(self) -> gym.Env:
        env = gym_super_mario_bros.make(self.game_id)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        return env


def process_reward(states: np.ndarray, actions: np.ndarray, next_states: np.ndarray, rewards: np.ndarray,
                   dones: np.ndarray, infos: list) -> np.ndarray:
    return rewards


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    n_action = 2  # 1:앞 | 2:점프
    resized_input_shape = (100, 80)
    n_processor = 8
    n_history = 4
    n_step = 5

    print('n_processor        :', n_processor)
    print('input  shape       :', env.observation_space.shape)
    print('output shape       :', env.action_space.n)
    print('resized input shape:', resized_input_shape)

    # Hyperparameters
    a2c_model = A2CModel(resized_input_shape, n_action, n_history=n_history)
    a2c_train = A2CMarioTrain('SuperMarioBros-v0', a2c_model, n_processor=n_processor, render=True, cuda=True,
                              n_step=n_step, n_action=n_action, input_shape=resized_input_shape,
                              process_reward=process_reward)
    a2c_train.train()


if __name__ == '__main__':
    main()
