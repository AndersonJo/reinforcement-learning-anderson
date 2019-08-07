import argparse
import os
import subprocess as sp
from argparse import ArgumentParser
from collections import deque
from time import sleep
from typing import List, Tuple

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Wrapper
from gym.spaces import Box
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.distributions import Categorical
from torch.multiprocessing import Process, get_context

# Create Context for sharing data among processors
ctx = get_context('spawn')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, help='train | test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/super_mario.model')
    parser.add_argument('--device', type=str, default='cuda', help='cuda | cpu')
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--n-step', type=int, default=64)
    parser.add_argument('--processor', type=int, default=8)

    parser.add_argument('--gamma', type=float, default=0.9, help='future discounted value')
    parser.add_argument('--lambda_', type=float, default=1, help='GAE hyper parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--lr', type=float, default=0.0001)

    args = parser.parse_args()

    assert isinstance(args.history, int)
    assert isinstance(args.processor, int)
    assert isinstance(args.lr, float)
    assert args.device in ['cpu', 'cuda']
    assert args.checkpoint.endswith('.model')
    assert args.processor >= 1

    args.mode = args.mode.lower()
    return args


def load_model(model: nn.Module, checkpoint_path: str):
    checkpoint_dir = os.path.dirname(checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if os.path.exists(checkpoint_path):
        print(f'Loaded checkpoint - {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path))


class ActorCriticModel(nn.Module):
    def __init__(self, input_shape: tuple, n_action: int, n_history: int):
        super(ActorCriticModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=n_history,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU())
        # output_size = self._calculate_output_size(input_shape, n_history)

        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)
        # self.lstm = nn.LSTMCell(3456, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, n_action)
        self._initialize_weights()

    def _calculate_output_size(self, input_shape: tuple, n_history: int) -> int:
        image = torch.zeros(1, n_history, *input_shape, dtype=torch.float)
        o = self.conv(image)
        output_size = int(np.prod(o.size()))
        return output_size

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, h_state: torch.Tensor, c_state: torch.Tensor) -> List[torch.Tensor]:
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        h_tensor, c_tensor = self.lstm(h, (h_state, c_state))
        logits = self.actor_linear(h_tensor)
        value = self.critic_linear(h_tensor)
        return [logits, value, h_tensor, c_tensor]


class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class RewardEnv(Wrapper):
    def __init__(self, env: gym.Env):
        super(RewardEnv, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.cur_score = 0

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        state = self.preprocess_state(state)

        reward += (info["score"] - self.cur_score) / 30.
        self.cur_score = info['score']
        if done:
            if info['flag_get']:
                reward += 64
            else:
                reward -= 64
        return state, reward / 15, done, info

    def reset(self):
        self.cur_score = 0
        state = self.preprocess_state(self.env.reset())
        return state

    @staticmethod
    def preprocess_state(state: np.ndarray):
        h = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        h = cv2.resize(h, (84, 84))[None, :, :]
        h = np.float32(h) / 255.
        return h


class SkipFrameEnv(Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super(SkipFrameEnv, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip

    def step(self, action: int):
        total_reward = 0

        state, reward, done, info = self.env.step(action)
        states = []
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            states.append(state)

        states = np.array(states).transpose([1, 0, 2, 3])

        return states, reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.array([state for _ in range(self.skip)], dtype=np.float32)
        states = states.transpose([1, 0, 2, 3])
        return states.astype(np.float32)


class SuperMarioA3C(ctx.Process):
    def __init__(self,
                 global_model: nn.Module,
                 global_optimizer: GlobalAdam,
                 opt: argparse.Namespace,
                 idx: int = 0,
                 training: bool = False):
        super(SuperMarioA3C, self).__init__()

        # Global
        self.idx = idx
        self.global_model = global_model
        self.global_optimizer = global_optimizer

        # Realtime Variables
        self.cur_step = 0
        self.cur_episode = 0

        # Options
        self.opt = opt
        self.n_processor: int = opt.processor
        self.n_action: int = opt.n_action
        self.n_history: int = opt.history
        self.n_step: int = opt.n_step
        self.resized_input_shape: tuple = opt.resized_input_shape
        self.device: str = opt.device
        self.training = training
        self.checkpoint_path = opt.checkpoint

        # Model Hyper Parameters
        self.gamma: float = opt.gamma
        self.lambda_: float = opt.lambda_
        self.entropy_coef: float = opt.entropy_coef

        self.processors: List[Process] = list()

        # Local
        self.local_model: nn.Module = None
        self.env: SkipFrameEnv = None

        self.states: List[np.ndarray] = deque(maxlen=self.n_step)
        self.next_states: List[np.ndarray] = deque(maxlen=self.n_step)
        self.next_h: List[torch.Tensor] = deque(maxlen=self.n_step)
        self.next_c: List[torch.Tensor] = deque(maxlen=self.n_step)
        self.dones: List[bool] = deque(maxlen=self.n_step)
        self.values: List[torch.Tensor] = deque(maxlen=self.n_step)
        self.log_policies: List[torch.Tensor] = deque(maxlen=self.n_step)
        self.rewards: List[float] = deque(maxlen=self.n_step)
        self.entropies: List[torch.Tensor] = deque(maxlen=self.n_step)
        self.moves = deque(maxlen=500)
        self.flag_gets = deque(maxlen=10)
        self.flag_gets.append(0)

    def _initialize_processors(self, n_processor: int, to_train: bool):
        for idx in range(n_processor):
            args = (self.global_model, self.global_optimizer, self.opt, idx + 1, to_train)
            p = SuperMarioA3C(*args)
            p.start()
            self.processors.append(p)

    def initialize_local(self):
        # Create local environment
        self.env = self.create_env()

        # Create local model
        args = (self.resized_input_shape,
                self.n_action,
                self.n_history)
        self.local_model = ActorCriticModel(*args)
        self.local_model.to(self.device)
        self.local_model.load_state_dict(self.global_model.state_dict())
        if self.training:
            # Dropout, Batchnorm 같이 training 할때 행동이 달라지는 레이어에 학습중이라는 것을 알린다
            # evaluation을 할때는 model.train(mode=False) 또는 model.eval()을 실행한다
            self.local_model.train()

    def initialize_game(self) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        self.env = self.create_env()

        h_s = torch.zeros((1, 512), dtype=torch.float).to(self.device)
        c_s = torch.zeros((1, 512), dtype=torch.float).to(self.device)

        state = self.env.reset()
        self.clear_deque()

        self.cur_episode += 1
        return state, h_s, c_s

    def get_value(self, state: np.ndarray,
                  h_s: torch.Tensor,
                  c_s: torch.Tensor) -> List[torch.Tensor]:
        state_tensor = torch.from_numpy(state).to(self.device)
        logits, value, next_h, next_c = self.local_model(state_tensor, h_s, c_s)
        return [logits, value, next_h, next_c]

    def get_action(self,
                   state: np.ndarray,
                   h_s: torch.Tensor,
                   c_s: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor,
                                               torch.Tensor, torch.Tensor, torch.Tensor]:

        logits, value, next_h, next_c = self.get_value(state, h_s, c_s)
        policy = F.softmax(logits, dim=-1)
        action = int(policy.multinomial(1).cpu().numpy().flatten()[0])

        # m = Categorical(policy)
        # action = m.sample().item()

        return action, logits, policy, value, next_h, next_c

    def create_env(self) -> gym.Env:
        world = np.random.randint(1, 9)
        stage = np.random.randint(1, 5)

        env_id = f'SuperMarioBros-{world}-{stage}-v0'
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
        env = RewardEnv(env)
        env = SkipFrameEnv(env, self.n_history)
        return env

    def train(self):
        self._initialize_processors(self.n_processor, True)

    def _train(self):
        torch.manual_seed(self.idx + 100)
        self.initialize_local()

        state, h_s, c_s = self.initialize_game()
        done = False

        while True:
            # Transfer global model to local model
            self.local_model.load_state_dict(self.global_model.state_dict())

            if done:
                state, h_s, c_s = self.initialize_game()

            # Play N-Step
            done, h_s, c_s = self._play_n_step(state, h_s, c_s)

            # Train
            self._update()

            # Save
            if done:
                self.save()

    def test(self):
        self._initialize_processors(1, False)

    def _test(self):
        torch.manual_seed(self.idx + 100)
        self.initialize_local()

        state, h_s, c_s = self.initialize_game()
        done = True
        while True:

            if done:
                self.clear_deque()
                self.local_model.load_state_dict(self.global_model.state_dict())
                state, h_s, c_s = self.initialize_game()

            state_tensor = torch.from_numpy(state).to(self.device)
            logits, value, h_s, c_s = self.local_model(state_tensor, h_s, c_s)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()

            state, reward, done, info = self.env.step(action)
            self.moves.append(info['x_pos'])
            self.env.render()

            _move_var = int(np.var(self.moves))
            if len(self.moves) == self.moves.maxlen and _move_var < 10:
                done = True

            sleep(0.012)

    def save(self):
        if self.cur_episode != 0 and self.cur_episode % 100 == 0:
            print(f'Global Model Saved! - {self.checkpoint_path}')
            torch.save(self.global_model.state_dict(), self.checkpoint_path)

    def wait(self):
        for p in self.processors:
            p.join()

    def run(self):
        if self.training:
            self._train()
        else:
            with torch.no_grad():
                self._test()

    def _play_n_step(self,
                     state: np.ndarray,
                     h_s: torch.Tensor,
                     c_s: torch.Tensor) -> Tuple[bool, torch.Tensor, torch.Tensor]:

        self.clear_deque()
        done = False
        for _ in range(self.n_step):

            # action, logits, policy, value, h_s, c_s = self.get_action(state, h_s, c_s)
            # log_policy = F.log_softmax(logits, dim=-1)  # [[-1.9515, -1.9586, ..., -1.9567]]
            # entropy = -(policy * log_policy).sum(-1, keepdim=True)  # [[1.9454]]

            state_tensor = torch.from_numpy(state).to(self.device)
            logits, value, h_s, c_s = self.local_model(state_tensor, h_s, c_s)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            m = Categorical(policy)
            action = m.sample().item()

            # Step!
            next_state, reward, done, info = self.env.step(action)

            self.states.append(state)
            self.next_states.append(next_state)
            self.next_h.append(h_s)
            self.next_c.append(c_s)
            self.dones.append(done)
            self.values.append(value)
            self.log_policies.append(log_policy[0, action])
            self.rewards.append(reward)
            self.entropies.append(entropy)
            self.moves.append(info['x_pos'])
            state = next_state

            # Increase step
            self.cur_step += 1

            if done:
                self.flag_gets.append(int(info['flag_get']))
                break

        h_s = h_s.detach()
        c_s = c_s.detach()
        return done, h_s, c_s

    def _update(self):
        # Calculate last value
        total_reward = torch.zeros((1, 1), dtype=torch.float)
        if not self.dones[-1]:
            last_state = self.next_states[-1]
            last_h = self.next_h[-1]
            last_c = self.next_c[-1]
            _, total_reward, _, _ = self.get_value(last_state, last_h, last_c)

        total_reward = next_value = total_reward.to(self.device)

        # Generalized Advantage Estimation (GAE)
        gae = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        actor_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        critic_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)
        entropy_loss = torch.zeros((1, 1), dtype=torch.float).to(self.device)

        assert len(self.values) == len(self.log_policies)
        assert len(self.log_policies) == len(self.rewards)
        assert len(self.rewards) == len(self.entropies)
        if self.dones[-1]:
            assert total_reward == 0
            assert next_value == 0

        n_step = len(self.values)
        for i in range(n_step - 1, -1, -1):
            done = self.dones[i]
            value: torch.Tensor = self.values[i]
            log_policy: torch.Tensor = self.log_policies[i]
            reward: float = self.rewards[i]
            entropy: torch.Tensor = self.entropies[i]

            gae = gae * self.gamma * self.lambda_
            gae = gae + reward + self.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            total_reward = total_reward * self.gamma + reward
            critic_loss = critic_loss + (total_reward - value) ** 2 / 2.
            entropy_loss = entropy_loss + entropy

            if done:
                break

        total_loss = -actor_loss + critic_loss - self.entropy_coef * entropy_loss
        self.global_optimizer.zero_grad()
        total_loss.backward()
        self.update_global_model()
        self.global_optimizer.step()

        # Logging
        _episode = self.cur_episode
        _step = self.cur_step
        _mean_reward = np.mean(self.rewards)
        _var_reward = round(np.var(self.rewards), 1)
        _move_var = int(np.var(self.moves))
        _total_reward = total_reward.data.cpu().view(-1).numpy()[0]
        _total_loss = total_loss.data.cpu().view(-1).numpy()[0]
        _critic_loss = critic_loss.data.cpu().view(-1).numpy()[0]
        _actor_loss = actor_loss.data.cpu().view(-1).numpy()[0]
        _entropy = entropy.data.cpu().view(-1).numpy()[0]
        _done = np.sum(self.dones)
        _flag_mean = np.mean(self.flag_gets)

        print(f'[{self.idx:2}|{_episode:4}] step:{_step:<3} | reward:{round(_total_reward, 1):4.1f} | '
              f'mean:{_mean_reward:5.2f} | var:{_var_reward:5.1f} | '
              f'loss:{_total_loss:5.1f} | '
              f'entropy:{_entropy:4.2f} | '
              f'actor:{_actor_loss:6.1f} | critic:{_critic_loss:6.1f} | '
              f'move:{_move_var:5} | '
              f'flag:{_flag_mean:3.1f}')

        # Clear Deques
        self.clear_deque()

    def update_global_model(self):
        local_params = self.local_model.parameters()
        global_params = self.global_model.parameters()
        for local_param, global_param in zip(local_params, global_params):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

    def clear_deque(self):
        self.states.clear()
        self.next_states.clear()
        self.next_h.clear()
        self.next_c.clear()
        self.dones.clear()
        self.values.clear()
        self.log_policies.clear()
        self.rewards.clear()
        self.entropies.clear()
        self.moves.clear()


def main():
    opt = parse_args()

    env_id = 'SuperMarioBros-v0'
    env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
    env = RewardEnv(env)
    env = SkipFrameEnv(env)

    # Hyper-parameters
    opt.n_action = env.action_space.n
    opt.resized_input_shape = (84, 84)

    # Create Actor Critic Model
    global_model = ActorCriticModel(opt.resized_input_shape, opt.n_action, opt.history)
    global_model.to(opt.device)
    global_model.share_memory()  # 다른 프로세서에서 복사하지 않고 데이터를 공유할 수 있도록 해준다 AWESOME!
    load_model(global_model, opt.checkpoint)

    # Initialize Global Optimizer
    global_optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)

    # Super Mario A3C
    supermario_a3c = SuperMarioA3C(global_model, global_optimizer, opt=opt)

    # Training & Test
    if opt.mode == 'train':
        supermario_a3c.train()
    supermario_a3c.test()
    supermario_a3c.wait()


if __name__ == '__main__':
    main()
