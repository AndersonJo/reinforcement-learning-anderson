import argparse
import os
from argparse import ArgumentParser
from typing import List

import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
from gym import Wrapper
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from torch.multiprocessing import Process, get_context

# Create Context for sharing data among processors
ctx = get_context('spawn')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('mode', type=str, help='train | test')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/mario.model')
    parser.add_argument('--device', type=str, default='cuda', help='cuda | cpu')
    parser.add_argument('--history', type=int, default=4)
    parser.add_argument('--processor', type=int, default=2)
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
        # output_size = self._calculate_output_size(input_shape, n_history)
        self.lstm = nn.LSTMCell(3456, 512)
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
                # state['step'] = 0
                # state['exp_avg'] = torch.zeros_like(p.data)
                # state['exp_avg_sq'] = torch.zeros_like(p.data)
                #
                # state['exp_avg'].share_memory_()
                # state['exp_avg_sq'].share_memory_()


class RewardEnv(Wrapper):
    def __init__(self, env: gym.Env):
        super(RewardEnv, self).__init__(env)

        self.cur_score = 0

    def step(self, action):
        pass


class SuperMarioEnv(RewardEnv):
    pass


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

        # Local
        self.local_model: nn.Module = None
        self.env: SuperMarioEnv = None

        # Options
        self.opt = opt
        self.n_processor: int = opt.processor
        self.n_action: int = opt.n_action
        self.n_history: int = opt.history
        self.resized_input_shape: tuple = opt.resized_input_shape
        self.device: str = opt.device
        self.training = training

        self.processors: List[Process] = list()

    def _initialize_processors(self, n_processor: int, to_train: bool):
        for idx in range(n_processor):
            args = (self.global_model, self.global_optimizer, self.opt, idx + 1, to_train)
            p = SuperMarioA3C(*args)
            p.start()
            self.processors.append(p)

    def _initialize_local(self):
        # Create local environment
        self.env = self.create_env()

        # Create local model
        args = (self.resized_input_shape,
                self.n_action,
                self.n_history)
        self.local_model = ActorCriticModel(*args)
        self.local_model.cuda()
        print(self.device)

    @staticmethod
    def create_env() -> gym.Env:
        world = 1
        stage = np.random.randint(1, 4)

        env_id = f'SuperMarioBros-{world}-{stage}-v0'
        env = JoypadSpace(gym_super_mario_bros.make(env_id), COMPLEX_MOVEMENT)
        env = SuperMarioEnv(env)
        return env

    def train(self):
        self._initialize_processors(self.n_processor, True)

    def test(self):
        self._initialize_processors(1, False)

    def wait(self):
        for p in self.processors:
            p.join()

    def run(self):
        torch.manual_seed(self.idx + 100)
        self._initialize_local()


def main():
    opt = parse_args()

    env_id = 'SuperMarioBros-v0'
    env = JoypadSpace(gym_super_mario_bros.make(env_id), SIMPLE_MOVEMENT)

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
