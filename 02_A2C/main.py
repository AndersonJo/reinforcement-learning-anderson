import gym
import gym_super_mario_bros
import torch.nn as nn
import torch.nn.functional as F
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class Flatten(nn.Module):
    def forward(self, input):
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
        self.actor = nn.Linear(in_features=512,
                               out_features=output_size)
        self.critic = nn.Linear(in_features=512,
                                out_features=1)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            elif isinstance(p, nn.Linear):
                nn.init.kaiming_uniform_(p.weight, a=1.)
                p.bias.data.zero_()

    def forward(self, state):
        h = self.feature(state)
        policy = self.actor(h)
        value = self.critic(h)
        return policy, value


def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Hyperparameters
    output_size = env.observation_space

    A2CModel(output_size)

    import ipdb
    ipdb.set_trace()


if __name__ == '__main__':
    main()
