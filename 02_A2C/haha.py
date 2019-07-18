from copy import deepcopy

import gym
import readchar as readchar
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = gym.make('BreakoutDeterministic-v4')

print(env.env.ale.lives())
print(type(env))

done = True
for step in range(5000):
    if done:
        state = env.reset()
    # action = env.action_space.sample()
    env.render()
    action = int(readchar.readchar())

    state, reward, done, info = env.step(action)
    print(reward, done, info)
    env.render()

env.close()
