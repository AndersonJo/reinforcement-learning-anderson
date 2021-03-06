from copy import deepcopy

import gym
import readchar as readchar
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT

# for world in range(8, 0, -1):
#     for stage in range(1, 5):
#         game = f'SuperMarioBros-{world}-{stage}-v0'
#         print(game, end=' ')
#         env = gym_super_mario_bros.make(game)
#         env.reset()
#         env.render()
#         print(readchar.readchar())

env = gym_super_mario_bros.make('SuperMarioBros-5-4-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# env = gym.make('BreakoutDeterministic-v4')

state = env.reset()

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
