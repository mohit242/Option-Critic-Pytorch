import torch
import numpy as np
import gym
from functools import reduce


def soft_update(target, source, polyak=0.001):
    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(polyak * local_param.data + (1.0 - polyak) * target_param.data)


def hard_update(target, source):
    for target_param, local_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(local_param.data)


class ActionSACWrapper(gym.ActionWrapper):

    def __init__(self, env):
        """Changes action space range from [action_space.low, action_space.high] to [-1,1]"""
        self.env = env
        super().__init__(env)
        self.action_space = env.action_space
        self.action_space.low = -np.ones_like(env.action_space.low)
        self.action_space.high = np.ones_like(env.action_space.high)

    def action(self, action):
        action = (action * (self.env.action_space.high - self.env.action_space.low)/2) +\
                 (self.env.action_space.high + self.env.action_space.low)/2
        return action


class DiscreteToBox(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        """Converts discrete observation to one-hot box observation"""
        super().__init__(env)
        self.observation_space = gym.spaces.Box(0, 1, (env.observation_space.n, ))
        self._observation_map = np.eye(env.observation_space.n)

    def observation(self, observation):
        obs = tuple(self._observation_map[observation])
        return obs


class FlattenArrayObservation(gym.ObservationWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        image_space = env.observation_space.spaces['image']
        img_len = reduce(lambda x, y: x*y, image_space.shape)
        self.observation_space = gym.spaces.Box(0, 1, (img_len, ))

    def observation(self, observation):
        obs = observation['image'].flatten().astype(np.float)
        obs /= 255
        return obs


