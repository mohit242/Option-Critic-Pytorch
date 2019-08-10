import torch.nn as nn
import os
import time
from copy import deepcopy
from collections import deque
from imageio import mimsave
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from .network import *
from .replay import *
from .utils import *

class OptionCritic:

    def __init__(self, env: gym.Env, policy: nn.Module, critic: nn.Module, writer: SummaryWriter, start_steps=10000,
                 train_after_steps=1, gradient_steps=1, epsilon=0.9, gradient_clip=1, gamma=0.99, minibatch_size=256,
                 buffer_size=10e5, polyak=0.001, max_eps_len=10e4, lr=3e-4, qloss_func=nn.MSELoss()):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.writer = writer
        self.start_steps = start_steps
        self.train_after_steps = train_after_steps
        self.gradient_steps = gradient_steps
        self.epsilon = epsilon
        self.gradient_clip = gradient_clip
        self.gamma = gamma
        self.polyak = polyak
        self.max_eps_len = max_eps_len
        self.lr = lr
        self.qloss_func = qloss_func

        self.replay = Replay(buffer_size, minibatch_size)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.step_counter = 0

    def _train_step(self):
        pass

    def _update_models(self):
        pass

    def learn(self, iterations=1e5):
        pass

    def _eval_step(self):
        pass

    def eval(self, gif_path=None):
        pass

    def save_model(self, dirpath='.'):
        if not os.path.exists(dirpath):
            raise Exception("Path does not exist")
        print(f"Saving models in directory {dirpath}")
        torch.save(self.policy.cpu().state_dict(), os.path.join(dirpath, 'policy.pt'))
        torch.save(self.critic.cpu().state_dict(), os.path.join(dirpath, 'critic.pt'))

    def load_model(self, dirpath='.'):
        if not os.path.exists(dirpath):
            raise Exception("Path does not exist")
        print(f"Loading models from directory {dirpath}")
        self.policy.load_state_dict(torch.load(os.path.join(dirpath, 'policy.pt')))
        self.critic.load_state_dict(torch.load(os.path.join(dirpath, 'critic.pt')))

    def _epsilon_probs(self, vals: torch.Tensor):
        probs = torch.zeros_like(vals)
        probs[torch.arange(probs.size()[0]), vals.argmax(-1)] += (1 - self.epsilon)
        probs += self.epsilon/probs.size()[-1]
        return probs



