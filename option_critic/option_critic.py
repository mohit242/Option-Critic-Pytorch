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

class OptionCritic(nn.Module):

    def __init__(self, env, policy, writer, start_steps=10000, train_after_steps=1, gradient_steps=1,
                 gradient_clip=1, gamma=0.99, minibatch_size=256, buffer_size=10e5, polyak=0.001,
                 max_eps_len=10e4, lr=3e-4, loss_func=nn.MSELoss()):
        pass

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
        pass

    def load_model(self, dirpath='.'):
        pass


