import torch.nn as nn
import os
import time
from copy import deepcopy
from collections import deque
from imageio import mimsave
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from .network import *
from .replay import *
from .utils import *

class OptionCritic:

    def __init__(self, env: gym.Env, policy: nn.Module, critic: nn.Module, log_dir="runs", start_steps=10000,
                 train_after_steps=1, gradient_steps=1, epsilon=0.9, gradient_clip=1, gamma=0.99, minibatch_size=256,
                 buffer_size=10e5, polyak=0.001, max_eps_len=10e4, lr=3e-4, qloss_func=nn.MSELoss()):
        self.env = env
        self.policy = policy
        self.critic = critic
        self.writer = SummaryWriter(log_dir=log_dir)
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

        self.state = env.reset()
        self.replay_buffer = Replay(buffer_size, minibatch_size)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        self.step_counter = 0

    def _train_step(self):
        if len(self.replay_buffer) < self.start_steps:
            for _ in range(self.start_steps):
                action_probs, log_probs, termination_probs = self.policy(self.state)
                option = np.random.randint(self.policy.num_options)
                action = Categorical(probs=action_probs[option, :]).sample()
                action = action.cpu().detach().numpy()
                print(action_probs)
                next_state, reward, done, info = self.env.step(action)
                if done:
                    self.state = self.env.reset()
                self.replay_buffer.add([self.state, option, reward, next_state, done])
                self.state = next_state
            print(f"Replay buffer initialized with {len(self.replay_buffer)} random steps")

            return 0, 0

    def _update_models(self):
        pass

    def learn(self, iterations=1e5):
        eps_reward = deque(maxlen=100)
        eps_reward.append(0)
        running_reward = 0
        start_time = time.time()
        for i in range(int(iterations)):
            reward, done = self._train_step()
            running_reward += reward
            if done:
                eps_reward.append(running_reward)
                running_reward = 0
                self.writer.add_scalar("train/mean_reward", np.mean(eps_reward), global_step=self.step_counter)
                self.writer.add_scalar("train/reward", eps_reward[-1], global_step=self.step_counter)

            if i % (iterations // 1000) == 0:
                fps = (iterations // 1000) / (time.time() - start_time)
                start_time = time.time()
                print("Steps: {:8d}\tFPS: {:4f}\tLastest Episode reward: {:4f}\tMean Rewards: {:4f}".format(i, fps,
                                                                                                            eps_reward[
                                                                                                                -1],
                                                                                                            np.mean(
                                                                                                                eps_reward)),
                      end='\r')
        print("\n")

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



