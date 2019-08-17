import torch.nn as nn
import os
import time
from copy import deepcopy
from collections import deque
from imageio import mimsave
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from .network import *
from .replay import *
from .utils import *

class OptionCritic:

    def __init__(self, env: gym.Env, policy: nn.Module, log_dir="runs", start_steps=10000,
                 train_after_steps=1, gradient_steps=1, epsilon=1.0, epsilon_decay=1e-4, epsilon_min=0.01, gradient_clip=1, gamma=0.99, minibatch_size=64,
                 buffer_size=10e4, polyak=0.001, max_eps_len=10e4, lr=3e-4, qloss_func=nn.MSELoss()):
        self.env = env
        self.policy = policy
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_steps = start_steps
        self.train_after_steps = train_after_steps
        self.gradient_steps = gradient_steps
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gradient_clip = gradient_clip
        self.gamma = gamma
        self.polyak = polyak
        self.max_eps_len = max_eps_len
        self.lr = lr
        self.qloss_func = qloss_func

        self.state = env.reset()
        self.current_option = None
        self.previous_option = None
        self.replay_buffer = Replay(buffer_size, minibatch_size)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.step_counter = 0

    def _train_step(self):
        if len(self.replay_buffer) < self.start_steps:
            for _ in range(self.start_steps):
                self.agent_step()
            self.agent_reset()
            print(f"Replay buffer initialized with {len(self.replay_buffer)} random steps")

        reward, done = self.agent_step()
        self.step_counter += 1

        if self.step_counter % self.train_after_steps == 0:
            for _ in range(self.gradient_steps):
                self._update_models()
            self.epsilon = max(self.epsilon_min, (1 - self.epsilon_decay)*self.epsilon)

        return reward, done

    def agent_step(self):
        with torch.no_grad():
            action_probs, log_probs, termination_probs, q_u, q_omega = self.policy(self.state)
            if self.current_option is None:
                self.current_option = Categorical(probs=self._epsilon_probs(q_omega[0])).sample()
            action = Categorical(probs=action_probs[0, self.current_option, :]).sample()
            action = action.cpu().detach().numpy()
            # action = self.env.action_space.sample()
            action = int(action)
        next_state, reward, done, info = self.env.step(action)
        # self.env.render()
        self.replay_buffer.add([self.state, action, self.current_option, self.previous_option, reward, next_state, done])
        self.state = next_state
        self.previous_option = self.current_option
        if done:
            self.agent_reset()
        elif termination_probs[0, self.current_option] >= torch.rand(1):
            self.current_option = None
        return reward, done

    def agent_reset(self):
        self.state = self.env.reset()
        self.current_option = None

    def _update_models(self):
        num_options = self.policy.num_options
        states, actions, options, previous_options, rewards, next_states, dones = self.replay_buffer.sample()
        next_action_probs, next_log_probs, next_termination_probs, next_q_u, next_q_omega = self.policy(next_states)
        next_values_max = next_q_omega.max(1)[0]

        action_probs, log_probs, termination_probs, q_u, q_omega = self.policy(states)

        target = tensor(rewards).reshape(-1, 1)
        val_fut = (1 - next_termination_probs.gather(1, torch.tensor(options).view(-1, 1)))\
                  * next_q_omega.gather(1, torch.tensor(options).view(-1, 1))\
                  + next_termination_probs.gather(1, torch.tensor(options).view(-1, 1)) * next_values_max.view(-1, 1)
        target += (1 - tensor(dones).reshape(-1, 1)) * self.gamma * val_fut

        mask = torch.zeros_like(q_u).byte()
        for i, j, k in zip(range(mask.size()[0]), options, actions):
            mask[i, j, k] = True
        q_loss = torch.pow(q_u.masked_select(mask) - target.squeeze(-1).detach(), 2)

        policy_loss = log_probs.masked_select(mask) * q_u.masked_select(mask).detach()

        termination_mask = tensor(options == previous_options).float()
        termination_mask.requires_grad = False
        adv = next_q_omega.gather(1, torch.tensor(options).view(-1, 1)) - next_values_max.view(-1, 1)
        termination_loss = termination_probs.gather(1, torch.tensor(options).view(-1, 1)) * adv.detach()
        termination_loss = termination_loss.view(-1) * termination_mask
        # print(q_loss.size(), termination_loss.size(), policy_loss.size())
        total_loss = (q_loss + termination_loss + policy_loss).mean()

        self.policy_opt.zero_grad()
        total_loss.backward()
        self.policy_opt.step()

    def learn(self, iterations=1e5):
        eps_reward = deque(maxlen=100)
        total_eps = 0
        # eps_reward.append(0)
        running_reward = 0
        start_time = time.time()
        for i in range(int(iterations)):
            reward, done = self._train_step()
            running_reward += reward
            if done:
                total_eps += 1
                eps_reward.append(running_reward)
                running_reward = 0
                self.writer.add_scalar("train/mean_reward", np.mean(eps_reward), global_step=self.step_counter)
                self.writer.add_scalar("train/reward", eps_reward[-1], global_step=self.step_counter)

            if i % (iterations // 1000) == 0:
                fps = (iterations // 1000) / (time.time() - start_time)
                start_time = time.time()
                str_info = "Steps: {:8d}\tFPS: {:4.2f}\tLastest Episode reward: {:4.2f}\tMean Rewards: {:4.4f}\tTotal Episodes: {:5d}"
                print(str_info.format(i, fps,eps_reward[-1] if len(eps_reward)>0 else 0,np.mean(eps_reward), total_eps),
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
        # torch.save(self.critic.cpu().state_dict(), os.path.join(dirpath, 'critic.pt'))

    def load_model(self, dirpath='.'):
        if not os.path.exists(dirpath):
            raise Exception("Path does not exist")
        print(f"Loading models from directory {dirpath}")
        self.policy.load_state_dict(torch.load(os.path.join(dirpath, 'policy.pt')))
        # self.critic.load_state_dict(torch.load(os.path.join(dirpath, 'critic.pt')))

    def _epsilon_probs(self, vals: torch.Tensor):
        probs = torch.zeros_like(vals)
        probs[vals.argmax(-1)] += (1 - self.epsilon)
        probs += self.epsilon/probs.size()[-1]
        return probs



