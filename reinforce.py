import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, state_dims, n_actions, hidden=128):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dims, hidden)
        self.affine2 = nn.Linear(hidden, n_actions)

        self.saved_log_probs = []
        self.rewards = []

#        self.inp_bias = torch.FloatTensor([9,
#        self.inp_var


    def forward(self, x):
        diap = torch.FloatTensor([9, 14, 5, 5])
        inp_bias = torch.FloatTensor([9, 14, 5, 5]) / 2
        inp_var = diap
        x = (x - inp_bias) / inp_var
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()


class Trainer:
    def __init__(self, env, policy, writer=None, lr=1e-3, max_len=1000, gamma=0.95):
        self.env = env
        self.policy = policy
        self.max_len = max_len
        self.gamma = gamma

        self.writer = writer
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.running_reward = None
        self.iter = 0

    def run_episode(self):
        self.env.reset()

        for t in range(self.max_len):  # Don't infinite loop while learning
            action = self.policy.select_action(self.env.get_state())
            res, state, reward = self.env.act(action)
            self.policy.rewards.append(reward)
            if res == 'FINISH':
                break

        rew = np.sum(self.policy.rewards)
        if self.running_reward is None:
            self.running_reward = rew
        else:
            self.running_reward = self.running_reward * 0.99 + rew * 0.01

        self.reinforce()

        return rew

    def reinforce(self):
        R = 0
        policy_loss = []
        gains = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            gains.insert(0, R)
        gains = torch.tensor(gains)
        gains = (gains - gains.mean()) / (gains.std() + 1e-6)

        for log_prob, gain in zip(self.policy.saved_log_probs, gains):
            policy_loss.append(-log_prob * gain)
        self.opt.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        if self.writer is not None:
            self.writer.add_scalar('loss', policy_loss, self.iter)
            self.writer.add_scalar('gain0', R, self.iter)
            self.writer.add_scalar('episode_len', len(self.policy.rewards), self.iter)

            for n, p in self.policy.named_parameters():
                self.writer.add_scalar(n + '_grad', p.grad.norm(), self.iter)
                self.writer.add_scalar(n, p.norm(), self.iter)

        self.opt.step()
        self.policy.rewards.clear()
        self.policy.saved_log_probs.clear()
        self.iter += 1
