import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, env, hidden=128):
        super().__init__()
        szs = env.get_sizes()
        state_dims = len(szs)
        n_actions = env.N_ACTIONS

        self.affine1 = nn.Linear(state_dims, hidden)
        self.affine2 = nn.Linear(hidden, n_actions)

        self.inp_diap = torch.FloatTensor(szs)


    def forward(self, x):
        x = (x - self.inp_diap / 2) / self.inp_diap
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Values(nn.Module):
    def __init__(self, env, hidden=128):
        super().__init__()
        szs = env.get_sizes()
        n_states = len(szs)

        self.inp_diap = torch.FloatTensor(szs)

        self.affine1 = nn.Linear(n_states, hidden)
        self.affine2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = (x - self.inp_diap / 2) / self.inp_diap

        x = F.relu(self.affine1(x))
        y = self.affine2(x)
        return y


class Reinforce:
    def __init__(self, env, policy, value=None, writer=None, lr=1e-3, max_len=1000, gamma=0.95):
        self.env = env
        self.policy = policy
        self.value = value
        self.max_len = max_len
        self.gamma = gamma

        self.writer = writer
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(value.parameters(), lr=lr)
        self.running_reward = None
        self.iter = 0

        self.saved_log_probs = []
        self.rewards = []
        self.states = []

    def run_episode(self):
        self.env.reset()
        for t in range(self.max_len):  # Don't infinite loop while learning
            st = self.env.get_state()
            action, log_prob = self.policy.select_action(st)
            res, state, reward = self.env.act(action)
            self.rewards.append(reward)
            self.saved_log_probs.append(log_prob)
            self.states.append(st)
            if res == 'FINISH':
                break

        rew = np.sum(self.rewards)
        if self.running_reward is None:
            self.running_reward = rew
        else:
            self.running_reward = self.running_reward * 0.99 + rew * 0.01

        self.reinforce()

        return rew

    def reinforce(self):
        G = 0
        policy_loss = []
        p_deltas = []
        gains = []

        rvalues = self.value.forward(torch.FloatTensor(self.states[::-1]))

        for r, v in zip(self.rewards[::-1], rvalues):
            G = r + self.gamma * G
            p_deltas.append(G-v)
            gains.append(G)
        p_deltas = torch.tensor(p_deltas[::-1])
        #gains = (gains - gains.mean()) / (gains.std() + 1e-6)

        for log_prob, delta in zip(self.saved_log_probs, p_deltas):
            policy_loss.append(-log_prob * delta)

        self.opt.zero_grad()
        self.value_opt.zero_grad()

        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()

        value_loss = (torch.tensor(gains) - rvalues)**2
        value_loss = value_loss.sum()
        value_loss.backward()

        self.opt.step()
        self.value_opt.step()

        if self.writer is not None:
            self.writer.add_scalar('loss', policy_loss, self.iter)
            self.writer.add_scalar('gain0', G, self.iter)
            self.writer.add_scalar('episode_len', len(self.rewards), self.iter)

            self.writer.add_scalar('value_loss', value_loss, self.iter)

            for n, p in self.policy.named_parameters():
                self.writer.add_scalar(n + '_grad', p.grad.norm(), self.iter)
                self.writer.add_scalar(n, p.norm(), self.iter)

        self.rewards.clear()
        self.saved_log_probs.clear()
        self.states.clear()
        self.iter += 1


if __name__ == '__main__':
    import race as R
    rf_race = R.RaceTrack(R.track3,2)

    policy = Policy(state_dims=len(rf_race.get_sizes()), n_actions=rf_race.N_ACTIONS, hidden=200)
    value = Values(rf_race)
    trainer = Trainer(rf_race, policy, value=value, lr=1e-3, max_len=200, gamma=0.99)
    for k in range(1000):
        trainer.run_episode()
