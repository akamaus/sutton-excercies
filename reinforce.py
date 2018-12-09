import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
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


    def forward(self, x, t):
        x = (x - self.inp_diap / 2) / self.inp_diap
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores / t, dim=1)

    def select_action(self, state, t=1):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state, t=t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), torch.sum(-probs * probs.log())


class SpatialPolicy(nn.Module):
    """ Policy with separate input neuron for each spatial state and axe velocity """
    def __init__(self, env, hidden=128):
        super().__init__()
        szs = env.get_sizes()
        self.szs = szs
        self.inp_size = szs[0] + szs[1] + szs[2] + szs[3]
        n_actions = env.N_ACTIONS

        self.affine1 = nn.Linear(self.inp_size, hidden)
        self.affine2 = nn.Linear(hidden, n_actions)

    def forward(self, x, t):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores / t, dim=1)

    def select_action(self, state, t=1):
        assert isinstance(state, tuple)
        x = torch.zeros([self.inp_size]).float()
        p0 = 0
        x[p0 + state[0]] = 1
        p0 += self.szs[0]

        x[p0 + state[1]] = 1
        p0 += self.szs[1]

        x[p0 + state[2]] = 1
        p0 += self.szs[2]

        x[p0 + state[3]] = 1

        assert len(state) == 4

        probs = self.forward(x.unsqueeze(0), t=t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)[0], torch.sum(-probs * probs.log())


class SpatialValues(nn.Module):
    def __init__(self, env, hidden=128):
        super().__init__()
        szs = env.get_sizes()
        self.szs = szs
        self.inp_size = szs[0] + szs[1] + szs[2] + szs[3]

        self.affine1 = nn.Linear(self.inp_size, hidden)
        self.affine2 = nn.Linear(hidden, 1)

    def forward(self, states):
        if isinstance(states, tuple):
            states = [states]
            sole = True
        else:
            sole = False

        x = torch.zeros([len(states), self.inp_size]).float()

        for i, state in enumerate(states):
            p0 = 0
            x[i, p0 + state[0]] = 1
            p0 += self.szs[0]

            x[i, p0 + state[1]] = 1
            p0 += self.szs[1]

            x[i, p0 + state[2]] = 1
            p0 += self.szs[2]

            x[i, p0 + state[3]] = 1

        x = F.relu(self.affine1(x))
        y = self.affine2(x)[:, 0]
        if sole:
            y = y[0]
        return y


class Reinforce:
    def __init__(self, env, policy, value=None, writer=None, lr=1e-3, max_len=1000, gamma=0.95, temperature=1):
        self.env = env
        self.policy = policy
        self.value = value
        self.max_len = max_len
        self.gamma = gamma
        self.temperature = 1

        self.writer = writer
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(value.parameters(), lr=lr)
        self.running_reward = None
        self.iter = 0

        self.table_value = torch.zeros(env.get_sizes()[:4]).float()

        self.actions = []
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.states = []

    def run_episode(self, autoclear=True):
        self.env.reset()
        for t in range(self.max_len):  # Don't infinite loop while learning
            st = self.env.get_state()
            action, log_prob, entropy = self.policy.select_action(st, t=max(self.temperature / (self.iter + 1), 1))
            res, state, reward = self.env.act(action)
            self.actions.append(action)
            self.rewards.append(reward)
            self.log_probs.append(log_prob)
            self.entropies.append(entropy)
            self.states.append(st)
            if res == 'FINISH':
                break

        rew = np.sum(self.rewards)
        if self.running_reward is None:
            self.running_reward = rew
        else:
            self.running_reward = self.running_reward * 0.99 + rew * 0.01

        self.reinforce()

        if autoclear:
            self.clear_episode_stats()

        self.iter += 1

        return rew

    def reinforce(self):
        G = 0
        policy_loss = []
        p_deltas = []
        rgains = []

        rvalues = self.value.forward(self.states[::-1])

        for r, v in zip(self.rewards[::-1], rvalues):
            G = r + self.gamma * G
            delta = G - v
            p_deltas.append(delta)
            rgains.append(G)
        p_deltas = torch.tensor(p_deltas[::-1])

        for log_prob, delta in zip(self.log_probs, p_deltas):
            policy_loss.append(-log_prob * delta)

        self.opt.zero_grad()
        self.value_opt.zero_grad()

        policy_loss = torch.stack(policy_loss).mean() - 5 * torch.tensor(self.entropies).mean()
        policy_loss.backward()
        utils.clip_grad_norm(self.policy.parameters(), 5)

        value_loss = (torch.tensor(rgains) - rvalues)**2
        value_loss = value_loss.mean()
        value_loss.backward()
        utils.clip_grad_norm(self.value.parameters(), 5)

        self.opt.step()
        self.value_opt.step()

        if self.writer is not None:
            self.writer.add_scalar('loss', policy_loss, self.iter)
            self.writer.add_scalar('gain0', G, self.iter)
            self.writer.add_scalar('episode_len', len(self.rewards), self.iter)

            self.writer.add_scalar('value_loss', value_loss, self.iter)
            #self.writer.add_scalar('mean_delta', torch.tensor(-p_deltas).abs().mean() , self.iter)
            self.writer.add_scalar('entropy', torch.tensor(self.entropies).mean(), self.iter)

            for n, p in self.policy.named_parameters():
                self.writer.add_scalar(n + '_grad', p.grad.norm(), self.iter)
                self.writer.add_scalar(n, p.norm(), self.iter)

    def clear_episode_stats(self):
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.entropies.clear()
        self.states.clear()


if __name__ == '__main__':
    import race as R
    rf_race = R.RaceTrack(R.track1, 2)

    policy = SpatialPolicy(rf_race, hidden=50)
    value = SpatialValues(rf_race, hidden=50)
    trainer = Reinforce(rf_race, policy, value=value, lr=1e-3, max_len=200, gamma=0.99)
    for k in range(1000):
        trainer.run_episode()
