import random
import numpy as np


class QPolicy:
    def __init__(self, env, eps=0.1):
        self.env = env
        self.eps = eps
        self.qvalues = np.zeros(env.get_sizes() + (env.N_ACTIONS,))

    def get_qvalue(self, s, a):
        return self.qvalues.__getitem__(tuple(s) + (a,))

    def update_qvalue(self, s, a, d):
        k = tuple(s) + (a,)
        self.qvalues.__setitem__(k, self.get_qvalue(s, a) + d)

    def eps_policy(self, state=None):
        if state is None:
            state = self.env.state

        s_idxs = np.argsort(self.qvalues[state[0], state[1]])
        greedy = random.random() > self.eps * (self.env.N_ACTIONS - 1) / self.env.N_ACTIONS
        if greedy:
            return s_idxs[-1]
        else:
            k = random.randint(0, self.env.N_ACTIONS - 2)
            return s_idxs[k]


class MonteCarlo(QPolicy):
    def __init__(self, env, eps=0.1):
        super().__init__(env, eps)
        self.nvisits = np.zeros_like(self.qvalues)

    def run_episode(self):
        self.env.reset()
        states = []
        actions = []
        rewards = []

        sa_dict = {}  # first visit

        # collecting episode
        s = self.env.state
        states.append(s)
        rewards.append(0)
        t = 0
        while True:
            a = self.eps_policy(s)
            res, s2, rew = self.env.act(a)
            states.append(s2)
            actions.append(a)
            rewards.append(rew)

            if (s, a) not in sa_dict:
                sa_dict[(s, a)] = t

            s = s2
            if res == 'FINISH':
                break
            t += 1

        # looping back for calculating reward
        g = 0
        t = t-1
        while t >= 0:
            g += rewards[t+1]
            s, a = states[t], actions[t]
            if sa_dict[(s, a)] == t:
                nv = self.nvisits[s[0], s[1], a]
                self.qvalues[s[0], s[1], a] = (self.qvalues[s[0], s[1], a] * nv + g) / (nv + 1)
                k = s + (a,)
                self.nvisits.__setitem__(k, self.nvisits.__getitem__(k) + 1)
            t -= 1

        return g


class Sarsa(QPolicy):
    def __init__(self, env, eps=0.1, alpha=0.1, gamma = 0.9):
        super().__init__(env, eps)
        self.alpha = alpha  # learning coeff
        self.gamma = gamma  # discount factor

    def run_episode(self):
        self.env.reset()
        prev_s = self.env.state
        prev_a = self.eps_policy(prev_s)
        g = 0
        while True:
            res, s, rew = self.env.act(prev_a)
            a = self.eps_policy()
            self.update_qvalue(prev_s, prev_a,
                               self.alpha * (rew + self.gamma * self.get_qvalue(s, a) - self.get_qvalue(prev_s, prev_a)))
            g += rew
            if res == 'FINISH':
                break
            prev_s = s
            prev_a = a
        return g
