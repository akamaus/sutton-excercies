import random
import numpy as np


class QPolicy:
    def __init__(self, env, eps=0.1, alpha=0.1):
        self.env = env
        self.eps = eps  # greedines
        self.alpha = alpha  # update smoothness
        self.qvalues = np.zeros(env.get_sizes() + (env.N_ACTIONS,))

    def get_qvalue(self, s, a=None):
        if a is None:
            pos = tuple(s)
        else:
            pos = tuple(s) + (a,)
        return self.qvalues.__getitem__(pos)

    def update_qvalue(self, s, a, tgt):
        k = tuple(s) + (a,)
        val = self.qvalues.__getitem__(k)
        self.qvalues.__setitem__(k, val + self.alpha*(tgt - val))

    def eps_policy(self, state=None, eps=None):
        if state is None:
            state = self.env.get_state()
        if eps is None:
            eps = self.eps

        s_idxs = np.argsort(self.get_qvalue(state, None))
        greedy = random.random() > eps * (self.env.N_ACTIONS - 1) / self.env.N_ACTIONS
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
        super().__init__(env, eps=eps, alpha=alpha)
        self.gamma = gamma  # discount factor

    def run_episode(self, callback=None):
        self.env.reset()
        prev_s = self.env.get_state()
        prev_a = self.eps_policy(prev_s)
        g = 0
        if callback is not None:
            callback()
        while True:
            res, s, rew = self.env.act(prev_a)
            if callback is not None:
                callback()

            a = self.eps_policy()
            self.update_qvalue(prev_s, prev_a, rew + self.gamma * self.get_qvalue(s, a))
            g += rew
            if res == 'FINISH':
                break
            prev_s = s
            prev_a = a

        return g


class QLearning(QPolicy):
    def __init__(self, env, eps=0.1, alpha=0.1, gamma = 0.9):
        super().__init__(env, eps=eps, alpha=alpha)
        self.gamma = gamma  # discount factor

    def run_episode(self, callback=None, eps=None):
        self.env.reset()
        g = 0
        s = self.env.get_state()

        if callback is not None:
            callback()
        while True:
            a = self.eps_policy(eps=eps)
            res, s2, reward = self.env.act(a)
            if callback is not None:
                callback()

            qs = self.get_qvalue(s2)

            self.update_qvalue(s, a, reward + self.gamma * max(qs))
            g += reward
            if res == 'FINISH':
                break
            s = s2
        return g


class NStepSarsa(QPolicy):
    def __init__(self, env, n, eps=0.1, alpha=0.1, gamma=0.9):
        super().__init__(env, eps=eps, alpha=alpha)
        self.gamma = gamma  # discount factor
        self.n = n  # length of backup chain

    def run_episode(self, callback=None, eps=None):
        self.env.reset()
        states = []
        actions = []
        rewards = []

        s = self.env.get_state()
        a = self.eps_policy(s, eps=eps)
        states.append(s)
        actions.append(a)
        rewards.append(0)

        t_max = 10**9  # length of current episode

        for t in range(10**9):
            if t < t_max:
                res, s, rew = self.env.act(a)
                states.append(s)
                rewards.append(rew)
                if callback is not None:
                    callback()
                if res == 'FINISH':
                    t_max = t+1
                else:
                    a = self.eps_policy(s, eps=eps)
                    actions.append(a)

            tau = t - self.n + 1  # time in past where q-value being updated

            if tau >= 0:
                gain = 0
                d = 1  # current cumulative discount
                for i in range(tau+1, min(tau+self.n, t_max)+1):
                    gain += d * rewards[i]
                    d *= self.gamma
                if tau + self.n < t_max:
                    gain += d * self.get_qvalue(states[tau+self.n], actions[tau+self.n])

                self.update_qvalue(states[tau], actions[tau], gain)

            if tau == t_max - 1:
                break

        return np.sum(rewards)
