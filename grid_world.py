import numpy as np
import random

class Environment:
    NORTH = 0
    EAST = 1
    WEST = 2
    SOUTH = 3

    N_ACTIONS = 4

    def __init__(self, rsize, csize):
        self.rsize = rsize
        self.csize = csize
        self.start_state = (0, 0)
        self.goal_state = (0, csize - 1)
        self.state = None
        self.reset()

    def reset(self):
        self.state = (0, 0)

    def act(self, move):
        if move == self.NORTH:
            self.state = (self.state[0] - 1, self.state[1])
        elif move == self.SOUTH:
            self.state = (self.state[0] + 1, self.state[1])
        elif move == self.WEST:
            self.state = (self.state[0], self.state[1] - 1)
        elif move == self.EAST:
            self.state = (self.state[0], self.state[1] + 1)
        else:
            raise ValueError('Unknown move')

        self.state = (min(self.rsize-1, max(self.state[0], 0)), min(self.csize-1, max(self.state[1], 0)))

        if 1 <= self.state[1] < self.csize - 1 and self.state[0] == 0:
            self.state = (0, 0)

        if self.state == self.goal_state:
            return 'FINISH', self.state, -1
        else:
            return 'CONT', self.state, -1


class QPolicy:
    def __init__(self, env, eps=0.1):
        self.env = env
        self.eps = eps
        self.qvalues = np.zeros([env.rsize, env.csize, env.N_ACTIONS])

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
                self.nvisits[s[0], s[1], a] += 1
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
            self.qvalues[prev_s[0], prev_s[1], prev_a] += self.alpha * (rew + self.gamma * self.qvalues[s[0], s[1], a] - self.qvalues[prev_s[0], prev_s[1], prev_a])
            g += rew
            if res == 'FINISH':
                break
            prev_s = s
            prev_a = a
        return g


if __name__ == '__main__':
    import unittest

    class EnvTest(unittest.TestCase):
        def test_fall(self):
            env1 = Environment(3, 3)
            r = env1.act(Environment.SOUTH)
            self.assertEqual(r, ('CONT', (1, 0), -1))
            r = env1.act(Environment.WEST)
            self.assertEqual(r, ('CONT', (1, 0), -1))
            r = env1.act(Environment.EAST)
            self.assertEqual(r, ('CONT', (1, 1), -1))
            r = env1.act(Environment.NORTH)
            self.assertEqual(r, ('CONT', (0, 0), -1))

        def test_finish(self):
            env1 = Environment(2, 3)
            r = env1.act(Environment.SOUTH)
            self.assertEqual(r, ('CONT', (1, 0), -1))
            r = env1.act(Environment.WEST)
            self.assertEqual(r, ('CONT', (1, 0), -1))
            r = env1.act(Environment.EAST)
            self.assertEqual(r, ('CONT', (1, 1), -1))
            r = env1.act(Environment.EAST)
            self.assertEqual(r, ('CONT', (1, 2), -1))
            r = env1.act(Environment.NORTH)
            self.assertEqual(r, ('FINISH', (0, 2), -1))

    class MCTest(unittest.TestCase):
        def test_tiny(self):
            env1 = Environment(3, 3)
            mc = MonteCarlo(env1, 0.1)
            mc.run_episode()

    class SarsaTest(unittest.TestCase):
        def test_tiny(self):
            env1 = Environment(3, 3)
            sr = Sarsa(env1, 0.1)
            sr.run_episode()



    unittest.main()
