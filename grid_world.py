import numpy as np

from table_algos import QPolicy

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

    def get_sizes(self):
        return self.rsize, self.csize

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
