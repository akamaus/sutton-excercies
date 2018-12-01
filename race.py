import random

import numpy as np


track1 = """
##########
###...F###
###...F###
##...#####
##...#####
##S#######
##########

"""


track2 = """
##########
###...F###
###...F###
##...#####
##...#####
##SSS#####
##########
"""

track3 = """
###############
###.......#####
###........####
##....##....###
##...####...###
##SSS####...###
######F.....###
######F....####
###############
"""

track4 = """
###############
###.......#####
###........####
##....##....###
##...####...###
##SSS####...###
######......###
#FF##......####
#..#..#########
#..#...########
#..##...#######
#..###....#####
#..###.....####
##..##......###
##..####....###
###.........###
######.....####
###############
"""

def sign(v):
    if v == 0:
        return 0
    else:
        return v // abs(v)

class RaceTrack:
    STILL = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    SOUTH = 4

    N_ACTIONS = 5

    def __init__(self, track_sym_map, max_speed):
        track_sym_map = [l.strip() for l in track_sym_map.splitlines() if len(l) > 0]

        self.rsize = len(track_sym_map)
        self.csize = len(track_sym_map[0])
        self.max_speed = max_speed

        self.start_states = set()
        self.finish_states = set()

        self.passable_mask = np.zeros((self.rsize, self.csize), dtype=bool)

        self.state_log = []

        for ri, row in enumerate(track_sym_map):
            for ci, c in enumerate(row):
                pos = (ri, ci)
                if c == 'S':
                    self.passable_mask[pos] = True
                    self.start_states.add(pos)
                elif c == 'F':
                    self.passable_mask[pos] = True
                    self.finish_states.add(pos)
                elif c == '.':
                    self.passable_mask[pos] = True
                elif c == '#':
                    pass
                else:
                    raise('Unknown symbol in map', c)

        self.pos = None
        self.vel = None
        self.reset()

    def get_state(self):
        return tuple(self.pos + self.vel)

    def get_sizes(self):
        return self.rsize, self.csize, self.max_speed*2+1, self.max_speed*2+1

    def reset(self):
        self.pos = list(random.choice(list(self.start_states)))
        self.vel = [0, 0]
        self.state_log = []

    def trim_vel(self, v):
        return min(self.max_speed, max(-self.max_speed, v))

    def act(self, move):
        if move == self.NORTH:
            self.vel[0] -= 1
        elif move == self.SOUTH:
            self.vel[0] += 1
        elif move == self.WEST:
            self.vel[1] += 1
        elif move == self.EAST:
            self.vel[1] -= 1
        elif move == self.STILL:
            pass
        else:
            raise ValueError('Unknown move')

        self.vel[0] = self.trim_vel(self.vel[0])
        self.vel[1] = self.trim_vel(self.vel[1])

        finished = False
        reward = -1

        rvel = list(self.vel)
        while rvel[0] != 0 or rvel[1] != 0:
            npos = list(self.pos)
            if abs(rvel[0]) > abs(rvel[1]):
                s = sign(rvel[0])
                npos[0] += s
                rvel[0] -= s
            else:
                s = sign(rvel[1])
                npos[1] += s
                rvel[1] -= s

            if tuple(npos) in self.finish_states:
                finished = True

            if self.passable_mask[npos[0], npos[1]]:
                self.pos = npos
            else:
                reward -= (self.vel[0]**2 + self.vel[1]**2)
                self.vel = [0, 0]
                break

        if finished:
            res = 'FINISH'
        else:
            res = 'CONT'

        self.state_log.append(self.get_state())
        return res, self.get_state(), reward

    def print(self):
        for ri in range(self.rsize):
            for ci in range(self.csize):
                c = '.'
                pos = (ri, ci)
                if pos in self.start_states:
                    c = 'S'
                elif pos in self.finish_states:
                    c = 'F'
                elif not self.passable_mask[pos[0], pos[1]]:
                    c = '#'
                if pos == tuple(self.pos):
                    c = 'C'
                print(c, end='')
            print()
        print()


if __name__ == '__main__':
    import unittest

    class TrackTest(unittest.TestCase):
        def test_move(self):
            t1 = RaceTrack(track1, max_speed=1)
            r = t1.act(RaceTrack.SOUTH)
            self.assertEqual(r, ('CONT', (5, 2, 0, 0), -2))

            r = t1.act(RaceTrack.NORTH)
            self.assertEqual(r, ('CONT', (4, 2, -1, 0), -1))
            r = t1.act(RaceTrack.NORTH)
            self.assertEqual(r, ('CONT', (3, 2, -1, 0), -1))
            r = t1.act(RaceTrack.NORTH)
            self.assertEqual(r, ('CONT', (3, 2, 0, 0), -2))

        def test_finish(self):
            t1 = RaceTrack(track1, max_speed=1)
            r = t1.act(RaceTrack.NORTH)
            self.assertEqual(r, ('CONT', (4, 2, -1, 0), -1))
            r = t1.act(RaceTrack.WEST)
            self.assertEqual(r, ('CONT', (3, 3, -1, 1), -1))
            r = t1.act(RaceTrack.STILL)
            self.assertEqual(r, ('CONT', (2, 4, -1, 1), -1))
            r = t1.act(RaceTrack.SOUTH)
            self.assertEqual(r, ('CONT', (2, 5, 0, 1), -1))
            r = t1.act(RaceTrack.STILL)
            self.assertEqual(r, ('FINISH', (2, 6, 0, 1), -1))


    unittest.main()
