import numpy as np


class RaceEpisodeVisualizer:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.visits = np.zeros(env.get_sizes()[:2], dtype=float)

    def process_episode(self, states):
        self.visits *= self.gamma
        for s in states:
            self.visits[s[0], s[1]] += 1

    def draw(self, ax):
        ax.imshow(self.visits)
#        ax.colorbar()

