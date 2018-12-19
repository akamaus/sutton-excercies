#!/usr/bin/env python3
from unittest import *

import approximators as A
import reinforce as RF
import actor_critic as AC

import race as R

# if rew > trainer.running_reward:
#     print('↑', end='')
# else:
#     print('↓', end='')


class TestReinforceTraining(TestCase):
    def test_small_norm(self):
        race = R.RaceTrack(R.track1, 2)
        policy = A.NormPolicy(race, n_hidden=10)
        value = A.NormValue(race, n_hidden=10)

        trainer = RF.Reinforce(race, policy, value=value, lr=1e-3, max_len=200, gamma=1)
        rew = None
        for k in range(2000):
            rew = trainer.run_episode()

        print('mean reward for rf_small_norm', trainer.running_reward)
        self.assertGreater(rew, -40)

    def test_small_spatial(self):
        race = R.RaceTrack(R.track1, 2)
        policy = A.SpatialPolicy(race, n_hidden=10)
        value = A.SpatialValue(race, n_hidden=10)

        trainer = RF.Reinforce(race, policy, value=value, lr=1e-3, max_len=200, gamma=1)
        rew = None
        for k in range(2000):
            rew = trainer.run_episode()

        print('mean reward for rf_small_spatial', trainer.running_reward)
        self.assertGreater(rew, -30)


class TestActorCriticTraining(TestCase):
    def test_small_norm(self):
        race = R.RaceTrack(R.track1, 2)
        policy = A.NormPolicy(race, n_hidden=10)
        value = A.NormValue(race, n_hidden=10)

        trainer = AC.AdvantageActorCritic(race, policy, value=value, lr=1e-3, t_max=200, gamma=1)
        rew = None
        for k in range(500):
            rew = trainer.run_episode()

        print('mean reward for ac_small_norm', trainer.running_reward)
        self.assertGreater(rew, -40)

    def test_small_spatial(self):
        race = R.RaceTrack(R.track1, 2)
        policy = A.SpatialPolicy(race, n_hidden=10)
        value = A.SpatialValue(race, n_hidden=10)

        trainer = AC.AdvantageActorCritic(race, policy, value=value, lr=1e-3, t_max=200, gamma=1)
        rew = None
        for k in range(500):
            rew = trainer.run_episode()

        print('mean reward ac_small_spatial', trainer.running_reward)
        self.assertGreater(rew, -30)

    def test_multiactor_spatial(self):
        race = R.RaceTrack(R.track1, 2)
        policy = A.SpatialPolicy(race, n_hidden=10)
        value = A.SpatialValue(race, n_hidden=10)

        gain = AC.multi_actor(lambda: R.RaceTrack(R.track1, 2), policy, value, n_actors=10, n_episodes=1000, lr=0.01, gamma=1)

        print('mean reward ac_small_spatial', gain)
        self.assertGreater(gain, -10)


if __name__ == '__main__':
    race = R.RaceTrack(R.track1, 2)
    policy = A.SpatialPolicy(race, n_hidden=10)
    value = A.SpatialValue(race, n_hidden=10)

    gain = AC.multi_actor(lambda: R.RaceTrack(R.track1, 2), policy, value, n_actors=10, n_episodes=1000, lr=0.01, gamma=1)

    main()
