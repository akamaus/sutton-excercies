import torch
import torch.nn.utils as utils

import numpy as np
import random

def multi_actor(env_constructor, policy, value, n_actors, n_episodes, writer=None, autosave_name=None, lr=0.01, t_max=None, gain_target=None, difficulty=None, max_difficulty=None, difficulty_gain_period=10**9, difficulty_gain_step=0.05, **kargs):
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=lr)

    actors = []
    gens = []

    def gen_t_max():
        if t_max is None:
            rnd_t_max = None
        else:
            rnd_t_max = t_max*random.uniform(0.8, 1.2)
        return rnd_t_max


    for k in range(n_actors):
        env = env_constructor()
        actor = AdvantageActorCritic(env, policy, value, async_mode=True, **kargs)
        actors.append(actor)
        gens.append(actor.gen_episode(difficulty=difficulty, autoclear=False, t_max=gen_t_max()))

    n_finished = 0
    gains = []
    last_levelup = 0

    while n_finished < n_episodes:
        policy_opt.zero_grad()
        value_opt.zero_grad()

        for i, g in enumerate(gens):
            try:
                next(g)
            except StopIteration as se:
                n_finished += 1
                if max_difficulty is not None:
                    df = min(difficulty + (max_difficulty - difficulty) / difficulty_gain_period * n_finished, max_difficulty)
                else:
                    df = difficulty
                if writer is not None:
                    actors[i].log_episode_stats(writer, n_finished)
                    if df is not None:
                        writer.add_scalar('difficulty', df, n_finished)
                actors[i].clear_episode_stats()
                gain = se.value
                gains.append(gain)
                if gain_target is not None and np.mean(gains[-10:]) > gain_target and n_finished - last_levelup > n_actors:
                    difficulty += difficulty_gain_step
                    last_levelup = n_finished
                    print('target gain', gain, 'achieved, raising difficulty to', difficulty)
                    if autosave_name is not None:
                        torch.save(policy.state_dict(), autosave_name + '_policy.cpy')
                        torch.save(value.state_dict(), autosave_name + '_value.cpy')

                print(se.value)
                gens[i] = actors[i].gen_episode(autoclear=False, difficulty=df, t_max=gen_t_max())

        utils.clip_grad_norm(policy.parameters(), 5)
        utils.clip_grad_norm(value.parameters(), 5)

        policy_opt.step()
        value_opt.step()

    return np.mean(gains)


class AdvantageActorCritic:
    def __init__(self, env, policy, value, writer=None, lr=1e-3, gamma=0.95, t_backup=5, t_max=None, temperature=1, async_mode=False):
        self.env = env
        self.policy = policy
        self.value = value
        self.gamma = gamma  # discount factor
        self.t_backup = t_backup  # maximum number of steps before backups
        self.t_max = t_max  # maximum episode length
        self.temperature = temperature
        self.async_mode = async_mode

        self.writer = writer

        if not async_mode:
            self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
            self.value_opt = torch.optim.Adam(value.parameters(), lr=lr)

        self.running_reward = None
        self.iter = 0

        # stats
        self.states = []
        self.entropies = []
        self.total_policy_loss = 0
        self.total_value_loss = 0
        self.n_backups = 0
        self.episode_len = 0
        self.gain = 0
        self.energy_spent = 0

    def run_episode(self):
        assert self.async_mode is False
        gen = self.gen_episode()
        try:
            while True:
                next(gen)
        except StopIteration as se:
            rew = se.value
            if self.running_reward is None:
                self.running_reward = rew
            else:
                self.running_reward = self.running_reward * 0.99 + rew * 0.01
            return rew

    def gen_episode(self, t_max=None, difficulty=None, autoclear=True):
        if t_max is None:
            t_max = self.t_max

        self.env.reset(difficulty=difficulty)

        if autoclear:
            self.clear_episode_stats()

        t = 1
        finish = False

        while not finish and (t_max is None or t < t_max):
            if not self.async_mode:
                self.policy_opt.zero_grad()
                self.value_opt.zero_grad()

            t_start = t
            st = self.env.get_state()
            self.states.append(st)

            entropies = []
            log_probs = []
            states = []
            rewards = []

            while not (finish or t - t_start == self.t_backup):
                states.append(st)
                action, log_prob, entropy = self.policy.select_action(st, t=max(self.temperature / (self.iter+1), 1))
                res, st, reward = self.env.act(action)

                finish = res == 'FINISH'

                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                self.entropies.append(entropy.detach())
                self.states.append(st)
                self.gain += reward
                self.energy_spent += abs(action - self.env.N_ACTIONS // 2)

                t += 1

            if finish:
                ret = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    ret = self.value.compute_value(st)

            states.reverse()
            log_probs.reverse()
            rewards.reverse()
            vs = self.value.compute_value(states)

            policy_loss = []
            value_loss = []

            for lp, r, v in zip(log_probs, rewards, vs):  # iterating in reverse order
                ret = r + ret * self.gamma
                policy_loss.append(-lp * (ret - v.detach()))
                value_loss.append((ret.detach() - v)**2)

            policy_loss = torch.stack(policy_loss).mean() - 10 * torch.tensor(entropies).mean()
            value_loss = torch.stack(value_loss).mean()

            policy_loss.backward()
            value_loss.backward()

            self.total_policy_loss += policy_loss.item()
            self.total_value_loss += value_loss.item()
            self.n_backups += 1

            if self.async_mode:
                if not finish and (t_max is None or t < t_max):
                    yield
            else:
                self.policy_opt.step()
                self.value_opt.step()

        self.iter += 1
        self.episode_len = t

        if self.writer is not None:
            self.log_episode_stats()

        return self.gain / t

    def log_episode_stats(self, writer=None, iter=None):
        if writer is None:
            writer = self.writer
        assert writer is not None

        if iter is None:
            iter = self.iter

        writer.add_scalar('policy_loss', self.total_policy_loss / self.n_backups, iter)
        writer.add_scalar('value_loss', self.total_value_loss / self.n_backups, iter)
        writer.add_scalar('entropy', torch.tensor(self.entropies).mean(), iter)
        writer.add_scalar('episode_len', self.episode_len, iter)
        writer.add_scalar('gain', self.gain, iter)
        writer.add_scalar('mean_power', self.energy_spent / self.episode_len, iter)

    def clear_episode_stats(self):
        self.states.clear()
        self.entropies.clear()
        self.total_policy_loss = 0
        self.total_value_loss = 0
        self.n_backups = 0
        self.episode_len = 0
        self.gain = 0
        self.energy_spent = 0


class BatchAdvantageActorCritic:
    def __init__(self, envs, policy, value, writer=None, name=None, lr=1e-3, gamma=0.95, t_backup=5, t_max=None, temperature=1,
                 difficulty=None, difficulty_step=0.02, mean_gain_target=None, gain_target=None, episode_len_target=None):
        self.name = name
        self.n = len(envs)
        self.env_b = envs
        self.policy = policy
        self.value = value
        self.gamma = gamma  # discount factor
        self.t_backup = t_backup  # maximum number of steps before backups
        self.t_max = t_max  # maximum episode length
        self.temperature = temperature

        self.difficulty = difficulty
        self.difficulty_step = difficulty_step
        self.mean_gain_target = mean_gain_target
        self.gain_target = gain_target
        self.episode_len_target = episode_len_target

        self.writer = writer

        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(value.parameters(), lr=lr)

        self.running_reward = None
        self.iter = 0

        # episode stats, for logging
        self.gain_history = []
        self.mean_gain_history = []
        self.episode_len_history = []
        # running episode stats
        self.state_stats = []
        self.entropy_stats = []
        self.total_policy_loss_stats = 0
        self.total_value_loss_stats = 0
        self.n_backups = 0
        self.gain_stats = torch.FloatTensor(self.n)
        self.energy_spent = torch.FloatTensor(self.n)
        self.clear_episode_stats()

    def run_episodes(self, n_episodes, t_max=None):
        if t_max is None:
            t_max = self.t_max

        if t_max is not None:
            d = torch.distributions.Uniform(t_max * 0.9, t_max * 1.1)
            t_max = d.sample((self.n,))

        for env in self.env_b:
            env.reset(difficulty=self.difficulty)

        self.clear_episode_stats()

        t = 1
        finished_episodes = 0
        t_episode_start_b = [t] * self.n
        e_gains = torch.zeros(self.n)
        last_log = finished_episodes
        last_difficulty_increment = t

        while finished_episodes < n_episodes:
            self.policy_opt.zero_grad()
            self.value_opt.zero_grad()

            t_start = t

            st_b = [env.get_state() for env in self.env_b]

            entropy_bs = []
            log_prob_bs = []
            state_bs = []
            reward_bs = []

            while not t - t_start == self.t_backup:
                state_bs.append(st_b)
                action_b, log_prob_b, entropy_b = self.policy.select_action(st_b, t=max(self.temperature / (self.iter+1), 1))

                res_b = []
                st_b = []
                finished_idx = []
                elapsed_idx = []
                reward_b = []
                for i in range(self.n):
                    res, st, reward = self.env_b[i].act(action_b[i].item())
                    res_b.append(res)
                    st_b.append(st)
                    reward_b.append(reward)
                    if res == 'FINISH':
                        finished_idx.append(i)
                    if t_max is not None and t - t_episode_start_b[i] >= t_max[i]:
                        elapsed_idx.append(i)

                log_prob_bs.append(log_prob_b)
                reward_bs.append(reward_b)
                entropy_bs.append(entropy_b)

                self.entropy_stats.append(entropy_b.detach())
                self.state_stats.append(st_b)
                self.gain_stats += torch.tensor(reward_b)
                #self.energy_spent_b += abs(action_b - self.env_b[0].N_ACTIONS // 2)

                t += 1
                self.iter += 1

                if len(finished_idx) > 0 or len(elapsed_idx) > 0:
                    break

            elapsed_set = set(elapsed_idx)

            with torch.no_grad():
                ret_b = self.value.compute_value(st_b)

            for i in set(finished_idx + elapsed_idx):
                if i not in elapsed_set:
                    ret_b[i] = 0
                t_max[i] = d.sample()
                ep_len = t - t_episode_start_b[i]
                self.mean_gain_history.append((self.gain_stats[i] / ep_len).item())
                self.gain_history.append(self.gain_stats[i].item())
                self.episode_len_history.append(ep_len)

                self.env_b[i].reset(difficulty=self.difficulty)
                t_episode_start_b[i] = t

                finished_episodes += 1

            state_bs.reverse()
            log_prob_bs.reverse()
            reward_bs.reverse()
            vs = self.value.compute_value(state_bs)

            policy_loss = []
            value_loss = []

            for lp, r, v in zip(log_prob_bs, reward_bs, vs):  # iterating in reverse order
                ret_b = torch.tensor(r).to(ret_b) + ret_b * self.gamma
                policy_loss.append(-lp * (ret_b - v.detach()))
                value_loss.append((ret_b.detach() - v)**2)

            policy_loss = torch.stack(policy_loss).mean() - 10 * torch.tensor(entropy_bs).to(ret_b).mean()
            value_loss = torch.stack(value_loss).mean()

            policy_loss.backward()
            value_loss.backward()

            self.total_policy_loss_stats += policy_loss.item()
            self.total_value_loss_stats += value_loss.item()
            self.n_backups += 1

            utils.clip_grad_norm(self.policy.parameters(), 5)
            utils.clip_grad_norm(self.value.parameters(), 5)

            self.policy_opt.step()
            self.value_opt.step()

            if finished_episodes - last_difficulty_increment > self.n:
                recent_mean_gain = torch.tensor(self.mean_gain_history[-self.n:]).mean()
                recent_gain = torch.tensor(self.gain_history[-self.n:]).mean()
                recent_episode_len = torch.tensor(self.episode_len_history[-self.n:]).float().mean()
                if (self.mean_gain_target is not None and recent_mean_gain > self.mean_gain_target) or \
                        (self.gain_target is not None and recent_gain > self.gain_target) or \
                        (self.episode_len_target is not None and recent_episode_len < self.episode_len_target):
                    self.difficulty += self.difficulty_step
                    last_difficulty_increment = finished_episodes
                    print(f'target gain level achieved, raising difficulty to', self.difficulty)
                    torch.save(self.policy.state_dict(), f'{self.name}_policy_best.cpy')
                    torch.save(self.value.state_dict(), f'{self.name}_value_best.cpy')

            if self.writer is not None and finished_episodes - last_log >= 10:
                self.log_episode_stats(log_from=last_log, iter=finished_episodes)
                self.clear_episode_stats()
                last_log = finished_episodes

            if finished_episodes % 100 == 0:
                torch.save(self.policy.state_dict(), f'{self.name}_policy_backup.cpy')
                torch.save(self.value.state_dict(), f'{self.name}_value_backup.cpy')

        return torch.tensor(self.gain_stats).mean()

    def log_episode_stats(self, log_from, writer=None, iter=None):
        if writer is None:
            writer = self.writer
        assert writer is not None

        if iter is None:
            iter = self.iter

        writer.add_scalar('policy_loss', self.total_policy_loss_stats / self.n_backups, iter)
        writer.add_scalar('value_loss', self.total_value_loss_stats / self.n_backups, iter)
        writer.add_scalar('entropy', torch.tensor(self.entropy_stats).mean(), iter)
        writer.add_scalar('episode_len', torch.tensor(self.episode_len_history[log_from:]).float().mean(), iter)
        writer.add_scalar('mean_gain', torch.tensor(self.mean_gain_history[log_from:]).mean(), iter)
        writer.add_scalar('gain', torch.tensor(self.gain_history[log_from:]).mean(), iter)
        #writer.add_scalar('mean_power', (self.energy_spent / self.episode_len_stats.float()).mean(), iter)
        writer.add_scalar('difficulty', self.difficulty, iter)
        print('mean gain', self.mean_gain_history[-1])

    def clear_episode_stats(self):
        self.state_stats.clear()
        self.entropy_stats.clear()
        self.total_policy_loss_stats = 0
        self.total_value_loss_stats = 0
        self.n_backups = 0
        self.gain_stats.zero_()
        self.energy_spent.zero_()


if __name__ == '__main__':
    from argparse import ArgumentParser
    import random
    from approximators import SpatialPolicy, SpatialValue
    from tensorboardX import SummaryWriter

    random.seed(0)
    torch.random.manual_seed(0)

    def mk_race():
        import race as R
        return R.RaceTrack(R.track5, 5)

    ac_race = mk_race()

    policy = SpatialPolicy(ac_race, n_hidden=20)
    value = SpatialValue(ac_race, n_hidden=20)

    parser = ArgumentParser()
    parser.add_argument('--num-episodes', default=2000, type=int)
    parser.add_argument('--name', default='unnamed')
    parser.add_argument('mode', choices="baac maac aac".split())
    args = parser.parse_args()
    mode = args.mode
    n_actors = 1
    if mode == 'baac':
        writer = SummaryWriter(f'logs/baac_track5s5_{args.name}_nactors{n_actors}_lr0.01_g1_tbackup8_tmax500_t1_ent10_test3_gclip')
        baac = BatchAdvantageActorCritic([mk_race() for _ in range(n_actors)], policy=policy, value=value, writer=writer, lr=0.01, gamma=1, t_max=500, t_backup=8, temperature=1)
        baac.run_episodes(args.num_episodes)
    elif mode == 'maac':
        writer = SummaryWriter(f'logs/maac_track5s5_{args.name}_nactors{n_actors}_lr0.01_g1_tbackup8_tmax500_t1_ent10_test')
        multi_actor(mk_race, policy, value, n_actors, args.num_episodes, writer=writer, lr=0.01, gamma=1, t_max=500, t_backup=8, temperature=1)
    elif mode == 'aac':
        writer = SummaryWriter('logs/actor_critic_lr0.01_tbackup8_tmax500_t1_ent10')
        trainer = AdvantageActorCritic(mk_race(), policy, value=value, writer=writer, lr=1e-2, gamma=0.99, t_max=500, t_backup=8, temperature=1)
        for k in range(args.num_episodes):
            n_steps = trainer.run_episode()
            print(n_steps)
    else:
        raise ValueError('unknown mode', mode)
