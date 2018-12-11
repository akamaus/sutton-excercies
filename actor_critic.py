import torch
import torch.nn.utils as utils


def multi_actor(env_constructor, policy, value, n_actors, n_episodes, writer=None, lr=0.01, **kargs):
    policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
    value_opt = torch.optim.Adam(value.parameters(), lr=lr)

    actors = []
    gens = []

    for k in range(n_actors):
        env = env_constructor()
        actor = AdvantageActorCritic(env, policy, value, async_mode=True, **kargs)
        actors.append(actor)
        gens.append(actor.gen_episode(autoclear=False))

    n_finished = 0

    while n_finished < n_episodes:
        policy_opt.zero_grad()
        value_opt.zero_grad()

        for i, g in enumerate(gens):
            try:
                next(g)
            except StopIteration as se:
                n_finished += 1
                if writer is not None:
                    actors[i].log_episode_stats(writer, n_finished)
                actors[i].clear_episode_stats()
                print(se.value)
                gens[i] = actors[i].gen_episode(autoclear=False)

        utils.clip_grad_norm(policy.parameters(), 5)
        utils.clip_grad_norm(value.parameters(), 5)

        policy_opt.step()
        value_opt.step()


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

    def run_episode(self):
        assert self.async_mode is False
        gen = self.gen_episode()
        try:
            next(gen)
            assert False
        except StopIteration as se:
            return se.value

    def gen_episode(self, t_max=None, autoclear=True):
        if t_max is None:
            t_max = self.t_max

        self.env.reset()

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

                t += 1

            if finish:
                ret = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    ret = self.value.forward(st)

            states.reverse()
            log_probs.reverse()
            rewards.reverse()
            vs = self.value.forward(states)

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

        if autoclear:
            self.clear_episode_stats()

        return t

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

    def clear_episode_stats(self):
        self.states.clear()
        self.entropies.clear()
        self.total_policy_loss = 0
        self.total_value_loss = 0
        self.n_backups = 0
        self.episode_len = 0


if __name__ == '__main__':
    from reinforce import SpatialPolicy, SpatialValues
    from tensorboardX import SummaryWriter

    torch.random.manual_seed(0)

    def mk_race():
        import race as R
        return R.RaceTrack(R.track3, 2)

    ac_race = mk_race()

    policy = SpatialPolicy(ac_race, hidden=20)
    value = SpatialValues(ac_race, hidden=20)

    if True:
        writer = SummaryWriter('logs/aac_nactors1_lr0.01_g0.99_tbackup10_tmax500_t1_ent10')
        multi_actor(mk_race, policy, value, 1, 5000, writer=writer, lr=0.01, gamma=0.99, t_max=500, t_backup=10, temperature=1)
    else:
        writer = SummaryWriter('logs/actor_critic_lr0.01_tbackup100_tmax500_t500_ent10')
        trainer = AdvantageActorCritic(mk_race(), policy, value=value, writer=writer, lr=1e-2, gamma=0.99, t_max=500, t_backup=10, temperature=1)
        for k in range(2000):
            n_steps = trainer.run_episode()
            print(n_steps)
