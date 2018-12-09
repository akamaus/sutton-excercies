import torch
import torch.nn.utils as utils


class AdvantageActorCritic:
    def __init__(self, env, policy, value, writer=None, lr=1e-3, gamma=0.95, t_backup=5, t_max=None, temperature=1):
        self.env = env
        self.policy = policy
        self.value = value
        self.gamma = gamma  # discount factor
        self.t_backup = t_backup  # maximum number of steps before backups
        self.t_max = t_max  # maximum episode length
        self.temperature = temperature

        self.writer = writer
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(value.parameters(), lr=lr)
        self.running_reward = None
        self.iter = 0

        self.states = []
        self.entropies = []

    def run_episode(self, t_max=None, autoclear=True):
        if t_max is None:
            t_max = self.t_max

        self.env.reset()

        t = 1
        finish = False

        total_policy_loss = 0
        total_value_loss = 0
        n_backups = 0

        while not finish and (t_max is None or t < t_max):
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
                self.states.append(st)
                entropies.append(entropy)
                self.entropies.append(entropy.detach())

                finish = res == 'FINISH'

                log_probs.append(log_prob)
                rewards.append(reward)

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

            policy_loss = torch.tensor(0.0)
            value_loss = torch.tensor(0.0)

            for lp, r, v in zip(log_probs, rewards, vs):  # iterating in reverse order
                ret = r + ret * self.gamma
                policy_loss -= lp * (ret - v.detach())
                value_loss += (ret.detach() - v)**2

            policy_loss = policy_loss / len(states) - 5 * torch.tensor(entropies).mean()
            value_loss /= len(states)

            policy_loss.backward()
            value_loss.backward()

            utils.clip_grad_norm(self.policy.parameters(), 5)
            utils.clip_grad_norm(self.value.parameters(), 5)

            self.policy_opt.step()
            self.value_opt.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_backups += 1

        if self.writer is not None:
            self.writer.add_scalar('policy_loss', total_policy_loss / n_backups, self.iter)
            self.writer.add_scalar('value_loss', total_value_loss / n_backups, self.iter)
            self.writer.add_scalar('entropy', torch.tensor(self.entropies).mean(), self.iter)
            self.writer.add_scalar('episode_len', t, self.iter)

        self.iter += 1

        if autoclear:
            self.clear_episode_stats()

    def clear_episode_stats(self):
        self.states.clear()


if __name__ == '__main__':
    import race as R
    from reinforce import SpatialPolicy, SpatialValues

    torch.random.manual_seed(0)

    ac_race = R.RaceTrack(R.track5, 2)
    policy = SpatialPolicy(ac_race, hidden=50)
    value = SpatialValues(ac_race, hidden=50)
    trainer = AdvantageActorCritic(ac_race, policy, value=value, lr=1e-3, gamma=0.99, t_max=1000)
    for k in range(1000):
        trainer.run_episode()
