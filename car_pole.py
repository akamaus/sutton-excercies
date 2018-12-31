#!/usr/bin/env python3

from math import sin, cos, pi
import random
import os
import sys

class PoleBalancer:
    N_ACTIONS = 11
    MAX_FORCE = 100

    def __init__(self, car_mass=5, pole_mass=2, pole_length=0.05):
        self.car_mass = car_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length

        self.x_coord = None
        self.angle = None
        self.x_velocity = None
        self.angle_velocity = None
        self.engine_force = None
        self.calmness = 0

        self.reset()

    def reset(self, difficulty=1.0):
        self.x_coord = random.uniform(0.2, 0.8)
        self.angle = random.gauss(0, difficulty)
        self.x_velocity = random.gauss(0, difficulty)
        self.angle_velocity = 10*random.gauss(0, difficulty)
        self.engine_force = 0

    def compute_acceleration(self):
        x = self.x_coord
        a = self.angle
        vx = self.x_velocity
        va = self.angle_velocity
        m = self.pole_mass
        M = self.car_mass
        l = self.pole_length
        f = self.engine_force

        g = 9.81

        denominator = (4 * (M + m) - 3 * m * cos(a) * cos(a))

        x_accel = (4 * f + 2 * m * l * sin(a) * va * va - 3 * m * g * cos(a) * sin(a)) / denominator

        a_accel = (6 * g * sin(a) * (M + m) - 6 * f * cos(a) - 3 * m * l * sin(a) * cos(a) * va * va) / l / denominator

        return x_accel, a_accel

    def move(self, step):
        x_accel, a_accel = self.compute_acceleration()
        self.x_coord += self.x_velocity * step
        self.x_velocity += x_accel * step
        self.angle += self.angle_velocity * step
        self.angle_velocity += a_accel * step

        if self.x_coord < 0:
            self.x_coord = 0
            self.x_velocity *= -0.9
            collision = True
        elif self.x_coord > 1:
            self.x_coord = 1
            self.x_velocity *= -0.9
            collision = True
        else:
            collision = False
        return collision

    def get_state(self):
        return self.x_coord, self.angle % (2*pi), self.x_velocity, self.angle_velocity

    def act(self, a):
        assert isinstance(a, int)
        assert 0 <= a <= 11
        f = (a - 5) / 11.0 * self.MAX_FORCE
        self.engine_force = f
        collision = self.move(0.001)
        if collision:
            penalty = 100
        else:
            penalty = 0

        a = self.angle % (2*pi)
        reward = - min(abs(a), abs(2*pi - a))

        if reward > -0.1:
            self.calmness += 1
        else:
            self.calmness = 0

        if self.calmness > 300:
            res = 'FINISH'
        else:
            res = 'CONT'

        return res, self.get_state(), reward - penalty


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import PyQt5.QtCore as QtCore


class App(QMainWindow):

    def __init__(self, policy=None, difficulty=0.001):
        super().__init__()

        self.car = PoleBalancer(5, 2, 0.05)
        self.car.reset(difficulty)
        self.policy = policy

        self.title = 'PyQt paint - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 440
        self.height = 280

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

        # Add paint widget and paint
        self.m = CarWidget(self)
        self.m.move(0, 0)
        self.m.resize(self.width, self.height)

        self.external_force = None

        self.m.set_car(self.car)

        self.show()
        self.k = 0

    def set_policy(self, policy):
        self.policy = policy

    def redraw(self):
        rew = 0
        if self.external_force is None and self.policy:
            act, prob, ent = self.policy.select_action(self.car.get_state())
            res, _, rew = self.car.act(act)
        else:
            self.car.engine_force = self.external_force or 0
            self.car.move(0.001)
            res = ''

        self.m.update()
        self.k += 1
        print(f'{self.k}: {res} reward {rew:6.2f} force {self.car.engine_force:6.2f} x_vel {self.car.x_velocity:6.2f} angle_vel {self.car.angle_velocity:6.2f} angle {self.car.angle:5.1f}')

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() != QtCore.Qt.NoButton:
                p = (event.pos().x() / self.size().width() - 0.5) * 100
                print(p)
                self.external_force = p
            else:
                self.external_force = None
        return QMainWindow.eventFilter(self, source, event)


class CarWidget(QWidget):

    def set_car(self, cp):
        self.car = cp

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setPen(Qt.black)
        size = self.size()

        cx = self.car.x_coord * size.width()
        cy = size.height() / 2
        qp.drawRect(cx-5, cy, 10, 10)
        qp.drawLine(cx, cy, cx + 20*sin(self.car.angle), cy - 20*cos(self.car.angle))


def run_visualizer(policy=None, difficulty=0.001):
    app = QApplication(sys.argv)
    ex = App(policy, difficulty)

    app.installEventFilter(ex)

    timer = QTimer()
    timer.timeout.connect(ex.redraw)
    timer.start(10)
    return app.exec_()


if __name__ == '__main__':
    from argparse import ArgumentParser
    import approximators as A
    import torch

    parser = ArgumentParser()
    parser.add_argument('--approximator', choices="scaled quantized".split(), default='scaled')
    parser.add_argument('--tbackup', type=int, default=50)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--difficulty', type=float, default=0.001)
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--lr-policy', type=float, default=0.01)
    parser.add_argument('--lr-value', type=float, default=0.01)
    parser.add_argument('--num-actors', type=int, default=10)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=20)
    parser.add_argument('--id')
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--short_name', action='store_true', help='just use id without any descripting suffixes for checkpoint names and logs')
    parser.add_argument('--trainer', choices='baac maac'.split())
    parser.add_argument('--load-policy', help='initialize policy from file')
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--qsteps', type=int, default=25)
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('mode', choices=['demo', 'train', 'curriculum'])
    args = parser.parse_args()

    print('ARGV', sys.argv)

    task = PoleBalancer()

    p_hidden = args.hidden
    v_hidden = args.hidden
    t_backup = args.tbackup

    diaps = [(0, 1),  # x_coord = 0.5
             (0, 2 * pi),  # angle = 0.01
             (-5, 5),  # x_velocity = 0
             (-80, 80)]  # angle_velocity = 0

    if args.approximator == 'scaled':
        policy = A.ScaledPolicy(task, diaps, n_hidden=p_hidden, num_layers=args.num_layers)
        value = A.ScaledValue(task, diaps, n_hidden=v_hidden)
    elif args.approximator == 'quantized':
        policy = A.QuantizedPolicy(task, diaps, [args.qsteps] * 4, n_hidden=p_hidden, num_layers=args.num_layers)
        value = A.QuantizedValue(task, diaps, [args.qsteps] * 4, n_hidden=v_hidden, num_layers=args.num_layers)
    else:
        raise ValueError('Unknown approximator', args.approximator)

    if args.gpu:
        d = torch.device('cuda')
        policy.set_device(d)
        value.set_device(d)

    if args.mode == 'demo':
        if args.load_policy is not None:
            policy_state = torch.load(args.load_policy)
            policy.load_state_dict(policy_state)
        else:
            policy = None
        res = run_visualizer(policy, args.difficulty)
        sys.exit(res)
    else:
        from tensorboardX import SummaryWriter
        import actor_critic as AC

        if args.short_name:
            full_name = args.id
        else:
            full_name = f'{args.id}-{args.trainer}-carpole-act{args.num_actors}-pen100-mf200-P{args.approximator}-{args.num_layers}-h{p_hidden}-V{args.approximator}-{args.num_layers}-h{v_hidden}-tbkp50-tmax5000rnd-sdf{args.difficulty:.2}-g{args.discount:.2}-{args.mode}'
        writer = SummaryWriter(os.path.join('logs', full_name))

        if args.mode == 'train':
            baac = AC.BatchAdvantageActorCritic([PoleBalancer() for _ in range(args.num_actors)], policy=policy, value=value, writer=writer, name=args.id,
                                                lr=args.lr, gamma=args.discount, t_max=5000, t_backup=t_backup, temperature=args.temperature,
                                                difficulty=args.difficulty, episode_len_target=2000)
            baac.run_episodes(args.episodes)
        elif args.mode == 'curriculum':
            if args.trainer == 'maac':
                gain = AC.multi_actor(PoleBalancer, policy, value, writer=writer, autosave_name='26-carpole-pen100-h20-h20-curriculum', n_actors=args.num_actors,
                                      n_episodes=args.episodes, t_max=5000, t_backup=t_backup, lr=args.lr, gamma=args.discount,
                                      difficulty=args.difficulty, max_difficulty=None, gain_target=-0.05)
            elif args.trainer == 'baac':
                baac = AC.BatchAdvantageActorCritic([PoleBalancer() for _ in range(args.num_actors)], policy=policy, value=value, writer=writer, name=full_name,
                                                    lr_policy=args.lr_policy, lr_value=args.lr_value, gamma=args.discount, t_max=5000, t_backup=t_backup, temperature=args.temperature,
                                                    difficulty=args.difficulty, episode_len_target=2000, restore=args.restore)
                baac.run_episodes(n_iters=args.iters, n_episodes=args.episodes)
                print('iter', baac.iter)
                print('finished_episodes', baac.finished_episodes)
                print('BO_RESULT', baac.difficulty)
            else:
                raise ValueError('unknown trainer', args.trainer)
        else:
            raise ValueError('unknown mode', args.mode)
