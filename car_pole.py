from math import sin, cos, pi
import random

class PoleBalancer:
    N_ACTIONS = 11

    def __init__(self, car_mass=5, pole_mass=2, pole_length=0.05):
        self.car_mass = car_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length

        self.x_coord = None
        self.angle = None
        self.x_velocity = None
        self.angle_velocity = None
        self.engine_force = None

        self.reset()

    def reset(self, difficulty=1.0):
        self.x_coord = random.uniform(0, 1)
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
        elif self.x_coord > 1:
            self.x_coord = 1
            self.x_velocity *= -0.9

    def get_state(self):
        return self.x_coord, self.angle % (2*pi), self.x_velocity, self.angle_velocity

    def act(self, a):
        assert isinstance(a, int)
        assert 0 <= a <= 11
        f = (a - 5) / 11.0 * 100.0
        self.engine_force = f
        self.move(0.001)
        a = self.angle % (2*pi)
        reward = - min(abs(a), abs(2*pi - a))
        return 'CONT', self.get_state(), reward


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
        self.initUI()

    def initUI(self):
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

        self.m.set_car(self.car)

        self.show()
        self.k = 0

    def set_policy(self, policy):
        self.policy = policy

    def redraw(self):
        rew = None
        if self.policy:
            act, prob, ent = self.policy.select_action(self.car.get_state())
            _, _, rew = self.car.act(act)
        else:
            self.car.move(0.001)

        self.m.update()
        self.k += 1
        print(self.k, 'reward', rew, 'force', self.car.engine_force,  'x_vel', self.car.x_velocity, 'angle_vel', self.car.angle_velocity, 'angle', self.car.angle)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.MouseMove:
            if event.buttons() != QtCore.Qt.NoButton:
                p = (event.pos().x() / self.size().width() - 0.5) * 100
                print(p)
                self.car.engine_force = p
            else:
                self.car.engine_force = 0
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

    if policy is None:
        app.installEventFilter(ex)

    timer = QTimer()
    timer.timeout.connect(ex.redraw)
    timer.start(10)
    return app.exec_()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('mode', choices=['demo', 'train'])
    args = parser.parse_args()
    if args.mode == 'demo':
        res = run_visualizer()
        sys.exit(res)
    elif args.mode == 'train':
        import approximators as A
        import actor_critic as AC

        task = PoleBalancer()

        diaps = [(0,1),  # x_coord = 0.5
                 (0, 0.2*pi),  # angle = 0.01
                 (-2, 2),  # x_velocity = 0
                 (-50, 50)]  # angle_velocity = 0

        policy = A.ScaledPolicy(task, diaps, n_hidden=10)
        value = A.ScaledValue(task, diaps, n_hidden=10)

        gain = AC.multi_actor(PoleBalancer, policy, value, n_actors=10, n_episodes=1, t_max = 2000, lr=0.01, gamma=1)


