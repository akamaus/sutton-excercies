from math import sin, cos

class PoleBalancer:
    def __init__(self, car_mass, pole_mass, pole_length):
        self.car_mass = car_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length

        self.x_coord = 0.5
        self.angle = 0.01
        self.x_velocity = 0
        self.angle_velocity = 0
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


import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import PyQt5.QtCore as QtCore


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt paint - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 440
        self.height = 280
        self.initUI()

    def initUI(self):
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

        self.car = PoleBalancer(5, 2, 0.05)
        self.m.set_car(self.car)

        self.show()
        self.k = 0

    def redraw(self):
        self.car.move(0.001)
        self.m.update()
        self.k += 1
        print(self.k, self.car.x_velocity, self.car.angle_velocity)

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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()

    app.installEventFilter(ex)

    timer = QTimer()
    timer.timeout.connect(ex.redraw)
    timer.start(10)

    sys.exit(app.exec_())
