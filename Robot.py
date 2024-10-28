from common import *

class DifferentialDrive:

    @staticmethod
    def f(x: 'np.ndarray[3]', u: 'np.ndarray[2]', h: float) -> 'np.ndarray[3]':

        theta = x[2]

        B = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, 1]
        ])

        x_next = x + h*B@u# Euler integration

        return x_next