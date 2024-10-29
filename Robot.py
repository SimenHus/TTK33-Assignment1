from common import *

class DifferentialDrive:
    """
    Modified system model to utilize acceleration as input instead of velocity.
    The acceleration is given as a single state value of forward/backward body acceleration,
    this is to make integration to velocity simple.
    This change is don simply to have more control of IMU measurement precision in simulation.
    Simulation wise it is equal when using a constant acceleration
    """

    @staticmethod
    def f(x: 'np.ndarray[4]', u: 'np.ndarray[2]') -> 'np.ndarray[4]':
        x_pos, y_pos, vel, theta = x
        acc, ang_vel = u

        xdot = np.array([
            np.cos(theta)*vel,
            np.sin(theta)*vel,
            acc,
            ang_vel
        ])

        x_next = x + dt*xdot# Euler integration
        return x_next