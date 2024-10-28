from common import *

class Sensor:
    bias: 'np.ndarray[n]'
    sigma: 'np.ndarray[n, m]'

    @classmethod
    def sample(clc, mean: 'np.ndarray[n]') -> 'np.ndarray[n]':
        return PDF(mean + clc.bias, clc.sigma).sample()


class Gyro(Sensor):
    bias = np.array([1])*0 # [rad/s]
    sigma = np.diag([1])*1


class Accelerometer(Sensor):
    bias = np.array([1, 1])*0 # [m/s^2]
    sigma = np.diag([1, 1])*10


class GNSS(Sensor):
    bias = np.array([1, 1])*0 # [m]
    sigma = np.diag([1, 1])*0.01

    @classmethod
    def sample(clc, true_state: 'np.ndarray[2]') -> GNSSData:
        pdf = super().sample(true_state)
        return GNSSData(pdf)


class IMU:

    @staticmethod
    def sample(true_state: 'np.ndarray[3]') -> IMUData:
        linear_velocity_sample = Accelerometer.sample(true_state[:2])
        angular_velocity_sample = Gyro.sample(true_state[2])
        return IMUData(linear_velocity_sample, angular_velocity_sample)