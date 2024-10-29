from common import *

class Sensor:
    """
    Base class for sensors
    """
    bias: 'np.ndarray[n]'
    sigma: 'np.ndarray[n, m]'

    @classmethod
    def sample(clc, mean: 'np.ndarray[n]') -> 'np.ndarray[n]':
        return PDF(mean + clc.bias, clc.sigma).sample()


class Gyro(Sensor):
    bias = np.array([1])*0 # [rad/s]
    sigma = np.diag([1])*sigma_gyro


class Accelerometer(Sensor):
    bias = np.array([1])*0 # [m/s^2]
    sigma = np.diag([1])*sigma_acc


class GNSS(Sensor):
    bias = np.array([1, 1])*0 # [m]
    sigma = np.diag([1, 1])*sigma_gnss
    sample_rate = 1 # [Hz]

    @classmethod
    def sample(clc, true_state: 'np.ndarray[2]') -> GNSSData:
        pdf = super().sample(true_state)
        return GNSSData(pdf)


class IMU:
    sample_rate = 100 # [Hz]

    @staticmethod
    def sample(true_acc: float, true_ang: float) -> IMUData:
        linear_velocity_sample = Accelerometer.sample(true_acc)
        angular_velocity_sample = Gyro.sample(true_ang)
        return IMUData(linear_velocity_sample, angular_velocity_sample)