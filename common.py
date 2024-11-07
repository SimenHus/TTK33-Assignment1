import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


RNG = np.random.default_rng() # RNG generator used in gaussian sampling
dt = 0.01 # Time step [s]

sigma_acc_base = 5 # Accelerometer std deviation
sigma_gyro_base = 2 # Gyro std deviation
sigma_GNSS_base = 0.01 # GNSS std deviation

@dataclass
class GNSSData:
    """
    Dataclass for GNSS measurements
    """
    x: float
    y: float = None

    def __post_init__(self) -> None:
        if type(self.x) == np.ndarray:
            self.y = self.x[1]
            self.x = self.x[0]

    @property
    def array(self) -> 'np.ndarray[2]':
        return np.array([self.x, self.y])
     

@dataclass
class IMUData:
    """
    Dataclass for IMU measurements
    """
    lin_acc: 'np.ndarray[1]'
    ang_vel: 'np.ndarray[1]' = None

    def __post_init__(self) -> None:
        if self.lin_acc.shape[0] == 2 and self.ang_vel is None:
            self.ang_vel = np.array([self.lin_acc[1]])
            self.lin_acc = self.lin_acc[0:1]

    @property
    def array(self) -> 'np.ndarray[2]':
        return np.concatenate([self.lin_acc, self.ang_vel])
    

@dataclass
class PDF:
    mean: 'np.ndarray[n]'
    sigma: 'np.ndarray[n, m]'

    def sample(self) -> 'np.ndarray[n]':
        return RNG.multivariate_normal(self.mean, self.sigma)
    