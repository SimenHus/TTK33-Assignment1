import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


RNG = np.random.default_rng()

@dataclass
class GNSSData:
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
    lin_acc: 'np.ndarray[2]'
    ang_vel: 'np.ndarray[1]' = None

    def __post_init__(self) -> None:
        if self.lin_acc.shape[0] == 3 and self.ang_vel is None:
            self.ang_vel = np.array([self.lin_acc[2]])
            self.lin_acc = self.lin_acc[:2]

    @property
    def array(self) -> 'np.ndarray[3]':
        return np.concatenate([self.lin_acc, self.ang_vel])

@dataclass
class PDF:
    mean: 'np.ndarray[n]'
    sigma: 'np.ndarray[n, m]'

    def sample(self) -> 'np.ndarray[n]':
        return RNG.multivariate_normal(self.mean, self.sigma)