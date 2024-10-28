from common import *
from Sensor import IMU, GNSS
from Robot import DifferentialDrive
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from functools import partial




h = 0.1 # Time step [s]
simulation_length = 30 # Simulation time [s]
timeseries = np.arange(0, simulation_length, h) # Timeseries



x_0 = np.array([0, 0, 0]) # (x, y, heading) Initial value

v_b = np.ones([len(timeseries)])*5 # [m/s] Body frame speed of the vehicle in simulation
w_b = np.ones([len(timeseries)])*1 # [rad/s] Body frame angular velocity of vehicle in simulation
trajectory_true = np.zeros([len(x_0), len(timeseries)]) # Variable to store true trajectory of the vehicle
trajectory_measured = np.zeros([len(x_0), len(timeseries)]) # Variable to store EKF measurement of the vehicle trajectory



trajectory_true[:, 0] = x_0
for i, t in enumerate(timeseries):

    pos_meas = GNSS.sample(trajectory_true[:2, i])
    IMU_meas = IMU.sample(trajectory_true[:, i])

    trajectory_measured[:, i] = np.append(pos_meas.array, 0)

    if i == len(timeseries)-1: break
    trajectory_true[:, i+1] = DifferentialDrive.f(trajectory_true[:, i], np.array([v_b[i], w_b[i]]), h)


fig, ax = plt.subplots()
xdata, ydata = [], []
est, = ax.plot([], [], label='Estimated')
gt, = ax.plot([], [], label='Ground truth')

lim = 10
ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.grid()
ax.legend()

def update(i):
    est.set_data(trajectory_measured[0, :i], trajectory_measured[1, :i])
    gt.set_data(trajectory_true[0, :i], trajectory_true[1, :i])
    return est, gt


FPS = 30
ani = FuncAnimation(fig, update, frames=len(timeseries), blit=True, interval=1000/FPS)
plt.show()