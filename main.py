from common import *
from Sensor import IMU, GNSS
from Robot import DifferentialDrive
from Filter import EKF
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


simulation_length = 20 # Simulation time [s]
timeseries = np.arange(0, simulation_length, dt) # Timeseries

x_0 = np.array([0, 0, 0, 0]) # (x, y, v, heading) Initial value  (modified to utilize acceleration as input instead of velocity)
cov_0 = 10e6*np.eye(len(x_0)) # Initialize covariance

acc_b = np.ones([len(timeseries)])*0.3 # [m/s^2] Body frame acceleration of the vehicle in simulation
w_b = np.ones([len(timeseries)])*1 # [rad/s] Body frame angular velocity of vehicle in simulation
trajectory_true = np.zeros([len(x_0), len(timeseries)]) # Variable to store true trajectory of the vehicle
trajectory_measured = np.zeros([len(x_0), len(timeseries)]) # Variable to store EKF measurement of the vehicle trajectory



trajectory_true[:, 0] = x_0
last_GNSS = 0 # Variable to keep track of when next GNSS sample should arrive
x = PDF(x_0, cov_0)
for i, t in enumerate(timeseries):
    
    last_IMU = t
    IMU_meas = IMU.sample(acc_b[i], w_b[i]) # Simulation is run at 100Hz and IMU is sampled every iteration
    x = EKF.prediction(x, IMU_meas.array) # Predict state

    if t - last_GNSS >= 1/GNSS.sample_rate: # If GNSS sample, correct with measurement
        pos_meas = GNSS.sample(trajectory_true[:2, i])
        x = EKF.correction(x, pos_meas.array)
    
    trajectory_measured[:, i] = x.mean
    

    if i == len(timeseries)-1: break # Break 1 iteration early to prevent error with indexing
    trajectory_true[:, i+1] = DifferentialDrive.f(trajectory_true[:, i], np.array([acc_b[i], w_b[i]]))


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


FPS = 120
ani = FuncAnimation(fig, update, frames=len(timeseries), blit=True, interval=1000/FPS)
plt.show()