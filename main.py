from common import *
from Robot import DifferentialDrive
from Filter import EKF, IEKF
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


# Simulation variables
simulation_length = 20 # Simulation time [s]
timeseries = np.arange(0, simulation_length, dt) # Timeseries

acc_b = np.ones([len(timeseries)])*0.3 # [m/s^2] Body frame acceleration of the vehicle in simulation
w_b = np.ones([len(timeseries)])*1 # [rad/s] Body frame angular velocity of vehicle in simulation
x_0 = np.array([0, 0, 0, 0]) # (x, y, v, heading) Initial value  (modified to utilize acceleration as input instead of velocity)
cov_0 = 0*np.eye(len(x_0)) # Initialize covariance

n_states = len(x_0) # Number of states in use

x0 = PDF(x_0, cov_0) # Initial PDF

# List of EKFs to run in the simulation
EKFS: list[EKF] = [
    EKF(x0, 0.1, 0.1, 1),
    EKF(x0, 1, 1, 1),
    EKF(x0, 10, 10, 1),
    EKF(x0, 10, 0.1, 1),
    IEKF(x0, 0.1, 0.1, 1),
    IEKF(x0, 1, 1, 1),
    IEKF(x0, 10, 10, 1),
    IEKF(x0, 10, 0.1, 1),
]


trajectory_true = np.zeros([n_states, len(timeseries)]) # Variable to store true trajectory of the vehicle
trajectory_measured = np.zeros([n_states*len(EKFS), len(timeseries)]) # Variable to store EKF measurement of the vehicle trajectory
trajectory_true[:, 0] = x_0


last_GNSS = 0 # Variable to keep track of when next GNSS sample should arrive
for i, t in enumerate(timeseries):
    
    pos_meas = t - last_GNSS >= 1
    if pos_meas: last_GNSS = t

    for j, filter in enumerate(EKFS):
        IMU_meas = filter.sample_IMU(acc_b[i], w_b[i]) # Simulation is run at 100Hz and IMU is sampled every iteration
        filter.prediction(IMU_meas.array) # Predict state
        if pos_meas:
            pos = filter.sample_GNSS(trajectory_true[:2, i])
            filter.correction(pos.array)
        
        trajectory_measured[n_states*j:n_states*(j+1), i] = filter.x.mean
    

    if i == len(timeseries)-1: break # Break 1 iteration early to prevent error with indexing
    trajectory_true[:, i+1] = DifferentialDrive.f(trajectory_true[:, i], np.array([acc_b[i], w_b[i]]))


n_columns = 4
n_rows = -(len(EKFS) // -n_columns) # Ceil division
fig, axs = plt.subplots(n_rows, n_columns)
lim = 10

lines = []

counter = 0 # Counter to have the same number of plots as EKF cases
for row in axs:
    for column in row:
        if counter == len(EKFS): break
        lines.append(column.plot([], [], label='Estimated')[0])
        lines.append(column.plot([], [], label='Ground truth')[0])
        column.set_xlim([-lim, lim])
        column.set_ylim([-lim, lim])
        column.grid()
        column.legend()
        column.set_title(EKFS[counter].label, rotation='horizontal', x=-0.1, y=0.5)
        counter += 1

def update(i):
    for j in range(len(EKFS)):
        lines[j*2].set_data(trajectory_measured[n_states*j, :i], trajectory_measured[n_states*j+1, :i])
        lines[j*2+1].set_data(trajectory_true[0, :i], trajectory_true[1, :i])

    return lines


FPS = 120
ani = FuncAnimation(fig, update, frames=len(timeseries), blit=True, interval=1000/FPS)
plt.show()