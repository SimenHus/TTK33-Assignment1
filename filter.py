from common import *

class EKF:
    x: PDF
    Q: 'np.ndarray[4, 4]'
    R: 'np.ndarray[2, 2]'
    sigma_acc: 'np.ndarray[1]'
    sigma_gyro: 'np.ndarray[1]'
    sigma_GNSS: 'np.ndarray[2, 2]'
    label: str

    def __init__(self, x0: PDF, acc_factor: float, gyro_factor: float, GNSS_factor: float):
        sigma_acc = sigma_acc_base*acc_factor
        sigma_gyro = sigma_gyro_base*gyro_factor
        sigma_GNSS = sigma_GNSS_base*GNSS_factor

        self.sigma_acc = np.diag([1])*sigma_acc
        self.sigma_gyro = np.diag([1])*sigma_gyro
        self.sigma_GNSS = np.diag([1, 1])*sigma_GNSS

        self.Q = np.diag([1, 1, sigma_gyro**2, sigma_acc**2])*dt
        self.R = np.diag([sigma_GNSS**2, sigma_GNSS**2])*1/dt
        self.x = x0
        self.label = ('EKF:'
                    f'$\\sigma_{'{acc}'}$ = {acc_factor}$\\sigma_{"{acc\\_base}"}$'
                    f' $\\sigma_{'{gyro}'}$ = {gyro_factor}$\\sigma_{"{gyro\\_base}"}$'
                    f' $\\sigma_{'{GNSS}'}$ = {GNSS_factor}$\\sigma_{"{GNSS\\_base}"}$')


    def sample_IMU(self, true_acc: float, true_gyro: float) -> IMUData:
        lin_acc = PDF(np.array([true_acc]), self.sigma_acc).sample()
        ang_vel = PDF(np.array([true_gyro]), self.sigma_gyro).sample()
        return IMUData(lin_acc, ang_vel)

    def sample_GNSS(self, true_pos: 'np.ndarray[2]') -> GNSSData:
        pos = PDF(true_pos, self.sigma_GNSS).sample()
        return GNSSData(pos)

    def prediction(self, u: 'np.ndarray[2]') -> None:

        F = self.F(self.x.mean, u)

        P_prior = F@self.x.sigma@F.T + self.Q
        x_prior = self.f(self.x.mean, u)

        self.x = PDF(x_prior, P_prior)
    

    def correction(self, y: 'np.ndarray[2]') -> None:
        P_prior = self.x.sigma
        x_prior = self.x.mean

        g = self.g(x_prior)
        G = self.G(x_prior)
        K = P_prior@G.T@np.linalg.inv(G@P_prior@G.T + self.R)

        P_posterior = (np.eye(4) - K@G)@P_prior
        x_posterior = x_prior + K@(y - g)

        self.x = PDF(x_posterior, P_posterior)

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

    @staticmethod
    def F(x: 'np.ndarray[4]', u: 'np.ndarray[2]') -> 'np.ndarray[4, 4]':
        x_pos, y_pos, vel, theta = x
        
        F = np.array([
            [0, 0, np.cos(theta), -np.sin(theta)*vel],
            [0, 0, np.sin(theta), np.cos(theta)*vel],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        return F
    

    @staticmethod
    def g(x: 'np.ndarray[4]') -> 'np.ndarray[2]':
        C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return C@x
    
    @staticmethod
    def G(x: 'np.ndarray[4]') -> 'np.ndarray[2, 4]':
        G = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return G
    
    


class IEKF(EKF):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'I' + self.label

    def correction(self, y: 'np.ndarray[2]') -> None:
        P_prior = self.x.sigma
        x_prior = self.x.mean

        x_iter = x_prior
        threshold = 1e-2
        max_iteratons = 100

        for _ in range(max_iteratons):
            g = self.g(x_iter)
            G = self.G(x_iter)
            K = P_prior@G.T@np.linalg.inv(G@P_prior@G.T + self.R)

            x_new = x_iter + K@(y - g)

            if np.linalg.norm(x_new - x_iter) < threshold: break

            x_iter = x_new

        P_posterior = (np.eye(4) - K@G)@P_prior
        x_posterior = x_iter

        self.x = PDF(x_posterior, P_posterior)