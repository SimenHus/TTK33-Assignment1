from common import *

class EKF:
    Q = EKF_Q*np.eye(4) # Assumed discrete process noise
    R = EKF_R*np.eye(2) # Assumed discrete GNSS measurement noise


    @classmethod
    def prediction(clc, x: PDF, u: 'np.ndarray[2]') -> PDF:
        F = clc.F(x.mean, u)

        P_priori = F@x.sigma@F.T + clc.Q
        x_priori = clc.f(x.mean, u)

        return PDF(x_priori, P_priori)
    

    @classmethod
    def correction(clc, x: PDF, y: 'np.ndarray[2]') -> PDF:
        P_priori = x.sigma
        x_priori = x.mean

        g = clc.g(x_priori)
        G = clc.G(x_priori)
        K = P_priori@G.T@np.linalg.inv(G@P_priori@G.T + clc.R)

        P_posterior = (np.eye(4) - K@G)@P_priori
        x_posterior = x_priori + K@(y - g)

        return PDF(x_posterior, P_posterior)

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