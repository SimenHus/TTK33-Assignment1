Problem:

Implementation of the Extended Kalman Filter (EKF) for a Multiple Input, Multiple Output nonlinear system.
An example can be a differential drive vehicle. The equattions for this dynamic model can be found (for example) following this link 
[look under "Differential Drive Kinematics/Variable Speed and Heading Rate Equation (Generalized)].
For this robot, you will assume noise-injected (but bias-free) angular rate and linear acceleration measurements alongside
noise-injected (but bias-free) position measurements. Angular rate and linear acceleration measurements should be at a high update rate,
namely 100Hz. Position measurements should be at a low update rate, namely 1Hz. Implement the EKF and present the effect of increasing noise
in angular rate and linear accceleration as well as in position measurements. 
Alternatively, consider the full 3D (6 Degrees of Freedom) model of a quadrotor vehicle. Note this is more involved so pick this only if it
reflects your interest.
Implementation of the Iterated EKF for the same problem as the EKF.
