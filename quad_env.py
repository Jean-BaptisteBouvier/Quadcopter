# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:09:01 2024

@author: Jean-Baptiste

Quadcopter environment modified from John Viljoen's
https://github.com/johnviljoen/231A_project
"""
import casadi as ca # for the MPC
import numpy as np


class QuadcopterEnv():
    """
    This environment describes a fully nonlinear quadcopter

    ## Action Space
    | Num | Action                 | Min | Max | Name | Unit |
    | --- | ---------------------- | --- | --- | ---- | ---- |
    | 0   | Torque on first rotor  | -1  |  1  | w0d  |  N   |
    | 1   | Torque on second rotor | -1  |  1  | w1d  |  N   |
    | 2   | Torque on third rotor  | -1  |  1  | w2d  |  N   |
    | 3   | Torque on fourth rotor | -1  |  1  | w3d  |  N   |

    ## Observation Space
    | Num | Observation                            | Min  | Max | Name | Unit  |
    | --- | -------------------------------------- | ---- | --- | ---- | ----- |
    | 0   | x-coordinate of the center of mass     | -Inf | Inf |  x   |  m    |
    | 1   | y-coordinate of the center of mass     | -Inf | Inf |  y   |  m    |
    | 2   | z-coordinate of the center of mass     | -Inf | Inf |  z   |  m    |
    | 3   | w-orientaiont of the body (quaternion) | -Inf | Inf |  q0  |  rad  |
    | 4   | x-orientaiont of the body (quaternion) | -Inf | Inf |  q1  |  rad  |
    | 5   | y-orientaiont of the body (quaternion) | -Inf | Inf |  q2  |  rad  |
    | 6   | z-orientaiont of the body (quaternion) | -Inf | Inf |  q3  |  rad  |
    
    | 7   | x-velocity of the center of mass       | -Inf | Inf |  xd  |  m/s  |
    | 8   | y-velocity of the center of mass       | -Inf | Inf |  yd  |  m/s  |
    | 9   | z-velocity of the center of mass       | -Inf | Inf |  zd  |  m/s  |
    | 10  | x-angular velocity of the body         | -Inf | Inf |  p   | rad/s |
    | 11  | y-angular velocity of the body         | -Inf | Inf |  q   | rad/s |
    | 12  | z-angular velocity of the body         | -Inf | Inf |  r   | rad/s |
    | 13  | angular velocity of the first rotor    | -Inf | Inf |  w0  | rad/s |
    | 14  | angular velocity of the second rotor   | -Inf | Inf |  w1  | rad/s |
    | 15  | angular velocity of the third rotor    | -Inf | Inf |  w2  | rad/s |
    | 16  | angular velocity of the fourth rotor   | -Inf | Inf |  w3  | rad/s |

    
    ## Starting State
    All observations start from hover with a Gaussian noise of magnitude `reset_noise_scale'
    
    ## Episode End
    1. Any of the states goes out of bounds
    2. The Quadcopter collides with one of the cylinder obstacles

    NOTES:
    John integrated the proportional control of the rotors directly into the 
    equations of motion to more accurately reflect the closed loop system
    we will be controlling with a second outer loop. This inner loop is akin
    to the ESC which will be onboard many quadcopters which directly controls
    the rotor speeds to be what is commanded.
    """
    
    
    def __init__(self, reset_noise_scale:float = 1e-2, dt: float = 0.01,
                 cylinder_radii = [0.7, 0.7]
        
        
        self.name = "Quadcopter"
        self.state_size = 17
        self.action_size = 4
        self.action_min = np.array([[-1., -1., -1., -1.]])
        self.action_max = np.array([[ 1.,  1.,  1.,  1.]])
        self.position_states = [0,1,2,3,4,5,6]
        self.velocity_states = [7,8,9,10,11,12,13,14,15,16]
        
        ### Mujoco Setup
        self.model = mj.MjModel.from_xml_path("./assets/mujoco_quad_above.xml")
        self.data = mj.MjData(self.model)
        self.viewer = vi.launch_passive(self.model, self.data)
        self.renderer = mj.Renderer(self.model, height=480, width=640) #renderer


        ### Obstacles: cylinders along the z-axis
        self.N_cylinders = 2 # 2 cylinders
        self.cylinder_radii = cylinder_radii # radius of the cylinders
        self.cylinder_xc = [2.5, 5.2] # cylinders x center position
        self.cylinder_yc = [0.5, -0.5] # cylinders y center position
        
        
        ### fundamental quad parameters
        self.g = 9.81 # gravity (m/s^2)
        self.mB = 1.2 # mass (kg)
        self.dxm = 0.16 # arm length (m)
        self.dym =  0.16 # arm length (m)
        self.dzm = 0.01  # arm height (m)
        self.IB = np.array([[0.0123, 0,      0     ],
                            [0,      0.0123, 0     ],
                            [0,      0,      0.0224]])  # Inertial tensor (kg*m^2)
        self.IRzz = 2.7e-5  # rotor moment of inertia (kg*m^2)
        self.Cd = 0.1  # drag coefficient (omnidirectional)
        self.kTh = 1.076e-5  # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
        self.kTo = 1.632e-7  # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
        self.minThr = 0.1*4  # Minimum total thrust (N)
        self.maxThr = 9.18*4  # Maximum total thrust (N)
        self.minWmotor = 75  # Minimum motor rotation speed (rad/s)
        self.maxWmotor = 925  # Maximum motor rotation speed (rad/s)
        self.tau = 0.015  # Value for second order system for Motor dynamics
        self.kp = 1.0  # Value for second order system for Motor dynamics
        self.damp = 1.0  # Value for second order system for Motor dynamics
        self.usePrecession = True  # model precession or not
        self.w_hover = 522.9847140714692 # hardcoded hover rotor speed (rad/s)
        
        ### post init useful parameters for quad
        self.B0 = np.array([[self.kTh, self.kTh, self.kTh, self.kTh],
                            [self.dym * self.kTh, - self.dym * self.kTh, -self.dym * self.kTh,  self.dym * self.kTh],
                            [self.dxm * self.kTh,  self.dxm * self.kTh, -self.dxm * self.kTh, -self.dxm * self.kTh],
                            [-self.kTo, self.kTo, - self.kTo, self.kTo]]) # actuation matrix

        self.x_lb = np.array([-10, -10, -10, *[-np.inf]*4, *[-10]*3, *[-10]*3, *[self.minWmotor]*4])
                               # xyz       q0123         xdydzd    pdqdrd    w0123
        
        self.x_ub = np.array([10, 10, 10, *[np.inf]*4, *[10]*3, *[10]*3, *[self.maxWmotor]*4])
                                # xyz      q0123        xdydzd   pdqdrd   w0123
        
        self.dt = dt # time step
        self.reset_noise = reset_noise
        self.hover_state = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, *[522.9847140714692]*4])
        self.position_states = [0,1,2,3,4,5,6]
        self.velocity_states = [7,8,9,10,11,12,13,14,15,16]
        
        self.state_ub = np.array([30, 30, 30,
                                  2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi,
                                  10, 10, 10,
                                  50, 50, 50, 
                                  self.maxWmotor, self.maxWmotor, self.maxWmotor, self.maxWmotor])

        self.state_lb = np.array([-30, -30, -30,
                                  -2*np.pi, -2*np.pi, -2*np.pi, -2*np.pi,
                                  -10, -10, -10, 
                                  -50, -50, -50,
                                  self.minWmotor, self.minWmotor, self.minWmotor, self.minWmotor])

        
        
    def reset(self):
        self.state = self.hover_state.copy() + self.reset_noise*np.random.randn(self.state_size)
        return self.state.copy()
        
    
    def reset_to(self, state):
        assert all(state > self.x_lb) and all(state < self.x_ub), "state is out of bounds x_lb, x_ub"
        self.state = state.copy()
        return state


    def is_in_obstacle(self, x, y):
        """Checks whether (x,y) position is inside an obstacle"""
        for i in range(self.N_cylinders):
            inside = self.cylinder_radii[i]**2 >= (x - self.cylinder_xc[i])**2 + (y - self.cylinder_yc[i])**2 
            if inside:
                return True
        return False
    

    def step(self, u):
        """Numpy"""
        
        q0 =    self.state[3]
        q1 =    self.state[4]
        q2 =    self.state[5]
        q3 =    self.state[6]
        xdot =  self.state[7]
        ydot =  self.state[8]
        zdot =  self.state[9]
        p =     self.state[10]
        q =     self.state[11]
        r =     self.state[12]
        wM1 =   self.state[13]
        wM2 =   self.state[14]
        wM3 =   self.state[15]
        wM4 =   self.state[16]
    
        # instantaneous thrusts and torques generated by the current w0...w3
        wMotor = np.stack([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, self.minWmotor, self.maxWmotor) # this clip shouldn't occur within the dynamics
        th = self.kTh * wMotor ** 2 # thrust
        to = self.kTo * wMotor ** 2 # torque
    
        # state derivates (from sympy.mechanics derivation)
        xd = np.stack(
            [
                xdot,
                ydot,
                zdot,
                -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                (self.Cd * np.sign(-xdot) * xdot**2
                    - 2 * (q0 * q2 + q1 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # xdd
                (
                     self.Cd * np.sign(-ydot) * ydot**2
                    + 2 * (q0 * q1 - q2 * q3) * (th[0] + th[1] + th[2] + th[3])
                )
                /  self.mB, # ydd
                (
                    - self.Cd * np.sign(zdot) * zdot**2
                    - (th[0] + th[1] + th[2] + th[3])
                    * (q0**2 - q1**2 - q2**2 + q3**2)
                    + self.g *  self.mB
                )
                /  self.mB, # zdd (the - in front turns increased height to be positive - SWU)
                (
                    ( self.IB[1,1] -  self.IB[2,2]) * q * r
                    -  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                    + (th[0] - th[1] - th[2] + th[3]) *  self.dym
                )
                /  self.IB[0,0], # pd
                (
                    ( self.IB[2,2] -  self.IB[0,0]) * p * r
                    +  self.usePrecession *  self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                    + (th[0] + th[1] - th[2] - th[3]) *  self.dxm
                )
                /  self.IB[1,1], #qd
                (( self.IB[0,0] -  self.IB[1,1]) * p * q - to[0] + to[1] - to[2] + to[3]) /  self.IB[2,2], # rd
                u[0]/self.IRzz, u[1]/self.IRzz, u[2]/self.IRzz, u[3]/self.IRzz # w0d ... w3d
            ]
        )
    
        self.state += xd * self.dt # one time step forward
        # Clip the rotor speeds within limits
        self.state[13:17] = np.clip(self.state[13:17], self.x_lb[13:17], self.x_ub[13:17])
        
        out_of_bound = any(self.state < self.x_lb) or any(self.state > self.x_ub) # out of bound state
        collided = self.is_in_obstacle(self.state[0], self.state[1])
        reward = 1 - out_of_bound - collided
        terminated = out_of_bound or collided
    
        return self.state.copy(), reward, terminated, False, None


    def casadi_step(self, state: ca.MX, cmd: ca.MX):
         """Returns next state of the Casadi state"""

         # Import params to numpy for CasADI
         # ---------------------------
         IB = self.IB
         IBxx = IB[0, 0]
         IByy = IB[1, 1]
         IBzz = IB[2, 2]

         # Unpack state tensor for readability
         # ---------------------------
         q0 =    state[3]
         q1 =    state[4]
         q2 =    state[5]
         q3 =    state[6]
         xdot =  state[7]
         ydot =  state[8]
         zdot =  state[9]
         p =     state[10]
         q =     state[11]
         r =     state[12]
         wM1 =   state[13]
         wM2 =   state[14]
         wM3 =   state[15]
         wM4 =   state[16]

         # a tiny bit more readable
         ThrM1 = self.kTh * wM1 ** 2
         ThrM2 = self.kTh * wM2 ** 2
         ThrM3 = self.kTh * wM3 ** 2
         ThrM4 = self.kTh * wM4 ** 2
         TorM1 = self.kTo * wM1 ** 2
         TorM2 = self.kTo * wM2 ** 2
         TorM3 = self.kTo * wM3 ** 2
         TorM4 = self.kTo * wM4 ** 2

         # Wind Model (zero in expectation)
         # ---------------------------
         velW, qW1, qW2 = 0, 0, 0

         # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
         # ---------------------------
         DynamicsDot = ca.vertcat(
                 xdot,
                 ydot,
                 zdot,
                 -0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r,
                 0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r,
                 0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r,
                 -0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r,
                 (
                     self.Cd
                     * ca.sign(velW * ca.cos(qW1) * ca.cos(qW2) - xdot)
                     * (velW * ca.cos(qW1) * ca.cos(qW2) - xdot) ** 2
                     - 2 * (q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                 )
                 / self.mB,
                 (
                     self.Cd
                     * ca.sign(velW * ca.sin(qW1) * ca.cos(qW2) - ydot)
                     * (velW * ca.sin(qW1) * ca.cos(qW2) - ydot) ** 2
                     + 2 * (q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                 )
                 / self.mB,
                 (
                     -self.Cd * ca.sign(velW * ca.sin(qW2) + zdot) * (velW * ca.sin(qW2) + zdot) ** 2
                     - (ThrM1 + ThrM2 + ThrM3 + ThrM4)
                     * (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2)
                     + self.g * self.mB
                 )
                 / self.mB,
                 (
                     (IByy - IBzz) * q * r
                     - self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * q
                     + (ThrM1 - ThrM2 - ThrM3 + ThrM4) * self.dym
                 )
                 / IBxx,  # uP activates or deactivates the use of gyroscopic precession.
                 (
                     (IBzz - IBxx) * p * r
                     + self.usePrecession * self.IRzz * (wM1 - wM2 + wM3 - wM4) * p
                     + (ThrM1 + ThrM2 - ThrM3 - ThrM4) * self.dxm
                 )
                 / IByy,  # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
                 ((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz,
                 cmd[0]/self.IRzz, cmd[1]/self.IRzz, cmd[2]/self.IRzz, cmd[3]/self.IRzz
         )

         if DynamicsDot.shape[1] == 17:
             print('fin')

         # State Derivative Vector
         next_state = state + DynamicsDot * self.dt
         return next_state


#%% Figures and animations

import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    """   Convert a quaternion into a rotation matrix.   """
    q0, q1, q2, q3 = q
    R = np.array([
        [1 - 2*(q2**2 + q3**2),     2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [    2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2),     2*(q2*q3 - q0*q1)],
        [    2*(q1*q3 - q0*q2),     2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    return R


def plot_traj(env, Traj, title=None):
    """Plot the important components of a given quadcopter trajectory"""
    
    assert len(Traj.shape) == 2, "Trajectory must be a 2D array"
    assert Traj.shape[1] == env.state_size, "Trajectory must contain the full state"
    T = Traj.shape[0]
    time = env.dt * np.arange(T)
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.plot(time, Traj[:,0], label="x", linewidth=3)
    plt.plot(time, Traj[:,1], label="y", linewidth=3)
    plt.plot(time, Traj[:,2], label="z", linewidth=3)
    plt.xlabel("t (s)")
    plt.ylabel("orientation (deg)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    for rotor_id in range(1,5):
        plt.plot(time, Traj[:, -rotor_id], label=f"rotor {rotor_id}", linewidth=3)
    plt.xlabel("t (s)")
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()
    
    rot = np.zeros((T, 3))
    quat_scalar_last = np.concatenate((Traj[:, 4:7],Traj[:, 3].reshape((T,1))), axis=1)
   
    for i in range(T):
        rot[i] = R.from_quat(quat_scalar_last[i]).as_euler('xyz', degrees=True)
        if i > 1:
            for angle in range(3):
                while rot[i, angle] > rot[i-1, angle] + 180:
                    rot[i, angle] -= 90*2
                while rot[i, angle] < rot[i-1, angle] - 180:
                    rot[i, angle] += 90*2
    
    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)        
    plt.plot(time, rot[:, 0], label="roll x", linewidth=3)
    plt.plot(time, rot[:, 1], label="pitch y", linewidth=3)
    plt.plot(time, rot[:, 2], label="yaw z", linewidth=3)
    plt.xlabel("t (s)")
    plt.yticks([-180, -90, 0, 90, 180])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.show()


def plot_xy_path(env, Traj):

    radii = env.cylinder_radii # [0.9, 0.9] # radius
    xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
    yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position

    fig, ax = nice_plot()
    plt.axis("equal")
    for i in range(len(radii)):
        cylinder = pat.Circle(xy=[xc[i], yc[i]], radius=radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(Traj[:,0], Traj[:,1], linewidth=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    saveas: str = None, legend_loc='best'):
    
    """Compares given quadcopter trajectories.
    Optional argument 'saveas' takes the filename to save the plots if desired"""
    
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"
    
    radii = env.cylinder_radii # [0.9, 0.9] # radius
    xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
    yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position

    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.axis("equal")
    for i in range(len(radii)):
        cylinder = pat.Circle(xy=[xc[i], yc[i]], radius=radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(traj_1[:,0], traj_1[:,1], label=label_1, linewidth=3)
    plt.plot(traj_2[:,0], traj_2[:,1], label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(traj_3[:,0], traj_3[:,1], label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(traj_4[:,0], traj_4[:,1], label=label_4, linewidth=3)
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.xlabel("x")
    plt.ylabel("y")
    if saveas is not None:
        plt.savefig(saveas + "_trajs.svg", bbox_inches='tight', format="svg", dpi=1200)
    plt.show()
    
   

def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax







