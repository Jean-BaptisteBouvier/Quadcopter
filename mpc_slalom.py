# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:53:43 2024

@author: Jean-Baptiste Bouvier from John Viljoen's code

MPC controller to realize a slalom between two cylinder obstacles
"""

import casadi as ca
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class MPC:
    def __init__(self, env, N: int) -> None:
        """ env: quadcopter environment    N: horizon length"""

        n = env.state_size # 17
        m = env.action_size # 4
        
        # create optimizer container and define its optimization variables
        self.opti = ca.Opti()
        # opti variables are optimized variables, we can only change their initial pre-optimized value at runtime
        self.X = self.opti.variable(n, N+1) # first state plays no role
        self.U = self.opti.variable(m, N+1) # final input plays no role

        ### Sequence of states to track
        self.s1 = np.array([3.5, 0, 0, 1,0,0,0,0,0,0,0,0,0,0,0,0,0]) # intermediary goal state
        self.s2 = np.array([7,   0, 0, 1,0,0,0,0,0,0,0,0,0,0,0,0,0]) # final goal state
        #           {x,y, z,q0,q1,q2,q3,xd,yd,zd,p,q,r,wM1,wM2,wM3,wM4}
        Q = np.diag([1,1, 1, 1, 1, 1, 1, 0, 0, 0, 0,0,0,0,  0,  0,  0  ])
        
        #           {wM1d, wM2d, wM3d, wM4d}
        R = np.diag([10,   10,   10,   10])

        # opti parameters are non-optimized variables that we can change at runtime
        self.init = self.opti.parameter(n,1)  # initial condition 
        self.ref = self.opti.parameter(n,1)   # tracking state
        self.opti.subject_to(self.X[:,0] == self.init) # apply initial condition constraints

        # apply dynamics constraints with euler integrator
        for k in range(N):
            self.opti.subject_to(self.X[:,k+1] == env.casadi_step(self.X[:,k], self.U[:,k]) )

        # apply state constraints
        for k in range(N+1):
            self.opti.subject_to(self.X[:,k] < env.state_ub)
            self.opti.subject_to(self.X[:,k] > env.state_lb)

        # define input constraints
        self.opti.subject_to(self.opti.bounded(-100, self.U, 100))
        
        # Multiple cylinder constraints
        self.radii = env.cylinder_radii # [0.9, 0.9] # radius
        self.xc = env.cylinder_xc # [2.5, 5.2] # cylinder x center position
        self.yc = env.cylinder_yc # [0.5, -0.5] # cylinder y center position
        N_cylinders = len(self.radii)
        
        for k in range(N):
            current_time = k*env.dt
            multiplier = 1 + current_time * 0.1
            current_x, current_y = self.X[0,k+1], self.X[1,k+1]
            for i in range(N_cylinders):
                outside = self.radii[i] ** 2 * multiplier <= (current_x - self.xc[i])**2 + (current_y - self.yc[i])**2
                self.opti.subject_to( outside )


        # define the cost function - can you parameterize the reference to be a ca.MX so that we can change our desired destination at runtime?
        def J(state, ctrl):
            state_error = self.ref - state
            cost = ca.MX(0)
            # lets get cost per timestep:
            for k in range(N+1):
                timestep_input = ctrl[:,k]
                timestep_state_error = state_error[:,k]
                cost += (timestep_state_error.T @ Q @ timestep_state_error + timestep_input.T @ R @ timestep_input)
            return cost
        
        # apply the static reference cost to the opti container
        self.opti.set_value(self.ref, self.s1)
        self.checkpoint = 0
        self.opti.minimize(J(self.X, self.U))

        # tell the opti container we want to use IPOPT to optimize, and define settings for the solver
        opts = {
            'ipopt.print_level':0, 
            'print_time':0,
            'ipopt.tol': 1e-6,
        } # silence!
        self.opti.solver('ipopt', opts)

        # lets do a solve in the __init__ function to test functionality - the only thing that we have parameterized to change at runtime is the initial condition
        self.opti.set_value(self.init, env.hover_state)
        
        # perform the solve
        sol = self.opti.solve()

        # extract the answer and save to an attribute we will later use to warm start the optimization variables for efficiency
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)
        

    def __call__(self, x):

        if self.checkpoint == 0 and x[0] > 3:
            self.checkpoint += 1
            self.opti.set_value(self.ref, self.s2)

        # assign the new initial condition to the runtime changeable parameter
        self.opti.set_value(self.init, x)

        # warm starting based off of previous solution
        old_x_sol = self.x_sol[:,2:] # ignore old start and first step (this step start)
        x_warm_start = np.hstack([old_x_sol, old_x_sol[:,-1:]]) # stack final solution onto the end again for next warm start
        old_u_sol = self.u_sol[:,1:] # ignore previous solution
        u_warm_start = np.hstack([old_u_sol, old_u_sol[:,-1:]]) # stack final u solution onto the end again for next warm start

        self.opti.set_initial(self.X[:,1:], x_warm_start)
        self.opti.set_initial(self.U[:,:], u_warm_start) 

        # perform the solve
        sol = self.opti.solve()

        # extract the answer
        self.x_sol, self.u_sol = sol.value(self.X), sol.value(self.U)
        
        # return first input to be used
        return self.u_sol[:,0]
    
    def get_predictions(self):
        return self.opti.value(self.X), self.opti.value(self.U)

    
        
    
if __name__ == "__main__":

    from utils.quad import Animator
    from quad_env import QuadcopterEnv, nice_plot
    import matplotlib.pyplot as plt
    import matplotlib.patches as pat    

    Ti, Tf, Ts = 0.0, 2.0, 0.01
    N = 60 # number of lookahead steps in MPC

    env = QuadcopterEnv(dt = Ts, reset_noise=0)#1e-1)
    
    s = env.reset()
    
    mpc = MPC(env, N=N)

    ctrl_pred_x = []
    memory = {'state': [s], 'cmd': [np.zeros(4)]}
    true_times = np.arange(Ti, Tf, Ts)
    for t in tqdm(true_times):

        try:
            u = mpc(s)
        except:
            print("\nFailed")
            break
        s = env.step(u)[0]

        ctrl_predictions = mpc.get_predictions()
        ctrl_pred_x.append(ctrl_predictions[0])

        memory['state'].append(np.copy(s))
        memory['cmd'].append(u)

    memory['state'] = np.vstack(memory['state'])
    memory['cmd'] = np.vstack(memory['cmd'])
    ctrl_pred_x = np.stack(ctrl_pred_x)

    T = memory['state'].shape[0]
    
    
    #%% Plotting x, y, z
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2, length=3)
    
    plt.plot(np.arange(T), memory['state'][:,0], label="x", linewidth=3)
    plt.plot(np.arange(T), memory['state'][:,1], label="y", linewidth=3)
    plt.plot(np.arange(T), memory['state'][:,2], label="z", linewidth=3)
    plt.xlabel("timesteps")
    plt.xticks([0, 100, 200])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7])
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9)
    plt.title("Quadcopter position")
    plt.savefig("data/Quad_xyz.pdf", dpi=1200, bbox_inches="tight")
    plt.show()
    
    
    #%% Plotting rotor angular speed
    
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2, length=3)
    
    for rotor_id in range(1,5):
        plt.plot(np.arange(T), memory['state'][:, -rotor_id], label=f"rotor {rotor_id}", linewidth=3)
    plt.xlabel("timesteps")
    plt.xticks([0, 100, 200])
    plt.ylabel("angular velocity (rad/s)")
    plt.legend(frameon=False, labelspacing=0.1, handletextpad=0.2, handlelength=0.9,
               bbox_to_anchor=(0.05, 0.0), loc='lower left')
    plt.title("Propellers angular velocity")
    plt.savefig("data/Quad_propellers.pdf", dpi=1200, bbox_inches="tight")
    plt.show()
    
    #%% Calculate orientation
    rot = np.zeros((T, 3))
    quat_scalar_last = np.concatenate((memory['state'][:, 4:7], memory['state'][:, 3].reshape((T,1))), axis=1)
   
    for i in range(T):
        rot[i] = R.from_quat(quat_scalar_last[i]).as_euler('xyz', degrees=True)
        if i > 1:
            if abs(rot[i, 0] - rot[i-1, 0]) > 90:
                rot[i, 0] = rot[i, 0] - np.sign(rot[i, 0])*180
            if abs(rot[i, 1] - rot[i-1, 1]) > 90:
                rot[i, 1] = rot[i, 1] - np.sign(rot[i, 1])*180
            if abs(rot[i, 2] - rot[i-1, 2]) > 90:
                rot[i, 2] = rot[i, 2] - np.sign(rot[i, 2])*180
      
    #%% Plot orientation 
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2, length=3)
    
    plt.plot(np.arange(T), rot[:, 0], label="roll x", linewidth=3)
    plt.plot(np.arange(T), rot[:, 1], label="pitch y", linewidth=3)
    plt.plot(np.arange(T), rot[:, 2], label="yaw z", linewidth=3)
    plt.yticks([-180, -90, 0, 90, 180], ["-180°", "-90°", "0°", "90°", "180°"])
    plt.xlabel("timesteps")
    plt.xticks([0, 100, 200])
    plt.ylabel("orientation (deg)")
    plt.legend(frameon=False, labelspacing=0.1, handletextpad=0.2, handlelength=0.9)
    plt.title("Quadcopter orientation")
    plt.savefig("data/Quad_orientation.pdf", dpi=1200, bbox_inches="tight")
    plt.show()
    
    
    #%% Plot slalom
    plt.figure(figsize=(6, 3))
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(width=2, length=3)
    ax.axis("equal")
    for i in range(len(mpc.radii)):
        cylinder = pat.Circle(xy=[mpc.xc[i], mpc.yc[i]], radius=mpc.radii[i], color="red")
        ax.add_patch(cylinder)
    plt.plot(memory['state'][:,0], memory['state'][:,1], linewidth=3)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([-1.5, 1.5])
    # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["0", "", "", "3", "", "", "", "7"])
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ["0", "1", "2", "3", "4", "5", "6", "7"])
    plt.title("Quadcopter trajectory")
    plt.savefig("data/Quad_slalom.pdf", dpi=1200, bbox_inches="tight")
    plt.show()
    
    # animator = Animator(memory['state'], true_times, memory['state'], max_frames=500, save_path='data', state_prediction=ctrl_pred_x, drawCylinder=True)
    # animator.animate()

    print('fin')

