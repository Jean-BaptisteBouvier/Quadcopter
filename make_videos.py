# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:23:37 2025

@author: John Viljoen
    Jean-Baptiste Bouvier
    
MuJoCo quadcopter videos
"""


import os
import torch
import numpy as np
import mujoco as mj
import mujoco_viewer
import imageio
import matplotlib.pyplot as plt


import warnings
warnings.simplefilter("ignore", UserWarning)

# import torch
# import numpy as np

# from quadcopter import QuadcopterEnv, traj_comparison
# from utils import State_Normalizer, set_seed, norm, open_loop
# from utils import open_loop_stats, barplot_comparison
# from DiT import ODE, Planner, SA_ODE, SA_Planner
# from conditional_Action_DiT import Action_Conditional_ODE, Action_Conditional_Planner
# from model_based_ID import ModelBasedInverseDynamics

# from projectors import Reference_Projector, Admissible_Projector, SA_Projector, Action_Projector


# EVERYTHING FOR MUJOCO RENDERING

# def mj_get_state(data, omegas):
#     # generalised positions/velocities not in right coordinates
#     qpos = data.qpos.copy()
#     qvel = data.qvel.copy()
#     qpos[1] *= -1 # y
#     qpos[2] *= -1 # z
#     qpos[5] *= -1 # q2
#     qvel[1] *= -1 # ydot
#     qvel[2] *= -1 # zdot
#     qvel[4] *= -1 # qdot
#     return np.concatenate([qpos, qvel, omegas]).flatten()

def state2qpv(state):
    qpos = np.zeros(len(state))
    qpos = state.copy().squeeze()[0:7]
    qpos[1] *= -1 # y
    qpos[2] *= -1 # z
    qpos[5] *= -1 # q2
    qvel = state.copy().squeeze()[7:13]
    qvel[1] *= -1 # ydot
    qvel[2] *= -1 # zdot
    qvel[4] *= -1 # qdot
    return qpos, qvel

def set_state(state, model, data):
    # convert state to mujoco compatible
    qpos, qvel = state2qpv(state)
    # apply
    data.qpos = qpos
    data.qvel = qvel
    mj.mj_step(model, data)
    
    
def video_1quad(traj, name:str, fixed_cam = True):
    """Generate a video of the quadcopter"""
    if type(traj) == torch.Tensor:
        traj = traj.numpy()
    if len(traj.shape) == 3:
        traj = traj[0]
    if traj.shape[1] == 17:
        traj = traj[:, :13]
    
    dirname = os.path.dirname(__file__)
    if fixed_cam:
        xml_path='datasets/mujoco_quad_above.xml'
    else:
        xml_path='datasets/mujoco_quad_tracking.xml'
        
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    model = mj.MjModel.from_xml_path(xml_path, )  # MuJoCo model
    data = mj.MjData(model)
    # Make renderer, render and show the pixels
    framerate = 24  # (Hz)
    viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=1280, height=720)
    
    if fixed_cam:
        viewer.cam.distance = 5.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
        viewer.cam.lookat = np.array([3.8, 0., 0.])
    else: # tracking
        viewer.cam.elevation = -10
        viewer.cam.distance = 1
        old_x = -1
        old_y = 0
        old_azimuth = 0
        
    images = []
    mj.mj_resetData(model, data)  # Reset state and time
    
    for state in traj:    
        if not fixed_cam:
            x, y, z = state[:3]
            azimuth = -np.arctan2(y - old_y, x-old_x)*180/np.pi
            while azimuth > old_azimuth + 90 and azimuth > -360:
                azimuth -= 180
            while azimuth < old_azimuth - 90 and azimuth < 360:
                azimuth += 180
            viewer.cam.azimuth = azimuth
            viewer.cam.lookat = np.array([x, -y, -z])
            old_x, old_y, old_azimuth = x, y, azimuth
        
        set_state(state, model, data)
        images.append(viewer.read_pixels(camid=-1))
    video = np.stack(images)
    
    ### Timelapse # doesn't get the propellers since darker than the background
    # if fixed_cam:
    #     timelapse_image = np.max(video[range(0, video.shape[0], 5)], axis=0) # 1 every 10 frame
    #     plt.imsave('videos/timelapse_fig.png', timelapse_image)
    
    writer = imageio.get_writer(f'videos/{name}.mp4', fps=framerate)
    for i, img in enumerate(video):
        writer.append_data(img)
    writer.close() # Close the writer to finalize the video file
    

    
#%% Two quads

# Importing the PIL library
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont




def set_2states(state, shadow_state, model, data):
    # convert state to mujoco compatible
    qpos_1, qvel_1 = state2qpv(state)
    qpos_2, qvel_2 = state2qpv(shadow_state)
    # apply
    data.qpos = np.concatenate((qpos_1, qpos_2))
    data.qvel = np.concatenate((qvel_1, qvel_2))
    mj.mj_step(model, data)
    
    
    
    
    
def video_2quads(traj:np.ndarray, shadow_traj:np.ndarray, name:str, 
                 label_1:str = "traj", label_2:str = "shadow",
                  fixed_cam = True):
    """Generate a video of two quadcopter trajectories"""
    if type(traj) == torch.Tensor:
        traj = traj.numpy()
    if len(traj.shape) == 3:
        traj = traj[0]
    if traj.shape[1] == 17:
        traj = traj[:, :13]
        
    if type(shadow_traj) == torch.Tensor:
        shadow_traj = shadow_traj.numpy()
    if len(shadow_traj.shape) == 3 and shadow_traj.shape[-1] == 13:
        shadow_traj = shadow_traj[0]
    if shadow_traj.shape[1] == 17:
        shadow_traj = shadow_traj[:, :13]
    
    if fixed_cam:
        xml_path='datasets/mujoco_2quads_above.xml'
    else:
        xml_path='datasets/mujoco_2quads_tracking.xml'
    
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath
    model = mj.MjModel.from_xml_path(xml_path, )  # MuJoCo model
    data = mj.MjData(model)
    # Make renderer, render and show the pixels
    framerate = 24  # (Hz)
    viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen', width=1280, height=720)
    
    if fixed_cam:
        viewer.cam.distance = 5.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -90
        viewer.cam.lookat = np.array([3.8, 0., 0.])
    else: # tracking
        viewer.cam.elevation = -10
        viewer.cam.distance = 1
        old_x = -1
        old_y = 0
        old_azimuth = 0
        
    N1, N2 = traj.shape[0], shadow_traj.shape[0]
    m = traj.shape[1]
    assert shadow_traj.shape[1] == m, "State dimensions don't match"
    ### Repeat last state if a trajectory is longer than the other
    if N1 < N2:
        traj = np.concatenate((traj, np.ones((N2-N1, m))*traj[-1]), axis=0)
    elif N2 < N1:
        shadow_traj = np.concatenate((shadow_traj, np.ones((N1-N2, m))*shadow_traj[-1]), axis=0)    
        
    N = max(N1, N2)
    images = np.zeros((N, 720, 1280, 3))
    mj.mj_resetData(model, data)  # Reset state and time
    
    # Custom font style and font size
    myFont = ImageFont.truetype('datasets/palatinolinotype_roman.ttf', 40)
    
    for t in range(N):
        state = traj[t]
        shadow_state = shadow_traj[t]
        
        if not fixed_cam:
            x, y, z = state[:3]
            azimuth = -np.arctan2(y - old_y, x-old_x)*180/np.pi
            while azimuth > old_azimuth + 90 and azimuth > -360:
                azimuth -= 180
            while azimuth < old_azimuth - 90 and azimuth < 360:
                azimuth += 180
            viewer.cam.azimuth = azimuth
            viewer.cam.lookat = np.array([x, -y, -z])
            old_x, old_y, old_azimuth = x, y, azimuth
        
        set_2states(state, shadow_state, model, data)
        images[t] = viewer.read_pixels(camid=-1)
        
        # Call draw Method to add 2D graphics in an image
        img = Image.fromarray(images[t].astype(np.uint8))
        I1 = ImageDraw.Draw(img)
        I1.text((10, 10), label_1, font=myFont, fill=(255, 255, 255))
        I1.text((10, 50), label_2, font=myFont, fill=(255, 155, 0))
        # img.show()
        images[t] = np.array(img)
    
    writer = imageio.get_writer(f'videos/{name}.mp4', fps=framerate)
    for i, img in enumerate(images):
        writer.append_data(img.astype(np.uint8))
    writer.close() # Close the writer to finalize the video file
    






    
    

if __name__ == "__main__":    
    data = np.load("datasets/quad_1000trajs.npz")
    Trajs = data['trajs'][:, :, :13] # remove propeller states
    
    video_2quads(Trajs[0], Trajs[1], name="above_2quads", fixed_cam=True)
    # video_2quads(Trajs[0], Trajs[1], name="track", fixed_cam=False)
    print('fin')