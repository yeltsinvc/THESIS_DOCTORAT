# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:10:53 2022

@author: valero
"""

import numpy as np
import pysocialforce as psf
from pysocialforce.utils.plot import SceneVisualizer
from pysocialforce.utils import Config, stateutils, logger


def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a − r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff

def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


OUTPUT_DIR = "images/"
n=10

pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])
pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])

x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
x_destination_left = 100.0 * np.ones((n, 1))
x_destination_right = -100.0 * np.ones((n, 1))

zeros = np.zeros((n, 1))




state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
state_right = np.concatenate(
    (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
)
initial_state = np.concatenate((state_left, state_right))
"""LSTM

"""
##calculate fd for LSTM
##[v_max,v_desired,delta_position_x,delta_position_y,velocity_x,velocity_y]
n_agents=len(initial_state)
v_max=1.8
v_desired=1.2
#pos=np.concatenate((pos_left,pos_right))
diff_to_goal=initial_state[:,4:6]-initial_state[:,:2]

fd=np.concatenate((v_max*np.ones((2*n,1)),v_desired*np.ones((2*n,1)),diff_to_goal),axis=-1)

##calculate fij for LSTM
##[v_max,v_desired,delta_position_x,delta_position_y,velocity_x,velocity_y]

diff=stateutils.each_diff(initial_state[:,:2])

vel_diff=stateutils.each_diff(initial_state[:,2:4])



obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)]
agent_colors = [(1, 0, 0)] * n + [(0, 0, 1)] * n
s = psf.Simulator(initial_state, obstacles=obstacles)
s.step(150)
with SceneVisualizer(s, OUTPUT_DIR + f"walkway_{n}", agent_colors=agent_colors) as sv:
    sv.ax.set_xlim(-30, 30)
    sv.ax.set_ylim(-20, 20)
    sv.animate()

