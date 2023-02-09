#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:47:49 2021

@author: yeltsin
"""

import sys, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import loadtxt


from trafficintelligence import storage, cvutils, utils

parser = argparse.ArgumentParser(description='The program displays feature or object trajectories overlaid over the video frames.', epilog = 'Either the configuration filename or the other parameters (at least video and database filenames) need to be provided.')
parser.add_argument('--cfg', dest = 'configFilename', help = 'name of the configuration file')
parser.add_argument('-d', dest = 'databaseFilename', help = 'name of the Sqlite database file (overrides the configuration file)')
parser.add_argument('-i', dest = 'videoFilename', help = 'name of the video file (overrides the configuration file)')
parser.add_argument('-t', dest = 'trajectoryType', help = 'type of trajectories to display', choices = ['feature', 'object'], default = 'feature')
parser.add_argument('-o', dest = 'homographyFilename', help = 'name of the image to world homography file')
parser.add_argument('--intrinsic', dest = 'intrinsicCameraMatrixFilename', help = 'name of the intrinsic camera file')
parser.add_argument('--distortion-coefficients', dest = 'distortionCoefficients', help = 'distortion coefficients', nargs = '*', type = float)
parser.add_argument('--undistorted-multiplication', dest = 'undistortedImageMultiplication', help = 'undistorted image multiplication', type = float)
parser.add_argument('-u', dest = 'undistort', help = 'undistort the video (because features have been extracted that way)', action = 'store_true')
parser.add_argument('-f', dest = 'firstFrameNum', help = 'number of first frame number to display', type = int)
parser.add_argument('-l', dest = 'lastFrameNum', help = 'number of last frame number to save (for image saving, no display is made)', type = int)
parser.add_argument('-r', dest = 'rescale', help = 'rescaling factor for the displayed image', default = 1., type = float)
parser.add_argument('-s', dest = 'nFramesStep', help = 'number of frames between each display', default = 1, type = int)
parser.add_argument('-n', dest = 'nObjects', help = 'number of objects to display', type = int)
parser.add_argument('--save-images', dest = 'saveAllImages', help = 'save all images', action = 'store_true')
parser.add_argument('--nzeros', dest = 'nZerosFilenameArg', help = 'number of digits in filenames', type = int)

args = parser.parse_args(["--cfg","tracking.cfg","-t","object"])
if args.homographyFilename is not None:
    invHomography = inv(loadtxt(args.homographyFilename))            
if args.intrinsicCameraMatrixFilename is not None:
    intrinsicCameraMatrix = loadtxt(args.intrinsicCameraMatrixFilename)
if args.distortionCoefficients is not None:
    distortionCoefficients = args.distortionCoefficients
if args.undistortedImageMultiplication is not None:
    undistortedImageMultiplication = args.undistortedImageMultiplication
if args.firstFrameNum is not None:
    firstFrameNum = args.firstFrameNum
if args.nObjects is not None:
    nObjects = args.nObjects
else:
    nObjects = None
    
params, videoFilename, databaseFilename, invHomography, intrinsicCameraMatrix, distortionCoefficients, undistortedImageMultiplication, undistort, firstFrameNum = storage.processVideoArguments(args)
objects = storage.loadTrajectoriesFromSqlite(databaseFilename, args.trajectoryType, nObjects)

# time_count = []
all_first_instcnace_frame = []
all_last_instance_frame = []
velocities_df_all_object = pd.DataFrame()
vehicle_total_frames = []

ID_Frame = []
vehicle_ID = []
Directions = []
vehicle_velocity_x = []
vehicle_velocity_y = []
vehicle_velocity = []
vehicle_id = 0
for obj in objects:
    all_first_instcnace_frame.append(obj.getFirstInstant())
    all_last_instance_frame.append(obj.getLastInstant())
    
    length_frames_individu = obj.getLastInstant()-obj.getFirstInstant() + 1
    vehicle_total_frames.append(length_frames_individu)
    
    # prepare each vehicle velocity data
    vehicle_ids = np.full(length_frames_individu, vehicle_id)
    vehicle_vel_x = obj.velocities.positions[0]
    vehicle_vel_y = obj.velocities.positions[1]
    
    vehicle_vel = np.sqrt(np.array(obj.velocities.positions[0])**2 + np.array(obj.velocities.positions[1])**2)
    
    speed_mean = (np.mean(vehicle_vel_x), np.mean(vehicle_vel_y))
    if speed_mean[0] < 0 and speed_mean[1]>0 : 
        Directions.extend(list(np.full(length_frames_individu, 1)))
    else : 
        Directions.extend(list(np.full(length_frames_individu, 0)))
            
    ID_Frame.extend(np.arange(obj.getFirstInstant(), obj.getLastInstant()+1))
    vehicle_ID.extend(list(vehicle_ids))
    vehicle_velocity_x.extend(vehicle_vel_x)
    vehicle_velocity_y.extend(vehicle_vel_y)
    vehicle_velocity.extend(vehicle_vel)
    
    vehicle_id = vehicle_id + 1
#     print(type(obj.velocities.positions)
#     time_count.append(obj.getFirstInstant())
#     obj.projectedPositions = obj.getPositions().homographyProject(homography)

velocities_df_all_object['ID_Frame'] = ID_Frame
velocities_df_all_object['vehicle_ID'] = vehicle_ID
velocities_df_all_object['vehicle_velocity_x'] = vehicle_velocity_x
velocities_df_all_object['vehicle_velocity_y'] = vehicle_velocity_y
velocities_df_all_object['vehicle_velocity'] = vehicle_velocity
velocities_df_all_object['Directions'] = Directions

vehicles_direction_0 = velocities_df_all_object.loc[velocities_df_all_object.Directions == 0]
mean_vel_direction_0 = vehicles_direction_0.groupby('ID_Frame').mean()['vehicle_velocity']*25*3.6

vehicles_direction_1 = velocities_df_all_object.loc[velocities_df_all_object.Directions == 1]
mean_vel_direction_1 = vehicles_direction_1.groupby('ID_Frame').mean()['vehicle_velocity']*25*3.6


plt.figure()
plt.plot(mean_vel_direction_0.index, mean_vel_direction_0.values, label = 'direction_0')
plt.plot(mean_vel_direction_1.index, mean_vel_direction_1.values, label = 'direction_1')
plt.legend()
plt.show()












