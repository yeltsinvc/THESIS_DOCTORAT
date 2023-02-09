#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:59:30 2019

@author: yeltsin
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sqlite3, logging
from numpy.linalg import inv
from numpy import loadtxt
import sys, argparse
import pandas as pd
#sys.path.insert(0,'/home/yeltsin/trafficintelligence')
#sys.path.insert(0,'C:/Users/valero/Documents/Stage/01Tracking/Trafff')
from trafficintelligence import moving, storage, cvutils

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

args = parser.parse_args(['--cfg',"tracking.cfg"])

params, videoFilename, databaseFilename, invHomography, intrinsicCameraMatrix, distortionCoefficients, undistortedImageMultiplication, undistort, firstFrameNum = storage.processVideoArguments(args)
homography=inv(invHomography)
#homography=invHomography
if args.homographyFilename is not None:
    homography = inv(loadtxt(args.homographyFilename))            
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
#homographyFilename='homography.txt'

img='world.jpg'


unitpixel=1/0.02350266155012682

    
trackerDatabaseFilename='data.db'

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    conn = sqlite3.connect(db_file)
    

    return conn
fps=25.0

databaseFilename='data.db' 
conn=create_connection(databaseFilename)

cur = conn.cursor()
#cur.execute("SELECT * FROM positions")

cur.execute('SELECT OF.object_id, P.frame_number, OF.road_user_type, P.x_coordinate, P.y_coordinate, P.line_n from positions P, objects OF WHERE P.trajectory_id = OF.object_id'+' ORDER BY OF.object_id, P.frame_number')
#cur.execute('SELECT P.trajectory_id, P.frame_number, P.x_coordinate, P.y_coordinate, P.line_n from positions P')

data=pd.DataFrame(np.array(cur.fetchall()),columns=["ID_Veh","Frame","Type","Position_x","Position_y",'Count_Line' ])

points=np.transpose(np.array(data[['Position_x','Position_y']]))
projected = cvutils.homographyProject(points, homography)*unitpixel

data['x_projected']=projected[0]
data['y_projected']=projected[1]

objets_position=data.groupby(['ID_Veh'])


import cv2
img=cv2.imread('world.jpg')
for name,group in objets_position:
    
    points_plot=np.array(group[['x_projected','y_projected']])
    cvXY=np.array([points_plot],np.int32)
    cvXY.reshape((-1,1,2))
    color=(0,250,250)
    if group['Type'].unique()==0:
        color=(0,250,250)
    elif group['Type'].unique()==1:
        color=(250,250,250)
    elif group['Type'].unique()==2:
        color=(250,0,250)
    elif group['Type'].unique()==3:
        color=(250,250,0)
    else:
        color=(0,0,0)
            
    cv2.polylines(img,[cvXY],False,color,thickness=3)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()  
cv2.imwrite("trajectoire4-p.jpg",img)
