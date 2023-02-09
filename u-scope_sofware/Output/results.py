#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:47:49 2021

@author: yeltsin
"""

import sys, argparse

from numpy.linalg import inv
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt
from trafficintelligence import storage, cvutils, utils
import datetime
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




fps=25.0
time_count=[]
for obj in objects:
    time_count.append(obj.getFirstInstant()/fps)
    obj.projectedPositions = obj.getPositions().homographyProject(invHomography)

TimeStart=datetime.datetime(year=2020, month=1, day=31, hour=8, minute=0, second=0)

intervale=1*60
delta = datetime.timedelta(seconds=intervale)
minTime=0
maxTime=300
bins=np.array(range(minTime,int(maxTime/intervale)+1))*intervale
dataHist=np.histogram(time_count, bins)

fig, ax = plt.subplots()
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Nombre des véhicules')
ax.set_xlabel('Heure(HH:MM:SS)')
ax.set_title('Débit du '+TimeStart.strftime("%m/%d/%Y")+': intervale ('+str(int(intervale/60))+ ' minutes)')
ax.set_xticks(dataHist[1][:-1])
labels=[(TimeStart+delta*i).strftime("%H:%M:%S")+"-"+(TimeStart+delta*(i+1)).strftime("%H:%M:%S") for i in range(len(bins)-1)]
ax.set_xticklabels(labels,rotation = 45)
ax.legend()
#ax.set_ylim(0, 100)
rects1=ax.bar(dataHist[1][:-1], dataHist[0], intervale, label='Car')
rects2=ax.bar(dataHist[1][:-1], dataHist[0], intervale, label='Vélo',bottom=dataHist[0])
fig.tight_layout()
ax.legend()
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
plt.show()
#rects2 = ax.bar(0.5, women_means, width, label='Women')