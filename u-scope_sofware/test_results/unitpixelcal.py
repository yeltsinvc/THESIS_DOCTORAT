#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:39:14 2019

@author: yeltsin
"""

import sys, argparse

import matplotlib.pyplot as plt
import numpy as np
import cv2

#from trafficintelligence import cvutils, utils, storage

worldFilename='world.JPG'
worldImg = plt.imread(worldFilename)
nPoints=2
#videoImg = plt.imread(args.videoFrameFilename)
plt.ion()
plt.figure()
plt.imshow(worldImg)
plt.tight_layout()
videoPts = np.array(plt.ginput(nPoints, timeout=3000))
#videoPts=np.array([[569.23565228,269.72999597],[541.96709005,253.36885863]])
dist = np.linalg.norm(videoPts[0]-videoPts[1])
dist_real=float(input("Real distance in meters:"))
unitpixel=dist_real/dist
print(unitpixel)

file_output = 'propiertes.txt'
f=open(file_output,"a")
f.write("Distance = " + str(dist) +"\n")
f.write("unitpixel = " + str(unitpixel) +"\n")
f.close()
