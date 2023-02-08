# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:41:25 2019

@author: valero
"""

from __future__ import absolute_import
from __future__ import print_function


import os
import subprocess
import sys
import optparse
import random
import numpy as np
import matplotlib.pyplot as plt


def runStage():
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    from sumolib import checkBinary  
    import traci


    sumoBinary = checkBinary('sumo')
    vehID="mixed.10"

    trajetorie=[[]]

    x=[]
    y=[]
    traci.start([sumoBinary, "-c", "sublane_model.sumocfg","--fcd-output", "fcd.xml"])
    for i in range(500):
        traci.simulationStep()
        for k in traci.vehicle.getIDList():
            if k==vehID:
                x_pos,y_pos = traci.vehicle.getPosition(vehID)
                x.append(x_pos)
                y.append(y_pos)
    traci.close()
    return x,y
if __name__ == '__main__':
    runStage()
