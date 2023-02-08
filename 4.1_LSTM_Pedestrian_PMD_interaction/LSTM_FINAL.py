# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:51:40 2023

@author: valero
"""

import outils as yvc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


all_files=["Cycle-Lane","Cycle-Lane-Pedestrians-Crossing","Cycle-Track-1",
      "Cycle-Track-2","Pedestrian-Street","Pedestrian-Street-Reverse-Bikes",
      "Road-Pedestrians-Crossing-1"]

path_global=path=r"../Garyfallia/"
path_local="/DataBaseEXCEL/output.xlsx"


for file in all_files:
    print(file)
    data=pd.read_excel(path_global+file+path_local)
    data=data[~data['accel'].isnull()]
    data_by_step=data.groupby("Time")
    
"""path=r"../Garyfallia/Cycle-Lane/DataBaseEXCEL/"

name_file="output.xlsx"

data=pd.read_excel(path+name_file)

data=data[~data['accel'].isnull()]

data_by_step=data.groupby("Time")

for time,data_current in data_by_step:
    print("Data")"""