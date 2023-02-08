# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 13:37:08 2022

@author: valero
"""


import outils as yvc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path=r"../Garyfallia/Cycle-Lane/DataBaseEXCEL/"

name_file="output.xlsx"

data=pd.read_excel(path+name_file)

data=data[~data['accel'].isnull()]

data_by_step=data.groupby("Time")

for time,data_current in data_by_step:
    print("Data")