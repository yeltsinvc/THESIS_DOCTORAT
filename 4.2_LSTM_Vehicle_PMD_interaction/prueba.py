# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 01:56:09 2020

@author: valero
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


def readCSVBycicle(dirFolder):
    dataset=pd.read_csv(dirFolder)
    dataset=dataset[dataset['class'] =='bicycle']
    return dataset
    


def generateData(dataset2):
   
    for id_vehicle in np.array(dataset2.Vehicle_ID.unique()):
        data_vehicle = dataset2.loc[dataset2['Vehicle_ID'] == id_vehicle].sort_values('Frame_ID')
        data_vehicle = data_vehicle[[
                 'Local_X', 'Local_Y','v_Vel_x','v_Vel_y','v_Length', 'v_Width',
                 
                 'exist_LeftPreceeding','LeftPreceeding_ID','lp_v_Length', 'lp_v_Width',
                 
                 'Local_X_LeftPreceeding','Local_Y_LeftPreceeding','v_X_Vel_LeftPreceeding','v_Y_Vel_LeftPreceeding',
                 
                 'exist_Preceeding','Preceeding_ID','p_v_Length', 'p_v_Width',
                 
                 'Local_X_Preceeding','Local_Y_Preceeding','v_X_Vel_Preceeding','v_Y_Vel_Preceeding',
                 
                 'exist_RightPreceeding','RightPreceeding_ID','rp_v_Length', 'rp_v_Width',
                 
                 'Local_X_RightPreceeding','Local_Y_RightPreceeding','v_X_Vel_RightPreceeding','v_Y_Vel_RightPreceeding',
                 
                 'exist_LeftFollower','LeftFollower_ID','lf_v_Length', 'lf_v_Width',
                 
                 'Local_X_LeftFollower','Local_Y_LeftFollower','v_X_Vel_LeftFollower','v_Y_Vel_LeftFollower',
                 
                 'exist_Follower','Follower_ID','f_v_Length', 'f_v_Width',
                 
                 'Local_X_Follower','Local_Y_Follower','v_X_Vel_Follower','v_Y_Vel_Follower',
                 
                 'exist_RightFollower','RightFollower_ID','rf_v_Length', 'rf_v_Width',
                 
                 'Local_X_RightFollower','Local_Y_RightFollower','v_X_Vel_RightFollower','v_Y_Vel_RightFollower','Vehicle_ID']].values
        data_vehicle[:,0]=data_vehicle[:,0]-data_vehicle[0,0]
        data_vehicle[:,1]=data_vehicle[:,1]-data_vehicle[0,1]
        
        
        for i in range(50, len(data_vehicle)): 
            X_train_us101.append(data_vehicle[i-50:i, :-1])
            y_train_us101.append(data_vehicle[i, 0:2])
            list_vehicle_id.append(data_vehicle[i, 54])
    return X_train_us101,y_train_us101,list_vehicle_id
    

BATCH_SIZE = 32
Time_steps = 50

dirFolder=r'Data_LSTM_CF/'

files=[item for item in os.listdir(dirFolder)]

dataset2 = [readCSVBycicle(dirFolder+file) for file in files]

# Creating a data structure with 50 timesteps and 1 output
x=y=l=[]
X_train_us101 = []
y_train_us101 = []  
list_vehicle_id = []

for data in dataset2:
    X_train_us101,y_train_us101,list_vehicle_id=generateData(data)
n_random=list(range(0,len(X_train_us101)))
trainin=random.sample(n_random,int(len(X_train_us101)*0.75))
test=list(set(n_random)-set(trainin))
X_train=[]
for i in trainin:
    X_train.append(X_train_us101[i])
