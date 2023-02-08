# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:44:35 2020

@author: valero
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from Scripts import preprocessing as pre

from keras.models import model_from_json
# load json and create model
json_file = open(r"Cmodel_CL.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"model_CL.h5")
print("Loaded model from disk")


dirFolder=r'Data_LSTM_CF/'

files=[item for item in os.listdir(dirFolder)]

dataset2 = [pre.readCSVBycicle(dirFolder+file) for file in files]
n=0
for data in dataset2:
    n+=1
    for id_vehicle in np.array(data.Vehicle_ID.unique()):
        X_train=[]
        y_train=[]
        list_vehicle_id=[]
        data_vehicle = data.loc[data['Vehicle_ID'] == id_vehicle].sort_values('Frame_ID')
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
        
        if len(data_vehicle)>50:
            
            for i in range(50, len(data_vehicle)):
                temp=(data_vehicle[i-50:i+1, :-1]).copy()
                temp[:,0]=temp[:,0]-temp[0,0]
                temp[:,1]=temp[:,1]-temp[0,1]
                X_train.append(temp[:50, :-1])
                y_train.append(temp[50, 0:2])
                list_vehicle_id.append(data_vehicle[i, 54])
            X_train, y_train = np.array(X_train), np.array(y_train)
            y_train_predicted = loaded_model.predict(X_train)
            y_real=y_train+data_vehicle[:len(y_train),0:2]
            y_real_predicted=y_train_predicted+data_vehicle[:len(y_train),0:2]
            
            fig = plt.figure()
            #plt.axis('equal')
            plt.plot(y_real[:,0],y_real[:,1],label= 'real (x,y)')
            plt.plot(y_real_predicted[:,0],y_real_predicted[:,1],label= 'predicted (x,y)')
            plt.xlabel('Position X')
            plt.ylabel('Position Y')
            #plt.plot(data_vehicle.predicted_train_x,data_vehicle.predicted_train_y, label = 'predicted_xy')
            plt.legend()
        #        for latPosition_LC in lane_change_info.loc[lane_change_info.Vehicle_ID == id_vehicle].Lane_change_Position_lat.values[0]: 
            fig.savefig(r'Results/%s_positionXY.png' %str(id_vehicle))
            plt.close()
        

            
            
            
            
            
            