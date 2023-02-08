# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:06:31 2022

@author: valero
"""

import outils as yvc
import numpy as np
import matplotlib.pyplot as plt
# Define simulation
"""
    Create database for training process using SFM
    n: Number of pedestrian by side --> number of pedestrian 2n
    [leng,width] : lenght and widht of the road
    
    
"""
## Train
n_train=50
lenght_road=50
width_road=50
if_obstacles=0
time_step=150

states_train=yvc.runSFM(n_train,lenght_road,width_road,time_step,if_obstacles)

##Test
n_test=50
states_test=yvc.runSFM(n_test,lenght_road,width_road,time_step,if_obstacles,r_s=15)

#LSTM model
areas_dist=[4,8,20] #areas 0-2,2-10,10-20,20- inf
angle_of_sigth=120*np.pi/180


train,test= yvc.train_and_test_LSTM(states_train,states_test,areas_dist,angle_of_sigth)



"""
    BackPropagation
"""


""" Back Propagation Learning

"""
#y_predicted,y_test=yvc.backPropagationTraining(input_variables,output_variable)

step=4
y_predicted_LSTM,y_test_LSTM=yvc.LSTM_training_and_test(train,test,step)
rmse=np.sqrt(np.mean((np.linalg.norm(y_predicted_LSTM-y_test_LSTM,axis=1))**2))
LSTM_architecture="4_10_20_4_100_100_256_tanh_256_tanh_16_tanh_2_tanh"
yvc.plot(n_test*2,y_predicted_LSTM,y_test_LSTM,LSTM_architecture)
#plt.plot(y_predicted_LSTM[100:200,1])
#plt.plot(y_test_LSTM[100:200,1])
#plt.plot(y_predictec_LSTM[1], y_test_LSTM[1])