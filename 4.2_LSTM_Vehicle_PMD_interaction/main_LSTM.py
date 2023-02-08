# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:28:22 2020

@author: valero
"""

import os
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from Scripts import preprocessing as pre



BATCH_SIZE = 32
Time_steps = 50

"""dataset2 = pd.read_csv(r'Data_LSTM_CF/data_LSTM_CF_LC2.csv')

# Creating a data structure with 50 timesteps and 1 output
X_train_us101 = []
y_train_us101 = []  
list_vehicle_id = []

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
"""

dirFolder=r'Data_LSTM_CF/'

files=[item for item in os.listdir(dirFolder)]

dataset2 = [pre.readCSVBycicle(dirFolder+file) for file in files]

# Creating a data structure with 50 timesteps and 1 output

global X_train_us101, y_train_us101,list_vehicle_id
X_train_us101 = []
y_train_us101 = []  
list_vehicle_id = []

for data in dataset2:
    X_total,y_total,list_vehicle_id=pre.generateData(data)
   
n_random=list(range(0,len(X_total)))
trainin=random.sample(n_random,int(len(X_total)*0.75))
test=list(set(n_random)-set(trainin))
X_train=[]
y_train=[]
list_vehicle=[]
for i in trainin:
    X_train.append(X_total[i])
    y_train.append(y_total[i])
    list_vehicle.append(list_vehicle_id[i])

      
X_train_us101, y_train_us101 = np.array(X_train), np.array(y_train)


# Reshaping '3D tensor with shape (batch_size, timesteps, input_dim)' for RNN in Keras doc 
X_train_us101 = np.reshape(X_train_us101, (X_train_us101.shape[0], X_train_us101.shape[1], 53))

X_test=[]
y_test=[]
list_vehicle_test=[]
for j in test:
    X_test.append(X_total[j])
    y_test.append(y_total[j])
    list_vehicle_test.append(list_vehicle_id[j])
    
X_test, y_test = np.array(X_test), np.array(y_test)


# Reshaping '3D tensor with shape (batch_size, timesteps, input_dim)' for RNN in Keras doc 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 53))


# Part 2 - Building the RNN
# Importing the Keras libraries and packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.layers import Activation
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 64, return_sequences = True, input_shape = (X_train_us101.shape[1], 53)))
#    regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 64))
#    regressor.add(Dropout(0.2))
# regressor.add(Dense(units = 64))

regressor.add(Dense(units = 32))
# Adding the output layer
regressor.add(Dense(units = 2 ))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model_training = regressor.fit(X_train_us101, y_train_us101, epochs = 20, batch_size = 32)

plt.figure()
plt.plot(model_training.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.xticks(np.arange(20))
plt.show()

# serialize model to JSON
model_json = regressor.to_json()
with open(r"Cmodel_CL.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights(r"model_CL.h5")
print("Saved model to disk")



 # later...
from keras.models import model_from_json
# load json and create model
json_file = open(r"Cmodel_CL.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"model_CL.h5")
print("Loaded model from disk")

y_train_us101_predicted = loaded_model.predict(X_train_us101)
y_predicted_training_df = pd.DataFrame(y_train_us101_predicted, columns = ['predicted_train_x', 'predicted_train_y'])
y_predicted_training_df.to_csv(r'train_data_predicted.csv', sep=';', index=False)

y_test_predicted=loaded_model.predict(X_test)

Y_train_df = pd.DataFrame(y_train_us101, columns=['train_real_x', 'train_real_y'])
Y_train_df['Vehicle_ID'] = list_vehicle_id
Y_train_df['predicted_train_x'] = y_predicted_training_df['predicted_train_x']
Y_train_df['predicted_train_y'] = y_predicted_training_df['predicted_train_y']
#Y_train_df['predicted_train_speedX'] = y_predicted_training_df['predicted_train_speedX']
#Y_train_df['predicted_train_speedY'] = y_predicted_training_df['predicted_train_speedY']

vehicle_rmse_x=[]
vehicle_rmse_y=[]
#vehicle_rmse_speedX=[]
#vehicle_rmse_speedY=[]

for id_vehicle in np.array(Y_train_df.Vehicle_ID.unique()):
    data_vehicle = Y_train_df.loc[Y_train_df['Vehicle_ID'] == id_vehicle]
    vehicle_rmse_x.append([id_vehicle, np.sqrt(np.mean((data_vehicle.predicted_train_x-data_vehicle.train_real_x)**2))])
    vehicle_rmse_y.append([id_vehicle, np.sqrt(np.mean((data_vehicle.predicted_train_y-data_vehicle.train_real_y)**2))])
    #vehicle_rmse_speedX.append([id_vehicle, np.sqrt(np.mean((data_vehicle.predicted_train_speedX-data_vehicle.train_real_speedX)**2))])
    #vehicle_rmse_speedY.append([id_vehicle, np.sqrt(np.mean((data_vehicle.predicted_train_speedY-data_vehicle.train_real_speedX)**2))])
    
    
    '''fig = plt.figure()
    plt.plot(data_vehicle.train_real_speedX,label= 'real_test_speedX')
    plt.plot(data_vehicle.predicted_train_speedX, label = 'predicted_speedX')
    plt.legend()
#        for latPosition_LC in lane_change_info.loc[lane_change_info.Vehicle_ID == id_vehicle].Lane_change_Position_lat.values[0]: 
    fig.savefig(r'Results/Vitesse/%s_speedX.png' %str(id_vehicle))
    plt.close()'''
    
    fig = plt.figure()
    plt.plot(data_vehicle.train_real_x,data_vehicle.train_real_y,label= 'real_xy')
    plt.plot(data_vehicle.predicted_train_x,data_vehicle.predicted_train_y, label = 'predicted_xy')
    plt.legend()
#        for latPosition_LC in lane_change_info.loc[lane_change_info.Vehicle_ID == id_vehicle].Lane_change_Position_lat.values[0]: 
    fig.savefig(r'Results/Position/%s_positionXY.png' %str(id_vehicle))
    plt.close()
    
    fig = plt.figure()
    plt.plot(data_vehicle.train_real_x,label= 'real_test_positionX')
    plt.plot(data_vehicle.predicted_train_x, label = 'predicted_positionX')
    plt.legend()
#        for latPosition_LC in lane_change_info.loc[lane_change_info.Vehicle_ID == id_vehicle].Lane_change_Position_lat.values[0]: 
    fig.savefig(r'Results/PositionX/%s_positionX.png' %str(id_vehicle))
    plt.close()
    
    fig = plt.figure()
    plt.plot(data_vehicle.train_real_y,label= 'real_test_positionY')
    plt.plot(data_vehicle.predicted_train_y, label = 'predicted_positionY')
    plt.legend()
#        for latPosition_LC in lane_change_info.loc[lane_change_info.Vehicle_ID == id_vehicle].Lane_change_Position_lat.values[0]: 
    fig.savefig(r'Results/PositionY/%s_positionY.png' %str(id_vehicle))
    plt.close()
    
    
x=np.power(y_train_us101[:,0]-y_train_us101_predicted[:,0],2)

y=np.power(y_train_us101[:,1]-y_train_us101_predicted[:,1],2)
RMSE=np.mean(x+y)

x=np.power(y_test[:,0]-y_test_predicted[:,0],2)

y=np.power(y_test[:,1]-y_test_predicted[:,1],2)
RMSE=np.mean(x+y)
    