# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 23:06:39 2022

@author: valero
"""
import numpy as np
import pysocialforce as psf
from pysocialforce.utils.plot import SceneVisualizer
import matplotlib.pyplot as plt
import os

def runSFM(n,lenght_road,width_road,time_step,if_obstacles,r_s=10):
    np.random.seed(r_s)
    pos_left = ((np.random.random((n, 2)) - 0.5) * 2) * np.array([lenght_road/2, width_road/2])
    pos_right = ((np.random.random((n, 2)) - 0.5) * 2) * np.array([lenght_road/2, width_road/2])
    
    x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
    x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
    x_destination_left = 300.0 * np.ones((n, 1))
    x_destination_right = -300.0 * np.ones((n, 1))
    
    zeros = np.zeros((n, 1))
    
    state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
    state_right = np.concatenate(
        (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
    )
    initial_state = np.concatenate((state_left, state_right)) ##(xi,yi,vxi,vyi,dx,dy)
    
    
    
    ##Simulation
    obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)]
    agent_colors = [(1, 0, 0)] * n + [(0, 0, 1)] * n
    if if_obstacles:
        s = psf.Simulator(initial_state, obstacles=obstacles)
    else:
        s = psf.Simulator(initial_state)
        
    s.step(time_step)
    return s.get_states()[0]

def get_destinationDiff(states):
    return states[:,:,4:6]-states[:,:,0:2]

def get_velocity(states):
    return states[:,:,2:4]

def get_interaction_areas(states,areas_dist,angle_of_sight):
    r_mean=[]
    v_cm=[]
    P2=[]
    null_vec=True
    
    for state in states:
        inter_dist=np.expand_dims(state[:,0:2],0)-np.expand_dims(state[:,0:2],1)
        inter_dist= inter_dist[~np.eye(inter_dist.shape[0], dtype=bool), :]
        inter_dist= inter_dist.reshape(state[:,0:2].shape[0], -1, state[:,0:2].shape[1])
        norm=np.transpose(np.expand_dims(np.linalg.norm(inter_dist,axis=2),1),(0,2,1))
        vec_unit_pos=inter_dist/norm
        vec_unit_vel=state[:,2:4]/np.expand_dims(np.linalg.norm(state[:,2:4],axis=1),1)
        #dot=np.transpose(np.cross(np.expand_dims(vec_unit_vel,1),vec_unit_pos))
        v1= [np.array(len(vec_unit_pos[0])*[elem]) for elem in vec_unit_vel]
        dot=np.sum(np.array(v1)*vec_unit_pos,axis=2)
        angle=np.arccos(dot).reshape(inter_dist.shape[0],inter_dist.shape[1],1)
        if_angle=angle<angle_of_sight/2
        
        d_min=0
        area_class=[]
        P=[]
        vel_temp_temp=[]
        v_cm_all=[]
        for k in range(len(state[:,2:4])):
            vel_temp_temp.append(np.delete(state[:,2:4],k,0))
        P2.append(np.stack(vel_temp_temp))
        # positive and negative areas
        pos_area=inter_dist[:,:,1]>=0
        pos_area=pos_area.reshape(inter_dist.shape[0],inter_dist.shape[1],1)
        neg_area=inter_dist[:,:,1]<0
        neg_area=neg_area.reshape(inter_dist.shape[0],inter_dist.shape[1],1)
        def_areas=[pos_area,neg_area]#definition d'areas
        i=0
        for d_max in areas_dist:
            if_area=np.logical_and(d_min<norm,norm <d_max)
            ## add x positive logical
            
            ##
            d_min=d_max
            for area_unit in def_areas:
                area_class.append(np.logical_and(if_angle,if_area))
                area_class[-1]=np.logical_and(area_class[-1],area_unit)
                vector=area_class[-1]*inter_dist
                vector[vector == 0] = np.nan
                P.append(vector)
                
                
                
                v_cm_temp=area_class[-1]*P2[-1]
                v_cm_temp[v_cm_temp == 0] = np.nan
                v_cm_all.append(v_cm_temp)
                
                if null_vec:
                    r_mean.append([np.nanmean(P[-1], axis=1)])
                    v_cm.append([np.nanmean(v_cm_all[-1], axis=1)])
                    
                else:
                    r_mean[i].append(np.nanmean(P[-1], axis=1))
                    v_cm[i].append(np.nanmean(v_cm_all[-1], axis=1))
                i+=1
        null_vec=False
    v_cm_train=[]
    r_mean_train=[]
    for v_cm_unit,r_mean_unit in zip(v_cm,r_mean):
        v_cm_train.append(np.stack(v_cm_unit))
        r_mean_train.append(np.stack(r_mean_unit))
    return np.concatenate(v_cm_train,2),np.concatenate(r_mean_train,2)

def create_dataLSTM(states,areas_dist,angle_of_sigth):
    ## First term
    rel_pos_of_des=get_destinationDiff(states)
    
    ## Second term
    vel_road_user=get_velocity(states)
    
    ## Third term
    
    
    
    v_cm,r_cm=get_interaction_areas(states,areas_dist,angle_of_sigth)
    
    
    #input variables
    input_variables=np.concatenate((rel_pos_of_des, vel_road_user,v_cm,r_cm),2)
    input_variables=np.nan_to_num(input_variables)
    input_variables=input_variables[1:]
    #output_variables
    #aceleration
    output_variable=(states[1:len(states),:,2:4]-states[:(len(states)-1),:,2:4])
    
    data=np.concatenate((input_variables,output_variable),2)
    return data

def train_and_test_LSTM(states_train,states_test,areas_dist,angle_of_sigth):
    train=create_dataLSTM(states_train,areas_dist,angle_of_sigth)
    test=create_dataLSTM(states_test,areas_dist,angle_of_sigth)
    return train,test
    

def backPropagationTraining_before(input_variable,output_variable):
    import random
    #list_random=random.sample(range(len(output_variable)),len(output_variable))
    
    X_train_sfm=np.concatenate(input_variable)
    y_train_sfm=np.concatenate(output_variable)
    list_random=random.sample(range(len(X_train_sfm)),len(X_train_sfm))
    X_train_sfm=X_train_sfm[list_random]
    y_train_sfm=y_train_sfm[list_random]
    #data=np.concatenate((X_train_sfm,y_train_sfm),1)
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow.keras.layers import Activation
    from keras.layers import Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    #scaler = MinMaxScaler()
    #scaled = scaler.fit_transform(data)
    # Initialising the RNN
    regressor = Sequential()
    regressor.add(Dense(64, input_shape=(16,), activation='tanh'))
    regressor.add(Dense(64, activation='tanh'))
    #regressor.add(Dense(64, activation='tanh'))
    
    regressor.add(Dense(32, activation='tanh'))
    #regressor.add(Dense(16, activation='tanh'))
    regressor.add(Dense(8, activation='tanh'))
    regressor.add(Dense(2, activation='tanh'))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    #regressor.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fitting the RNN to the Training set
    model_training = regressor.fit(X_train_sfm, y_train_sfm, epochs = 100, batch_size = 4)
    
    plt.figure()
    plt.plot(model_training.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.xticks(np.arange(10))
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
    
    
    ### Test
    n=10
    lenght_road=50
    width_road=10
    if_obstacles=0
    time_step=150
    
    states_test=runSFM(n,lenght_road,width_road,time_step,if_obstacles)
    
    
    """
        BackPropagation
    """
    ## First term
    rel_pos_of_des_test=get_destinationDiff(states_test)
    
    ## Second term
    vel_road_user_test=get_velocity(states_test)
    
    ## Third term
    
    areas_dist=[2,5,20] #areas 0-2,2-10,10-20,20- inf
    angle_of_sigth=120*np.pi/180
    
    v_cm_test,r_cm_test=get_interaction_areas(states_test,areas_dist,angle_of_sigth)
    
    
    #input variables
    input_variables_test=np.concatenate((rel_pos_of_des_test, vel_road_user_test,v_cm_test,r_cm_test),2)
    input_variables_test=np.nan_to_num(input_variables_test)
    input_variables_test=np.concatenate(input_variables_test[:-1])
    #output_variables
    #aceleration
    output_variable_test=np.concatenate(states_test[1:len(states_test),:,2:4]-states_test[:(len(states_test)-1),:,2:4])
    
    
    y_test_predicted=loaded_model.predict(input_variables_test)
    
    #y_test_predicted=scaler.inverse_transform(y_test_predicted)
    
    return y_test_predicted,output_variable_test

def backPropagationTraining(input_variable,output_variable):
    
    import random
    #list_random=random.sample(range(len(output_variable)),len(output_variable))
    
    X_train_sfm=np.concatenate(input_variable)
    y_train_sfm=np.concatenate(output_variable)
    list_random=random.sample(range(len(X_train_sfm)),len(X_train_sfm))
    X_train_sfm=X_train_sfm[list_random]
    y_train_sfm=y_train_sfm[list_random]
    data_train=np.concatenate((X_train_sfm,y_train_sfm),1)
    
    ### Test
    n=10
    lenght_road=50
    width_road=10
    if_obstacles=0
    time_step=150
    
    states_test=runSFM(n,lenght_road,width_road,time_step,if_obstacles)
    
    
    """
        BackPropagation
    """
    ## First term
    rel_pos_of_des_test=get_destinationDiff(states_test)
    
    ## Second term
    vel_road_user_test=get_velocity(states_test)
    
    ## Third term
    
    areas_dist=[2,5,20] #areas 0-2,2-10,10-20,20- inf
    angle_of_sigth=120*np.pi/180
    
    v_cm_test,r_cm_test=get_interaction_areas(states_test,areas_dist,angle_of_sigth)
    
    
    #input variables
    input_variables_test=np.concatenate((rel_pos_of_des_test, vel_road_user_test,v_cm_test,r_cm_test),2)
    input_variables_test=np.nan_to_num(input_variables_test)
    input_variables_test=np.concatenate(input_variables_test[:-1])
    #output_variables
    #aceleration
    output_variable_test=np.concatenate(states_test[1:len(states_test),:,2:4]-states_test[:(len(states_test)-1),:,2:4])
    
    X_train_sfm_test=input_variables_test
    y_train_sfm_test=output_variable_test
    
    data_test=np.concatenate((X_train_sfm_test,y_train_sfm_test),1)
    
    data=np.concatenate((data_train,data_test))
    
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow.keras.layers import Activation
    from keras.layers import Dropout
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    train, test = scaled[0:len(data_train)], scaled[len(data_train):len(data)]
    
    # Initialising the RNN
    regressor = Sequential()
    regressor.add(Dense(64, input_shape=(16,), activation='sigmoid'))
    regressor.add(Dense(64, activation='sigmoid'))
    #regressor.add(Dense(64, activation='tanh'))
    
    regressor.add(Dense(32, activation='sigmoid'))
    #regressor.add(Dense(16, activation='tanh'))
    regressor.add(Dense(8, activation='sigmoid'))
    regressor.add(Dense(2, activation='sigmoid'))
    
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    #regressor.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    #regressor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Fitting the RNN to the Training set
    model_training = regressor.fit(train[:,0:16], train[:,16:18], epochs = 10, batch_size = 16)
    
    plt.figure()
    plt.plot(model_training.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.xticks(np.arange(10))
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
    
    y_test_predicted=loaded_model.predict(test[:,0:16])
    test_predicted=scaler.inverse_transform(np.concatenate((test[:,0:16],y_test_predicted),1))
    
    y_test_predicted=test_predicted[:,16:18]
    
    return y_test_predicted,data_test[:,16:18]

def LSTMTraining(input_variable,output_variable,step):
    import random
    #list_random=random.sample(range(len(output_variable)),len(output_variable))
    
    X_train_sfm=np.concatenate(input_variable)
    y_train_sfm=np.concatenate(output_variable)
    
    #list_random=random.sample(range(len(X_train_sfm)),len(X_train_sfm))
    #X_train_sfm=X_train_sfm[list_random]
    #y_train_sfm=y_train_sfm[list_random]
    data_train=np.concatenate((X_train_sfm,y_train_sfm),1)
    
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train= scaler.fit_transform(data_train)
    
    train=np.reshape(train, (input_variable.shape[0], input_variable.shape[1], 
                                   -1))
    X_train_sfm=[]
    y_train_sfm=[]
    data_train_LSTM=[]
    for i in range(step, len(input_variable)): 
        data_train_LSTM.append(np.swapaxes(train[i-step:i, :,:],0,1))
        y_train_sfm.append(np.transpose(np.swapaxes(train[step, :,16:18],0,1)))
    data_train_LSTM=np.concatenate(data_train_LSTM)
    #X_train_sfm=data_train_LSTM[:,:,0:16]
    X_train_sfm=data_train_LSTM
    y_train_sfm=np.concatenate(y_train_sfm)
    #y_train_sfm=data_train_LSTM[:,-1,16:18]
    #y_train_sfm=np.reshape(y_train_sfm, (y_train_sfm.shape[0], 1, 
    #                               y_train_sfm.shape[1]))
    list_random=random.sample(range(len(X_train_sfm)),len(X_train_sfm))
    X_train_sfm=X_train_sfm[list_random]
    y_train_sfm=y_train_sfm[list_random]
    # Reshaping '3D tensor with shape (batch_size, timesteps, input_dim)' for RNN in Keras doc 
    #X_train_sfm = np.reshape(X_train_sfm, (X_train_sfm.shape[0], X_train_sfm.shape[1], X_train_sfm.shape[2]))        
    
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow.keras.layers import Activation
    from keras.layers import Dropout
    
    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train_sfm.shape[1], X_train_sfm.shape[2])))
    #regressor.add(Dropout(0.2))
    
    regressor.add(LSTM(units = 256, activation='tanh'))
    #regressor.add(Dropout(0.2))
    
    regressor.add(Dense(units =32,activation='tanh'))
    #regressor.add(Dropout(0.2))
    regressor.add(Dense(units =8,activation='tanh'))
    #regressor.add(Dense(units = 32))
    
    # Adding the output layer
    regressor.add(Dense(units = 2,activation='tanh'))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    model_training = regressor.fit(X_train_sfm, y_train_sfm, epochs = 100, batch_size = 32)
    
    plt.figure()
    plt.plot(model_training.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.xticks(np.arange(20))
    plt.show()

    #train, test = scaled[0:len(data_train)], scaled[len(data_train):len(data)]
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
    ### Test
    n=10
    lenght_road=50
    width_road=10
    if_obstacles=0
    time_step=150
    
    states_test=runSFM(n,lenght_road,width_road,time_step,if_obstacles)
    
    
    """
        #BackPropagation
    """
    ## First term
    rel_pos_of_des_test=get_destinationDiff(states_test)
    
    ## Second term
    vel_road_user_test=get_velocity(states_test)
    
    ## Third term
    
    areas_dist=[2,5,20] #areas 0-2,2-10,10-20,20- inf
    angle_of_sigth=120*np.pi/180
    
    v_cm_test,r_cm_test=get_interaction_areas(states_test,areas_dist,angle_of_sigth)
    
    
    #input variables
    input_variables_test=np.concatenate((rel_pos_of_des_test, vel_road_user_test,v_cm_test,r_cm_test),2)
    input_variables_test=np.nan_to_num(input_variables_test)
    input_variables_test=input_variables_test[:-1]
    #output_variables
    #aceleration
    output_variable_test=(states_test[1:len(states_test),:,2:4]-states_test[:(len(states_test)-1),:,2:4])
    X_train_sfm_test=input_variables_test
    y_train_sfm_test=output_variable_test
    
    data_test=np.concatenate((X_train_sfm_test,y_train_sfm_test),2)
    data_test=np.concatenate(data_test)
    test=scaler.transform(data_test)
    test=np.reshape(test, (input_variables_test.shape[0], input_variables_test.shape[1], 
                                   -1))
    X_test_sfm=[]
    y_test_sfm=[]
    data_test_LSTM=[]
    for i in range(step, len(input_variables_test)): 
        data_test_LSTM.append(np.swapaxes(test[i-step:i, :,:],0,1))
        y_test_sfm.append(np.transpose(np.swapaxes(train[i-step, :,16:18],0,1)))
    data_test_LSTM=np.concatenate(data_test_LSTM)
    #X_train_sfm=data_train_LSTM[:,:,0:16]
    X_test_sfm=data_test_LSTM
    y_train_sfm=np.concatenate(y_test_sfm)
    #X_test_sfm=data_test_LSTM[:,:,0:16]
    #y_test_sfm=data_test_LSTM[:,-1,16:18]
    #y_test_sfm=np.reshape(y_test_sfm, (y_test_sfm.shape[0], 1, 
    #                               y_test_sfm.shape[1]))
    
    #data=np.concatenate((data_train,data_test))
    
    y_test_predicted=loaded_model.predict(X_test_sfm)
    
    #data_predicted=np.concatenate((np.concatenate(X_test_sfm),y_test_predicted))
    y_test_predicted=scaler.inverse_transform(np.concatenate((np.ones((2920,16)),y_test_predicted),1))[:,16:18]
    
    
    data_test_LSTM_real=[]
    for i in range(step, len(input_variables_test)): 
        data_test_LSTM_real.append(np.swapaxes(output_variable_test[i-step:i, :,:],0,1))
    data_test_LSTM_real=np.concatenate(data_test_LSTM_real)
    #X_test_sfm_real=data_test_LSTM[:,:,0:16]
    y_test_sfm_real=data_test_LSTM_real[:,-1]
    
    return y_test_predicted,y_test_sfm_real


def get_X_and_Y(train,step):
    X_sfm=[]
    y_sfm=[]
    #data_train_LSTM=[]
    for i in range(step, len(train)): 
        X_sfm.append(np.swapaxes(train[i-step:i, :,:],0,1))
        y_sfm.append(np.transpose(np.swapaxes(train[step, :,-2:],0,1)))
    
    return np.concatenate(X_sfm),np.concatenate(y_sfm)
    
def LSTM_training_and_test(train,test,step):
    
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled= scaler.fit_transform(np.concatenate(train))
    train_scaled=np.reshape(train_scaled, (train.shape[0], train.shape[1], 
                                   -1))
    X_train_sfm,y_train_sfm = get_X_and_Y(train_scaled,step)
    
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from tensorflow.keras.layers import Activation
    from keras.layers import Dropout
    
    # Initialising the RNN
    regressor = Sequential()
    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 256, activation='tanh',return_sequences = True, input_shape = (X_train_sfm.shape[1], X_train_sfm.shape[2])))
    #regressor.add(Dropout(0.2))
    
    #regressor.add(LSTM(units = 36, activation='tanh',input_shape = (X_train_sfm.shape[1], X_train_sfm.shape[2])))
    
    #regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 256, activation='tanh'))
    
    #regressor.add(Dense(units =18,activation='tanh'))
    #regressor.add(Dropout(0.2))
    #regressor.add(Dense(units =32,activation='tanh'))
    regressor.add(Dense(units = 16,activation='tanh'))
    
    # Adding the output layer
    regressor.add(Dense(units = 2,activation='tanh'))
    
    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    # Fitting the RNN to the Training set
    model_training = regressor.fit(X_train_sfm, y_train_sfm, epochs = 100, batch_size = 32)
    
    plt.figure()
    plt.plot(model_training.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.xticks(np.arange(100))
    plt.show()

    #train, test = scaled[0:len(data_train)], scaled[len(data_train):len(data)]
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
    
    test_scaled=scaler.transform(np.concatenate(test))
    test_scaled=np.reshape(test_scaled, (test.shape[0], test.shape[1], 
                                   -1))
    X_test_sfm,y_test_sfm = get_X_and_Y(test_scaled,step)
    
    y_test_predicted=loaded_model.predict(X_test_sfm)
    
    size_test=(X_test_sfm.shape[0],X_test_sfm.shape[2]-2)
    y_test_predicted=scaler.inverse_transform(np.concatenate((np.ones(size_test),y_test_predicted),1))[:,-2:]
    
    X_test_real_sfm,y_test_real_sfm = get_X_and_Y(test,step)
    
    return y_test_predicted,y_test_real_sfm

def plot(n_test,y_predicted_LSTM,y_test_LSTM,name):
    import shutil

    y_predicted=np.reshape(y_predicted_LSTM,(n_test,-1,y_predicted_LSTM.shape[1]))
    y_test=np.reshape(y_test_LSTM,(n_test,-1,y_predicted_LSTM.shape[1]))
    path=r'C:/Users/valero/Documents/Yeltsin 2.0/Doctorat/Results/'
    if os.path.exists(path+name):
        shutil.rmtree(path+name)
        os.makedirs(path+name)
        os.makedirs(path+name+"/X")
        os.makedirs(path+name+"/Y")
    else:
        os.makedirs(path+name)
        os.makedirs(path+name+"/X")
        os.makedirs(path+name+"/Y")
    #plot trajectories
    ID_veh=0
    
    for p,t in zip(y_predicted,y_test):
        
        fig = plt.figure()
        plt.plot(p[:,0],label='predicted')
        plt.plot(t[:,0],label='real')
        plt.xlabel("Time Step")
        plt.ylabel("Acelleration x")
        plt.title(' Pedestrian acceleration ID: %d' %ID_veh)
        plt.legend()
        plt.savefig(path+name+'/X/%d.png' %ID_veh)
        plt.close()
        
        fig = plt.figure()
        plt.plot(p[:,1],label='predicted')
        plt.plot(t[:,1],label='real')
        plt.xlabel("Time Step")
        plt.ylabel("Acelleration y")
        plt.title(' Pedestrian acceleration ID: %d' %ID_veh)
        plt.legend()
        plt.savefig(path+name+'/Y/%d.png' %ID_veh)
        plt.close()
        
        ID_veh+=1
        
def get_destinationDiff_AIMSUN(states):
    data_by_id=states.groupby(by=['       VehNr'])
    Diff_Destination_x=[]
    Diff_Destination_y=[]
    for veh_id,unit_by_id in data_by_id:
        #print(veh_id)
        Diff_Destination_x.append(list(unit_by_id['      WorldX']-unit_by_id['      WorldX'].iloc[-1]))
        Diff_Destination_y.append(list(unit_by_id['      WorldY']-unit_by_id['      WorldY'].iloc[-1]))
    
    states['Diff_Destination_x']=np.concatenate(Diff_Destination_x)
    states['Diff_Destination_y']=np.concatenate(Diff_Destination_y)
    return states

def create_dataLSTM_AIMSUN(states,areas_dist,angle_of_sigth):
    ## First term
    rel_pos_of_des=get_destinationDiff_AIMSUN(states)
    
    ## Second term
    vel_road_user=get_velocity(states)
    
    ## Third term
    
    
    
    v_cm,r_cm=get_interaction_areas(states,areas_dist,angle_of_sigth)
    
    
    #input variables
    input_variables=np.concatenate((rel_pos_of_des, vel_road_user,v_cm,r_cm),2)
    input_variables=np.nan_to_num(input_variables)
    input_variables=input_variables[1:]
    #output_variables
    #aceleration
    output_variable=(states[1:len(states),:,2:4]-states[:(len(states)-1),:,2:4])
    
    data=np.concatenate((input_variables,output_variable),2)
    return data
def train_and_test_LSTM_AIMSUN(states,n_train,areas_dist,angle_of_sigth):
    n_values = states['       VehNr'].nunique()
    n_train = int(n_values*n_train)
    n_test = n_values - n_train
    
    list_id=states['       VehNr'].unique()
    list_train_id = list_id[:n_train]
    list_test_id = list_id[n_train:]
    
    states_train_first=states[states['       VehNr']<=list_train_id[-1]]
    states_test_first=states[states['       VehNr']>list_train_id[-1]]
    
    train = create_dataLSTM_AIMSUN(states_train_first,areas_dist,angle_of_sigth)
    test = create_dataLSTM_AIMSUN(states_test_first,areas_dist,angle_of_sigth)
    
    return train, test
  

