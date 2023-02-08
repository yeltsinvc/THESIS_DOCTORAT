# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:00:43 2022

@author: valero
"""


import numpy as np
import pysocialforce as psf
from pysocialforce.utils.plot import SceneVisualizer
import matplotlib.pyplot as plt
class roadUserType:
    def __init__(self,id_roadUserType):
        self.id=id_roadUserType
        self.type_str="Pedestrian"
        self.x=0
        self.y=0
        self.vx=0
        self.vy=0
        self.dx=0
        self.dy=0
    
    
    
    
    
    
    
    ##SETS
    def setUserType(self,UserType):
        self.type_str=UserType
        
    def setInitialCondition(self,initial_conditions):
        self.x=initial_conditions[0]
        self.y=initial_conditions[1]
        self.vx=initial_conditions[2]
        self.vy=initial_conditions[3]
        self.dx=initial_conditions[4]
        self.dy=initial_conditions[5]
    ##GETS
    def getID(self):
        return self.id
    

#obs=obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)] #(xi,xf,yi,yf)

n=30
pos_left = ((np.random.random((n, 2)) - 0.5) * 10.0) * np.array([25.0, 5.0])
pos_right = ((np.random.random((n, 2)) - 0.5) * 10.0) * np.array([25.0, 5.0])

x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
x_destination_left = 1000.0 * np.ones((n, 1))
x_destination_right = -1000.0 * np.ones((n, 1))

zeros = np.zeros((n, 1))

state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
state_right = np.concatenate(
    (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
)
initial_state = np.concatenate((state_left, state_right)) ##(xi,yi,vxi,vyi,dx,dy)

agents=[roadUserType(i) for i in range(len(initial_state))]

for veh,initial_cond in zip(agents,initial_state):
    veh.setInitialCondition(initial_cond)
    
""" Long Shor Temr Memory based in Particles of Dynamic 
    Input = [P1,P2,r_mean,v_cm]
    Output =[Q1,Q2]
    
    Desired Point = [P1,P2]
    P1: Relative position of desired point (x,y)(m)
    P2: Velocity of user-type (x,y) (m/s)

    Interactions = I = [P3,P4,P5,...]
    Distance of areas = [d1,d2,d3,...,dn] d: distace from road user-type m.
    m1+m2+m3+...+m_n=M
    r_mean = position of CM.
    m1=m2=m3=...=m_n
    r_mean=(m1*r1+m2*r2+...m_n*r_n)/(m1+m2+..+m_n)=(r1+r2+..+r_n)/n
    
    v_cm=(v1+v2+...+v_n)/n
    
    
         

"""

##Simulation
obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)]
agent_colors = [(1, 0, 0)] * n + [(0, 0, 1)] * n
s = psf.Simulator(initial_state, obstacles=obstacles)
s.step(150)
states=s.get_states()[0]
initial_first_if=True

r_mean=[]
v_cm=[]

P1=[]
P2=[]
for initial_state in states:
    # First term
    P1.append(initial_state[:,4:6]-initial_state[:,0:2])
    
    # Second term
    vel_temp_temp=[]
    for k in range(len(initial_state[:,2:4])):
        vel_temp_temp.append(np.delete(initial_state[:,2:4],k,0))
    P2.append(np.stack(vel_temp_temp))
    
    ##Interaction term r_mean
    inter_dist=np.expand_dims(initial_state[:,0:2],0)-np.expand_dims(initial_state[:,0:2],1)
    inter_dist= inter_dist[~np.eye(inter_dist.shape[0], dtype=bool), :]
    inter_dist= inter_dist.reshape(initial_state[:,0:2].shape[0], -1, initial_state[:,0:2].shape[1])
    
    norm=np.transpose(np.expand_dims(np.linalg.norm(inter_dist,axis=2),1),(0,2,1))
    
    vec_unit_pos=inter_dist/norm
    
    vec_unit_vel=initial_state[:,2:4]/np.expand_dims(np.linalg.norm(initial_state[:,2:4],axis=1),1)
    
    dot=np.transpose(np.cross(np.expand_dims(vec_unit_vel,1),vec_unit_pos))
    angle=np.arccos(dot).reshape(inter_dist.shape[0],inter_dist.shape[1],-1)
    
    Areas=[2,5,20] #Areas 0-2,2-10,10-20,20- inf
    angle_of_sight=120*np.pi/180
    if_angle=angle<angle_of_sight/2
    
    area_class=[]
    d_min=0
    
    P=[]
    v_cm_all=[]
    
    if initial_first_if:
        for i,d_max in enumerate(Areas):
            
                if_area=np.logical_and(d_min<norm,norm <d_max)
                d_min=d_max
                area_class.append(np.logical_and(if_angle,if_area))
                vector=area_class[-1]*inter_dist
                vector[vector == 0] = np.nan
                P.append(vector)
                r_mean.append([np.nanmean(P[-1], axis=1)])
                
                v_cm_temp=area_class[-1]*P2[-1]
                v_cm_temp[v_cm_temp == 0] = np.nan
                v_cm_all.append(v_cm_temp)
                v_cm.append([np.nanmean(v_cm_all[-1], axis=1)])
                
        initial_first_if=False
    else:
        for i,d_max in enumerate(Areas):
            if_area=np.logical_and(d_min<norm,norm <d_max)
            d_min=d_max
            area_class.append(np.logical_and(if_angle,if_area))
            vector=area_class[-1]*inter_dist
            
            vector[vector == 0] = np.nan
            P.append(vector)
            r_mean[i].append(np.nanmean(P[-1], axis=1))
            
            v_cm_temp=area_class[-1]*P2[-1]
            v_cm_temp[v_cm_temp == 0] = np.nan
            v_cm_all.append(v_cm_temp)
            v_cm[i].append(np.nanmean(v_cm_all[-1], axis=1))
            
            #initial_first=False

#output
            
acceleration=(states[1:151,:,2:4]-states[:150,:,2:4])/0.1
            
""" LSTM Training

"""

BATCH_SIZE = 32
Time_steps = 4

#dataset2 = pd.read_csv(r'Data_LSTM_CF/data_LSTM_CF_LC2.csv')

# Creating a data structure with 50 timesteps and 1 output
X_train_sfm = []
y_train_sfm = []  
list_vehicle_id = []
        
#y_train_sfm=acceleration

P1_mod=np.stack(P1)
P2_mod=states[:,:,2:4]

v_cm_train=[]
r_mean_train=[]
for v_cm_unit,r_mean_unit in zip(v_cm,r_mean):
    v_cm_train.append(np.stack(v_cm_unit))
    r_mean_train.append(np.stack(r_mean_unit))




union=np.concatenate((P1_mod, P2_mod,np.concatenate(v_cm_train,2),np.concatenate(r_mean_train,2)),2)
union=np.nan_to_num(union)

for i in range(Time_steps, 150): 
    X_train_sfm.append(np.swapaxes(union[i-Time_steps:i, :],0,1))
    #y_train_sfm.append(acceleration[i, :])
    #list_vehicle_id.append(data_vehicle[i, 54])
X_train_sfm=np.concatenate(X_train_sfm)
y_train_sfm=np.concatenate(acceleration[Time_steps:150])        
#X_train_sfm, y_train_sfm = np.concatenate(X_train_sfm), np.array(y_train_sfm)

# Reshaping '3D tensor with shape (batch_size, timesteps, input_dim)' for RNN in Keras doc 
X_train_sfm = np.reshape(X_train_sfm, (X_train_sfm.shape[0], X_train_sfm.shape[1], X_train_sfm.shape[2]))        

#X_train_sfm=X_train_sfm[:150,:,:]

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
regressor.add(LSTM(units = 16, return_sequences = True, input_shape = (X_train_sfm.shape[1], X_train_sfm.shape[2])))
#regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 16, activation='tanh'))
#regressor.add(Dropout(0.2))

regressor.add(Dense(units = 8,activation='softmax'))
#regressor.add(Dropout(0.2))

#regressor.add(Dense(units = 32))

# Adding the output layer
regressor.add(Dense(units = 2))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model_training = regressor.fit(X_train_sfm, y_train_sfm, epochs = 10, batch_size = 32)

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

### Test

n=5
pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])
pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])

x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
x_destination_left = 1000.0 * np.ones((n, 1))
x_destination_right = -1000.0 * np.ones((n, 1))

zeros = np.zeros((n, 1))

state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
state_right = np.concatenate(
    (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
)
initial_state = np.concatenate((state_left, state_right)) ##(xi,yi,vxi,vyi,dx,dy)

agents=[roadUserType(i) for i in range(len(initial_state))]

for veh,initial_cond in zip(agents,initial_state):
    veh.setInitialCondition(initial_cond)
    
""" Long Shor Temr Memory based in Particles of Dynamic 
    Input = [P1,P2,r_mean,v_cm]
    Output =[Q1,Q2]
    
    Desired Point = [P1,P2]
    P1: Relative position of desired point (x,y)(m)
    P2: Velocity of user-type (x,y) (m/s)

    Interactions = I = [P3,P4,P5,...]
    Distance of areas = [d1,d2,d3,...,dn] d: distace from road user-type m.
    m1+m2+m3+...+m_n=M
    r_mean = position of CM.
    m1=m2=m3=...=m_n
    r_mean=(m1*r1+m2*r2+...m_n*r_n)/(m1+m2+..+m_n)=(r1+r2+..+r_n)/n
    
    v_cm=(v1+v2+...+v_n)/n
    
    
         

"""

##Simulation
obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)]
agent_colors = [(1, 0, 0)] * n + [(0, 0, 1)] * n
s = psf.Simulator(initial_state, obstacles=obstacles)
s.step(150)
states=s.get_states()[0]
initial_first_if=True

r_mean=[]
v_cm=[]

P1=[]
P2=[]
for initial_state in states:
    # First term
    P1.append(initial_state[:,4:6]-initial_state[:,0:2])
    
    # Second term
    vel_temp_temp=[]
    for k in range(len(initial_state[:,2:4])):
        vel_temp_temp.append(np.delete(initial_state[:,2:4],k,0))
    P2.append(np.stack(vel_temp_temp))
    
    ##Interaction term r_mean
    inter_dist=np.expand_dims(initial_state[:,0:2],0)-np.expand_dims(initial_state[:,0:2],1)
    inter_dist= inter_dist[~np.eye(inter_dist.shape[0], dtype=bool), :]
    inter_dist= inter_dist.reshape(initial_state[:,0:2].shape[0], -1, initial_state[:,0:2].shape[1])
    
    norm=np.transpose(np.expand_dims(np.linalg.norm(inter_dist,axis=2),1),(0,2,1))
    
    vec_unit_pos=inter_dist/norm
    
    vec_unit_vel=initial_state[:,2:4]/np.expand_dims(np.linalg.norm(initial_state[:,2:4],axis=1),1)
    
    dot=np.transpose(np.cross(np.expand_dims(vec_unit_vel,1),vec_unit_pos))
    angle=np.arccos(dot).reshape(inter_dist.shape[0],inter_dist.shape[1],-1)
    
    Areas=[2,5,20] #Areas 0-2,2-10,10-20,20- inf
    angle_of_sight=120*np.pi/180
    if_angle=angle<angle_of_sight/2
    
    area_class=[]
    d_min=0
    
    P=[]
    v_cm_all=[]
    
    if initial_first_if:
        for i,d_max in enumerate(Areas):
            
                if_area=np.logical_and(d_min<norm,norm <d_max)
                d_min=d_max
                area_class.append(np.logical_and(if_angle,if_area))
                vector=area_class[-1]*inter_dist
                vector[vector == 0] = np.nan
                P.append(vector)
                r_mean.append([np.nanmean(P[-1], axis=1)])
                
                v_cm_temp=area_class[-1]*P2[-1]
                v_cm_temp[v_cm_temp == 0] = np.nan
                v_cm_all.append(v_cm_temp)
                v_cm.append([np.nanmean(v_cm_all[-1], axis=1)])
                
        initial_first_if=False
    else:
        for i,d_max in enumerate(Areas):
            if_area=np.logical_and(d_min<norm,norm <d_max)
            d_min=d_max
            area_class.append(np.logical_and(if_angle,if_area))
            vector=area_class[-1]*inter_dist
            
            vector[vector == 0] = np.nan
            P.append(vector)
            r_mean[i].append(np.nanmean(P[-1], axis=1))
            
            v_cm_temp=area_class[-1]*P2[-1]
            v_cm_temp[v_cm_temp == 0] = np.nan
            v_cm_all.append(v_cm_temp)
            v_cm[i].append(np.nanmean(v_cm_all[-1], axis=1))
            
            #initial_first=False

#output
            
acceleration=(states[1:151,:,2:4]-states[:150,:,2:4])/0.1
            
""" LSTM Test

"""



X_train_sfm = []
y_train_sfm = []  
list_vehicle_id = []
        
#y_train_sfm=acceleration

P1_mod=np.stack(P1)
P2_mod=states[:,:,2:4]

v_cm_train=[]
r_mean_train=[]
for v_cm_unit,r_mean_unit in zip(v_cm,r_mean):
    v_cm_train.append(np.stack(v_cm_unit))
    r_mean_train.append(np.stack(r_mean_unit))




union=np.concatenate((P1_mod, P2_mod,np.concatenate(v_cm_train,2),np.concatenate(r_mean_train,2)),2)
union=np.nan_to_num(union)

for i in range(Time_steps, 150): 
    X_train_sfm.append(np.swapaxes(union[i-Time_steps:i, :],0,1))
    #y_train_sfm.append(acceleration[i, :])
    #list_vehicle_id.append(data_vehicle[i, 54])
X_train_sfm=np.concatenate(X_train_sfm)
y_train_sfm=np.concatenate(acceleration[Time_steps:150])        
#X_train_sfm, y_train_sfm = np.concatenate(X_train_sfm), np.array(y_train_sfm)

# Reshaping '3D tensor with shape (batch_size, timesteps, input_dim)' for RNN in Keras doc 
X_test_sfm = np.reshape(X_train_sfm, (X_train_sfm.shape[0], X_train_sfm.shape[1], X_train_sfm.shape[2]))        


y_test_predicted=loaded_model.predict(X_test_sfm)

plt.plot(y_test_predicted[:,0],y_test_predicted[:,1],"bo")
plt.plot(y_train_sfm[:,0],y_train_sfm[:,1],"go")    

