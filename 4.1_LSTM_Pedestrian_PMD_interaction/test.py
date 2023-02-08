# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:50:17 2022

@author: valero
"""
import numpy as np

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
    

obs=obstacles = [(-25, 25, 5, 5), (-25, 25, -5, -5)] #(xi,xf,yi,yf)

n=5
pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])
pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([25.0, 5.0])

x_vel_left = np.random.normal(1.34, 0.26, size=(n, 1))
x_vel_right = np.random.normal(-1.34, 0.26, size=(n, 1))
x_destination_left = 100.0 * np.ones((n, 1))
x_destination_right = -100.0 * np.ones((n, 1))

zeros = np.zeros((n, 1))

state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
state_right = np.concatenate(
    (pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1
)
initial_state = np.concatenate((state_left, state_right)) ##(xi,yi,vxi,vyi,dx,dy)

agents=[roadUserType(i) for i in range(len(initial_state))]

for veh,initial_cond in zip(agents,initial_state):
    veh.setInitialCondition(initial_cond)

"""for veh in agents:
    for veh1 in agents:
"""
pos=initial_state[:,0:2]

vec_pos=np.expand_dims(pos,1)-np.expand_dims(pos,0)
vec_pos = vec_pos[~np.all(vec_pos == 0, axis=2)] 
vec_pos = vec_pos.reshape(pos.shape[0], -1, pos.shape[1])
vec_norm=np.transpose(np.expand_dims(np.linalg.norm(vec_pos,axis=2),1),(0,2,1))
unit_vector=vec_pos/(np.ones((1,2))*vec_norm)

velocity=initial_state[:,2:4]
velocity_norm=np.linalg.norm(velocity,axis=1)
unit_vector_vel=velocity/np.transpose((np.ones((2,1))*velocity_norm))
unit_vector_vel = unit_vector_vel.reshape(unit_vector_vel.shape[0], -1, unit_vector_vel.shape[1])
#unit_vector_vel=np.ones((10,9,2))*unit_vector_vel
#dot_product = np.dot(unit_vector_vel, unit_vector)

dot_product=np.tensordot(unit_vector_vel,unit_vector,axes=([2], [2]))
dot_product=dot_product.reshape(dot_product.shape[0],dot_product.shape[2],dot_product.shape[3])
dot_product=dot_product[:,0,:]
angle=np.arccos(dot_product)

""" Angle of sight"""
angle_sight=120*np.pi/180

#angle = np.arccos(dot_product)
#ang = np.arctan2(vec_pos[:, 1], vecs[:, 0])        
#vec_pos = vec_pos[~np.eye(vec_pos.shape[0], dtype=bool), :]  
 

""" LSTM """
#Desired term
vec_pos=np.expand_dims(pos,1)-np.expand_dims(pos,0)
    
    
    
    
    