# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 10:55:28 2020

@author: valero
"""
import matplotlib.pyplot as plt
        
import numpy as np
import os
import subprocess
import sys
import math
from numpy import loadtxt

import cv2
import pandas as pd
from scipy.signal import savgol_filter
from shapely.geometry import Point,Polygon
from geopandas import GeoDataFrame
import shapely.wkt
import pandas as pd

from Scripts import preprocessing as pre

ind_dir="Data"

#data_ind=pd.read_csv(ind_dir+r"\00_tracks.csv")

col_names=['Vehicle_ID','Frame_ID','Total_Frames','Global_Time','Local_X','Local_Y','Global_X','Global_Y','v_Length','v_Width','v_Class','v_Vel','v_Acc','Lane_ID','Preceeding','Following','Space_Hdwy','Time_Hdwy']

#Vehicle_ID	Frame_ID	Local_X	Local_Y	v_Vel	v_Acc	v_Length	v_Width	Lane_ID	Following	Preceeding	left_follower_ID	left_leader_ID	right_follower_ID	right_leader_ID	Vehicle_ID_Following	Local_X_Flowwing	Local_Y_Following	v_Vel_Following	v_Acc_Following	v_Length_Following	v_Width_Following	Lane_ID_Following	Vehicle_ID_Preceeding	Local_X_Preceeding	Local_Y_Preceeding	v_Vel_Preceeding	v_Acc_Preceeding	v_Length_Preceeding	v_Width_Precdeing	Lane_ID_Preceeding	Vehicle_ID_LeftLeader	Local_X_LeftLeader	Local_Y_LeftLeader	v_Vel_LeftLeader	v_Acc_LeftLeader	v_Length_LeftLeader	v_Width_LeftLeader	Lane_ID_LeftLeader	Vehicle_ID_LeftFollower	Local_X_LeftFollower	Local_Y_LeftFollower	v_Vel_LeftFollower	v_Acc_LeftFollower	v_Length_LeftFollower	v_Width_LeftFollower	Lane_ID_LeftFollower	Vehicle_ID_RightLeader	Local_X_RightLeader	Local_Y_RightLeader	v_Vel_RightLeader	v_Acc_RightLeader	v_Length_RightLeader	v_Width_RightLeader	Lane_ID_RightLeader	Vehicle_ID_RightFollower	Local_X_RightFollower	Local_Y_RightFollower	v_Vel_RightFollower	v_Acc_RightFollower	v_Length_RightFollower	v_Width_RightFollower	Lane_ID_RightFollower"	



##read files of database


data_ind, static_info, meta_info=pre.read_all_recordings_from_csv(ind_dir)
ortho_px_to_meter = meta_info["orthoPxToMeter"][0]

##Geometry
#background=r'07_background.png'
#pre.getGeometry(ind_dir,background,ortho_px_to_meter)#comment if the geometry already exist


geomet=pd.read_csv(ind_dir+r'\geometry.csv')
poligons=[]
k=0

geomet=geomet.T
geomet = geomet[geomet[0] != 0]
geo=[pre.Geometry(indx) for indx in geomet.index]

for objgeo in geo:
    objgeo.polygon=shapely.wkt.loads(geomet[0][objgeo.name])

data_ngsim=pd.DataFrame() 
data_ngsim['Vehicle_ID']=data_ind['trackId']
data_ngsim['Frame_ID']=data_ind['frame']
data_ngsim['xCenter']=data_ind['xCenter']
data_ngsim['yCenter']=data_ind['yCenter']
data_ngsim['v_Length']=data_ind['length']
data_ngsim['v_Width']=data_ind['width']
#data_ngsim['v_Class']=data_ind['length']
data_ngsim['v_Vel']=data_ind.apply(lambda x: (x['xVelocity']**2+x['yVelocity']**2)**.5, axis=1)
data_ngsim['v_Acc']=data_ind.apply(lambda x: (x['xAcceleration']**2+x['yAcceleration']**2)**.5, axis=1)

points=[Point(xy) for xy in zip(data_ngsim['xCenter'], -data_ngsim['yCenter'])]
for objgeo in geo:
    objgeo.isLane=[]
    for point in points:
        objgeo.isLane.append(point.within(objgeo.polygon))
    data_ngsim[objgeo.name]=objgeo.isLane
for i in range(len(geo)):
    data_ngsim.loc[data_ngsim['Lane '+str(i+1)] == True, 'Lane_ID'] = i+1

data_ngsim.dropna(subset = ["Lane_ID"], inplace=True)

##Determining pente
vector_trayectories=pre.calculate_pente(data_ngsim) #[x,y]
data_ngsim=pre.rotate_trayectories(vector_trayectories,data_ngsim)

#data_ngsim['dist_relative_X_leader']=math.inf
#data_ngsim['dist_relative_X_follower']=math.inf
data_ngsim['Preceeding']=data_ngsim['Following']=0
data_ngsim['Space_Hdwy_Preceeding']=data_ngsim['Space_Hdwy_Following']=math.inf

for temps in range(data_ngsim['Frame_ID'].min(),data_ngsim['Frame_ID'].max()+1):
    data_vehicle = data_ngsim.loc[data_ngsim['Frame_ID'] == temps]
    
    data_vehicle_by_lane=data_vehicle.groupby(['Lane_ID'])
    for group_by_lane in data_vehicle_by_lane.groups:
        data_vehicle_lane=data_vehicle_by_lane.get_group(group_by_lane)
        #print(data_vehicle_lane)
        k=0
        for index,row in data_vehicle_lane.iterrows():
            #print(index,row)
            vehicle=row.T
            dist_to_leader=math.inf
            dist_to_follower=math.inf
            for index1,row1 in data_vehicle.iterrows():
                if index != index1:
                    
                    space_headway_following=row1['Local_X']-row['Local_X']-row['v_Length']/2-row1['v_Length']/2
                    space_headway_preceeding=row['Local_X']-row1['Local_X']-row['v_Length']/2-row1['v_Length']/2
                    if row['Local_X']-row1['Local_X']<0:
                        dist_to_leader=space_headway_following
                        if  space_headway_following < data_ngsim['Space_Hdwy_Following'][index]:
                            dist_to_leader=space_headway_following
                            data_ngsim['Following'][index]=row1['Vehicle_ID']
                            data_ngsim['Space_Hdwy_Following'][index]=dist_to_leader
                        
                    else:
                        dist_to_preeding=space_headway_preceeding
                        if space_headway_preceeding < data_ngsim['Space_Hdwy_Preceeding'][index]:
                            dist_to_preeding=space_headway_preceeding
                            data_ngsim['Preceeding'][index]=row1['Vehicle_ID']
                            data_ngsim['Space_Hdwy_Preceeding'][index]=dist_to_preeding

#data_ngsim = data_ngsim[(data_ngsim.Preceeding != 0) | (data_ngsim.Following != 0)]
                        
                        
## Generate right and left
data_ngsim['x_rFollowing']=math.inf
data_ngsim['y_rFollowing']=math.inf
data_ngsim['id_rFollowing']=0
data_ngsim['x_lFollowing']=math.inf
data_ngsim['y_lFollowing']=math.inf
data_ngsim['id_lFollowing']=0


data_ngsim['x_rPrecceding']=math.inf
data_ngsim['y_rPrecceding']=math.inf
data_ngsim['id_rPrecceding']=0
data_ngsim['x_lPrecceding']=math.inf
data_ngsim['y_lPrecceding']=math.inf
data_ngsim['id_lPrecceding']=0


for temps in range(data_ngsim['Frame_ID'].min(),data_ngsim['Frame_ID'].max()+1):
    data_vehicle = data_ngsim.loc[data_ngsim['Frame_ID'] == temps]
    for index,row in data_vehicle.iterrows():
        for index1,row1 in data_vehicle.iterrows():
            if index != index1:
                rl_lane=row['Lane_ID']
                if (rl_lane+1==row1['Lane_ID']):
                    if row['Local_X']-row1['Local_X']<0:
                        x_rfFollowing=row1['Local_X']-row['Local_X']-row['v_Length']/2-row1['v_Length']/2
                        y_rfFollowing=row1['Local_Y']-row['Local_Y']
                        if y_rfFollowing >0:
                            data_ngsim['id_rFollowing'][index]=row1['Vehicle_ID']
                            data_ngsim['x_rFollowing'][index]=x_rfFollowing
                            data_ngsim['y_rFollowing'][index]=y_rfFollowing
                        else:
                            data_ngsim['id_lFollowing'][index]=row1['Vehicle_ID']
                            data_ngsim['x_lFollowing'][index]=x_rfFollowing
                            data_ngsim['y_lFollowing'][index]=y_rfFollowing
                    else:
                        x_rfPrecceding=row['Local_X']-row1['Local_X']-row['v_Length']/2-row1['v_Length']/2
                        y_rfPrecceding=row1['Local_Y']-row['Local_Y']
                        if y_rfPrecceding >0:
                            data_ngsim['id_rPrecceding'][index]=row1['Vehicle_ID']
                            data_ngsim['x_rPrecceding'][index]=x_rfPrecceding
                            data_ngsim['y_rPrecceding'][index]=y_rfPrecceding
                        else:
                            data_ngsim['id_lPrecceding'][index]=row1['Vehicle_ID']
                            data_ngsim['x_lPrecceding'][index]=x_rfPrecceding
                            data_ngsim['y_lPrecceding'][index]=y_rfPrecceding
                elif(rl_lane-1==row1['Lane_ID']):
                    if row['Local_X']-row1['Local_X']<0:
                        x_rfFollowing=row1['Local_X']-row['Local_X']-row['v_Length']/2-row1['v_Length']/2
                        y_rfFollowing=row1['Local_Y']-row['Local_Y']
                        if y_rfFollowing >0:
                            data_ngsim['id_rFollowing'][index]=row1['Vehicle_ID']
                            data_ngsim['x_rFollowing'][index]=x_rfFollowing
                            data_ngsim['y_rFollowing'][index]=y_rfFollowing
                        else:
                            data_ngsim['id_lFollowing'][index]=row1['Vehicle_ID']
                            data_ngsim['x_lFollowing'][index]=x_rfFollowing
                            data_ngsim['y_lFollowing'][index]=y_rfFollowing
                    else:
                        x_rfPrecceding=row['Local_X']-row1['Local_X']-row['v_Length']/2-row1['v_Length']/2
                        y_rfPrecceding=row1['Local_Y']-row['Local_Y']
                        if y_rfPrecceding >0:
                            data_ngsim['id_rPrecceding'][index]=row1['Vehicle_ID']
                            data_ngsim['x_rPrecceding'][index]=x_rfPrecceding
                            data_ngsim['y_rPrecceding'][index]=y_rfPrecceding
                        else:
                            data_ngsim['id_lPrecceding'][index]=row1['Vehicle_ID']
                            data_ngsim['x_lPrecceding'][index]=x_rfPrecceding
                            data_ngsim['y_lPrecceding'][index]=y_rfPrecceding
                   

#################### pour LSTM
                            
LSTM_data=data_ngsim[['Vehicle_ID','Local_X', 'Local_Y','v_Length', 'v_Width']].copy()                          

col_names = ['Vehicle_ID','Type_Car','Type_Pedestrian','Type_Motorcycle','Type_bicycle','Type_Bus','Type_Truck','Type_Escooter',
             'Frame_ID', 'Local_X', 'Local_Y','v_Vel_x','v_Vel_y','v_Length', 'v_Width',
             
             'exist_LeftPreceeding','LeftPreceeding_ID','lp_v_Length', 'lp_v_Width',
             'lpType_Car','lpType_Pedestrian','lpType_Motorcycle','lpType_bicycle','lpType_Bus','lpType_Truck','lpType_Escooter',
             'Local_X_LeftPreceeding','Local_Y_LeftPreceeding','v_X_Vel_LeftPreceeding','v_Y_Vel_LeftPreceeding',
             
             'exist_Preceeding','Preceeding_ID','p_v_Length', 'p_v_Width',
             'pType_Car','pType_Pedestrian','pType_Motorcycle','pType_bicycle','pType_Bus','pType_Truck','pType_Escooter',
             'Local_X_Preceeding','Local_Y_Preceeding','v_X_Vel_Preceeding','v_Y_Vel_Preceeding',
             
             'exist_RightPreceeding','RightPreceeding_ID','rp_v_Length', 'rp_v_Width',
             'rpType_Car','rpType_Pedestrian','rpType_Motorcycle','rpType_bicycle','rpType_Bus','rpType_Truck','rpType_Escooter',
             'Local_X_RightPreceeding','Local_Y_RightPreceeding','v_X_Vel_RightPreceeding','v_Y_Vel_RightPreceeding',
             
             'exist_LeftFollower','LeftFollower_ID','lf_v_Length', 'lf_v_Width',
             'lfType_Car','lfType_Pedestrian','lfType_Motorcycle','lfType_bicycle','lfType_Bus','lfType_Truck','lfType_Escooter',
             'Local_X_LeftFollower','Local_Y_LeftFollower','v_X_Vel_LeftFollower','v_Y_Vel_LeftFollower',
             
             'exist_Follower','Follower_ID','f_v_Length', 'f_v_Width',
             'fType_Car','fType_Pedestrian','fType_Motorcycle','fType_bicycle','fType_Bus','fType_Truck','fType_Escooter',
             'Local_X_Follower','Local_Y_Follower','v_X_Vel_Follower','v_Y_Vel_Follower',
             
             'exist_RightFollower','RightFollower_ID','rf_v_Length', 'rf_v_Width',
             'rfType_Car','rfType_Pedestrian','rfType_Motorcycle','rfType_bicycle','rfType_Bus','rfType_Truck','rfType_Escooter',
             'Local_X_RightFollower','Local_Y_RightFollower','v_X_Vel_RightFollower','v_Y_Vel_RightFollower',
             ]                   


data_ngsim.to_csv(r'YeltsinNGSIM.csv')





