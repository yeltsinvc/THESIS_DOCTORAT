# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:59:45 2020

@author: GRETTIA
"""


import numpy as np
import pandas as pd
import math

global X_train_us101
global y_train_us101
global list_vehicle_id
X_train_us101 = []
y_train_us101 = []  
list_vehicle_id = []
#objet that represent the lane
class Geometry:
    def __init__(self,name):
        self.name=name
        self.isLane=[]


#funtion to create csv with the geometry(border of lanes) -----> yvc
def getGeometry(indDirectory,img_world,ortho_px_to_meter):
    from geopandas import GeoSeries
    from shapely.geometry import Polygon
    
    
    dataframe=pd.DataFrame()
    
    def on_mouse(event, x, y, buttons,param):
        FINAL_LINE_COLOR = (255, 255, 255)
        
        global user_param
        
           
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            if user_param is None:
                
                user_param=[]
            else:
                 user_param.append((x, y))
            
            print(user_param)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            #print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            user_param.append((x, y))
            cv2.polylines(img, np.array([user_param]), True, FINAL_LINE_COLOR, 1)
            n_lane=input("Enter ID of lane: ")
            
            dataframe['Lane '+n_lane]=[Polygon(np.array(user_param)*ortho_px_to_meter*12)]#####modifi
            dataframe.to_csv(indDirectory + "\geometry.csv")
            user_param=[]
            print(user_param)
    
    
    
    import cv2
    global user_param
    
    #img = cv2.imread("world.jpg")
    img = cv2.imread(indDirectory+"/"+img_world)
    cv2.namedWindow("figure")
    param=[]
    cv2.setMouseCallback("figure", on_mouse)
    user_param=[] 
    while True:
        # both windows are displaying the same img
        
        cv2.imshow("figure", img)
        param=[]
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()









def calculate_pente(data_ngsim):
    data_vehicle_by_lane=data_ngsim.groupby(['Lane_ID','Vehicle_ID'])
    vector_x=[]
    vector_y=[]
    for group_by_lane in data_vehicle_by_lane.groups:
        data_vehicle_lane=data_vehicle_by_lane.get_group(group_by_lane)
        idx_frame_min=data_vehicle_lane['Frame_ID'].idxmin()
        idx_frame_max=data_vehicle_lane['Frame_ID'].idxmax()
        vector_x.append(data_vehicle_lane['xCenter'][idx_frame_max]-data_vehicle_lane['xCenter'][idx_frame_min])
        vector_y.append(data_vehicle_lane['yCenter'][idx_frame_max]-data_vehicle_lane['yCenter'][idx_frame_min])
    return [np.mean(vector_x),np.mean(vector_y)]
        

def rotate_trayectories(vector_trayectories,data_ngsim):
    '''
    x'=xcos(angle)+ysin(angle)
    y'=-xsin(angle)+ycos(angle)
    '''
    angle=np.arctan2(vector_trayectories[1],vector_trayectories[0])
    data_ngsim['Local_X']=data_ngsim['xCenter']*math.cos(angle)+data_ngsim['yCenter']*math.sin(angle)
    data_ngsim['Local_Y']=data_ngsim['xCenter']*-math.sin(angle)+data_ngsim['yCenter']*math.cos(angle)
    return data_ngsim




### copy some funtions of run_track_visualization
    
def read_all_recordings_from_csv(base_path):
    """
    This methods reads the tracks and meta information for all recordings given the path of the inD dataset.
    :param base_path: Directory containing all csv files of the inD dataset
    :return: a tuple of tracks, static track info and recording meta info
    """
    track_file = base_path + r"/07_tracks.csv"
    static_tracks_file = base_path + r"/07_tracksMeta.csv"
    recording_meta_file = base_path + r"/07_recordingMeta.csv"
    
    tracks, static_info, meta_info=read_from_csv(track_file, static_tracks_file, recording_meta_file)
    

    return tracks, static_info, meta_info


def read_from_csv(track_file, static_tracks_file, recordings_meta_file):
    """
    This method reads tracks including meta data for a single recording from csv files.

    :param track_file: The input path for the tracks csv file.
    :param static_tracks_file: The input path for the static tracks csv file.
    :param recordings_meta_file: The input path for the recording meta csv file.
    :return: tracks, static track info and recording info
    """
    static_info = read_static_info(static_tracks_file)
    meta_info = read_meta_info(recordings_meta_file)
    tracks = read_tracks(track_file)
    return tracks, static_info, meta_info


def read_tracks(track_file):
    """
    This method reads the static info file from highD data.

    :param static_tracks_file: the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    return pd.read_csv(track_file)#.to_dict(orient="records")

def read_static_info(static_tracks_file):
    """
    This method reads the static info file from highD data.

    :param static_tracks_file: the input path for the static csv file.
    :return: the static dictionary - the key is the track_id and the value is the corresponding data for this track
    """
    return pd.read_csv(static_tracks_file)#.to_dict(orient="records")


def read_meta_info(recordings_meta_file):
    """
    This method reads the recording info file from ind data.

    :param recordings_meta_file: the path for the recording meta csv file.
    :return: the meta dictionary
    """
    return pd.read_csv(recordings_meta_file)#.to_dict(orient="records")[0]

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
        
        for i in range(50, len(data_vehicle)):
            temp=(data_vehicle[i-50:i+1, :-1]).copy()
            temp[:,0]=temp[:,0]-temp[0,0]
            temp[:,1]=temp[:,1]-temp[0,1]
            X_train_us101.append(temp[:50, :-1])
            y_train_us101.append(temp[50, 0:2])
            list_vehicle_id.append(data_vehicle[i, 54])
    return X_train_us101,y_train_us101,list_vehicle_id

def readCSVBycicle(dirFolder):
    dataset=pd.read_csv(dirFolder)
    dataset=dataset[dataset['class'] =='bicycle']
    return dataset