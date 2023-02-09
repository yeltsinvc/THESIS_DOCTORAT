# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:15:08 2021

@author: valero
"""

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
"""
0:person
1:bicycle
2:car
3:motorbike
5:bus
6:train
7:truck

order 2-->7-->5-->3-->1-->0
"""

def plot_by_type(binsf,TimeStart,intervale,name_Sens,name_CountLine,dbs_by_type):
    fig, ax = plt.subplots()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Nombre des véhicules')
    ax.set_xlabel('Heure(HH:MM:SS)')
    ax.set_title('Débit du '+TimeStart.strftime("%m/%d/%Y")+': intervale ('+str(int(intervale/60))+ ' minutes)')
    #ax.set_xticks(flow[0].index)
    
    #ax.set_ylim(0, 100)
    i=0
    bottom=0
    for id_type in [2,7,5,3,1,0]:
        if id_type in dbs_by_type.index.get_level_values(0).unique():
                
            db_by_type=dbs_by_type.loc[id_type]
            if id_type==2:
                ax.bar(binsf[:-1], db_by_type.values, intervale, label='Car',bottom=bottom)
                bottom+=db_by_type
            
            else:
                if id_type==7:
                    ax.bar(binsf[:-1], db_by_type.values, intervale, label='Truck',bottom=bottom)
                    bottom+=db_by_type
                else:
                    if id_type==5:
                        ax.bar(binsf[:-1], db_by_type.values, intervale, label='Bus',bottom=bottom)
                        bottom+=db_by_type
                    else:
                        if id_type==3:
                            ax.bar(binsf[:-1], db_by_type.values, intervale, label='Moto',bottom=bottom)
                            bottom+=db_by_type
                        else:
                            if id_type==1:
                                ax.bar(binsf[:-1], db_by_type.values, intervale, label='Vélo',bottom=bottom)
                                bottom+=db_by_type
                            else:
                                if id_type==0:
                                    ax.bar(binsf[:-1], db_by_type.values, intervale, label='Person',bottom=bottom)
                                    bottom+=db_by_type
    ax.xaxis.set_ticks(binsf[:-1])
    labels=[(TimeStart+delta*i).strftime("%H:%M:%S")+"-"+(TimeStart+delta*(i+1)).strftime("%H:%M:%S") for i in range(len(binsf)-1)]
    ax.set_xticklabels(labels,rotation = 90)  
    ax.yaxis.set_major_locator(MultipleLocator(5))    
    ax.yaxis.set_minor_locator(MultipleLocator(1))                  
    fig.tight_layout()
    ax.legend()
    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)
    plt.show()
    #rects2 = ax.bar(0.5, women_means, width, label='Women')
    plt.savefig("L"+str(int(name_CountLine))+"S"+str(name_Sens)+".svg", format="svg")


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    conn = sqlite3.connect(db_file)
    

    return conn
fps=25.0

databaseFilename='data.db' 
conn=create_connection(databaseFilename)

cur = conn.cursor()
#cur.execute("SELECT * FROM positions")

cur.execute('SELECT OF.object_id, P.frame_number, OF.road_user_type, P.x_coordinate, P.y_coordinate, P.line_n from positions P, objects OF WHERE P.trajectory_id = OF.object_id'+' ORDER BY OF.object_id, P.frame_number')
data=pd.DataFrame(np.array(cur.fetchall()),columns=["ID_Veh","Frame","Type","Position_x","Position_y",'Count_Line' ])


#Determine the sens of the vehicule
direction=pd.DataFrame()
list_id_veh=data[data["Count_Line"]!=99]['ID_Veh']
trajectories_data = data[data['ID_Veh'].isin(list_id_veh.unique())]
fist_pos=trajectories_data.groupby(['ID_Veh'])['Position_x','Position_y'].first()
last_pos=trajectories_data.groupby(['ID_Veh'])['Position_x','Position_y'].last()
direction['Direction_x']=last_pos['Position_x']-fist_pos['Position_x']
direction['Direction_y']=last_pos['Position_y']-fist_pos['Position_y']
direction['Sens']=1
direction.Sens[(direction['Direction_x']>0)&(direction['Direction_y']<0)]=2


trajectories_data=trajectories_data.join(direction['Sens'],on='ID_Veh')

trajectories_data=trajectories_data[trajectories_data['Count_Line']!=99]
trajectories_data['Time']=trajectories_data['Frame']/fps
TimeStart=datetime.datetime(year=2020, month=1, day=31, hour=8, minute=0, second=0)
intervale=1*300
delta = datetime.timedelta(seconds=intervale)
minTime=0
maxTime=60*30
binst=np.array(range(minTime,int(maxTime/intervale)+1))*intervale

data_by_type=[]
sens=[]
countsLines=[]
data_by_Sens=trajectories_data.groupby(['Sens'])
for name_Sens,group_by_Sens in data_by_Sens:
    daba_by_CountLine=group_by_Sens.groupby(['Count_Line'])
    for name_CountLine,group_by_CountLine in daba_by_CountLine:
        sens.append(name_Sens)
        countsLines.append(name_CountLine)
        data_by_type.append(group_by_CountLine.groupby(['Type'])['Time'].value_counts(bins=binst).sort_index())
        plot_by_type(binst,TimeStart,intervale,name_Sens,name_CountLine,data_by_type[-1])

"""daba_by_CountLine=data.groupby(['Count_Line','Type'])
flow=[]
names=[]
for name,group in daba_by_CountLine:
    names.append(name)
    flow.append(group['Time'].value_counts(bins=bins).sort_index())

fig, ax = plt.subplots()
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Nombre des véhicules')
ax.set_xlabel('Heure(HH:MM:SS)')
ax.set_title('Débit du '+TimeStart.strftime("%m/%d/%Y")+': intervale ('+str(int(intervale/60))+ ' minutes)')
#ax.set_xticks(flow[0].index)
labels=[(TimeStart+delta*i).strftime("%H:%M:%S")+"-"+(TimeStart+delta*(i+1)).strftime("%H:%M:%S") for i in range(len(bins)-1)]
ax.set_xticklabels(labels,rotation = 90)
ax.legend()
#ax.set_ylim(0, 100)
i=0
for line,veh_type in names:
    if veh_type==2:
        ax.bar(bins[:-1], flow[0].values, intervale, label='Car')
    elif veh_type==0:
        ax.bar(bins[:-1], flow[i].values, intervale, label='Pedestrian',bottom=flow[i].values)
    i+=1
fig.tight_layout()
ax.legend()
#ax.bar_label(rects1, padding=3)
#ax.bar_label(rects2, padding=3)
plt.show()
#rects2 = ax.bar(0.5, women_means, width, label='Women')
plt.savefig("test.svg", format="svg")"""

 