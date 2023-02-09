# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:27:31 2021

@author: valero
"""
import cv2
import numpy as np

from . import utilsCounterTraffic

        
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
        
            


def getCounterLines(im0):
    img = cv2.imread(im0)
    polygon_mask = np.zeros((720, 1024, 3), dtype=np.uint8)
    cv2.namedWindow('videoframe')
    param=[]
    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
             print ('Start Mouse Position: '+str(x)+', '+str(y))
             start=(x,y)
             params.append([start])
    
        elif event == cv2.EVENT_LBUTTONUP:
            print ('End Mouse Position: '+str(x)+', '+str(y))
            params[-1].append((x,y))
            cv2.line(img,param[-1][0],param[-1][1],color=(200,200,200),thickness=8)
            
            cv2.imshow('videoframe',img)
            
           
    cv2.imshow('videoframe',img)
    cv2.setMouseCallback('videoframe', on_mouse,param)
    
    while(True):                
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break    
        
    cv2.destroyAllWindows()
    for star,fin in param:
        cv2.line(polygon_mask,star,fin,color=(200,200,200),thickness=8)
    
    counter_lines=[utilsCounterTraffic.CounterLine(i) for i in param]
    
    return counter_lines,polygon_mask