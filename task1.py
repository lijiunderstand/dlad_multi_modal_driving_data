# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 14:09:24 2021

@author: Jason
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import data
velodyne = data["velodyne"]
print(velodyne[:,0])

""" matplotlib version """
plt.scatter(velodyne[:,0],velodyne[:,1],c = velodyne[:,3], s=0.1)

""" opencv version """
range_lidar = 120
res = 0.2

img_birdseye = np.zeros((int(range_lidar/res),int(range_lidar/res))) # documentation states rage of 120m in all directions
for point in range(len(velodyne)):
    x = int(velodyne[point,0]/res)+range_lidar
    y = int(velodyne[point,1]/res)+range_lidar
    refl = velodyne[point,3]
    
    refl_curr = img_birdseye[x,y]
    if refl_curr < refl:
        img_birdseye[x,y] =  refl
        
img_birdseye_90 = cv2.rotate(img_birdseye, cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow('Birdseye_View', img_birdseye_90)
cv2.waitKey(10000)
cv2.destroyAllWindows()
#cv2.imwrite('messigray.png',img)