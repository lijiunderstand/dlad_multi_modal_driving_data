# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:15:00 2021

@author: Jason
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import data


"""load in Image 2"""

img_cam_2 = data["image_2"]
cv2.imshow('Image_2',np.array(img_cam_2, dtype=np.uint8)) #colors are still off
cv2.waitKey(1000)
cv2.destroyAllWindows()
cv2.imwrite('img_cam_2_raw.png',img_cam_2)

"""-----------------------------------
project 3D points on to Image Plane
-----------------------------------"""

velodyne = data["velodyne"]
K_cam2 = data["K_cam2"]
T_cam2_velo = data["T_cam2_velo"]
P_rect_20 = data["P_rect_20"]
sem_label = data["sem_label"]
color_map = data["color_map"]

"""filter points"""
# Idea, just roughly filter points behind the car, dont mind points outside the field of view
# this should insure that we dont have wrong projections
# concatonate the points and the sem label together, filter and then seperate again
velo_sem_combo = np.concatenate((velodyne, sem_label), axis = 1)
velo_sem_combo_filtered = velo_sem_combo[velo_sem_combo[:,0]>0]
velo_filtered = velo_sem_combo_filtered[:,0:4]
sem_filtered = velo_sem_combo_filtered[:,4]

"""indexing dict with for loop, there must be a better way"""
colors = np.zeros((sem_filtered.size,3))
i = 0 
for key in sem_filtered:
    colors[i,:] = color_map[key]
    i = i + 1
    
#colors = colors.tolist()
"""project points"""
points_C = T_cam2_velo @ velo_filtered.T # @ is shorthand for np.matmult
uv_img_cords =  K_cam2 @ points_C[0:3,:] / points_C[2,:]


plt.imshow(img_cam_2)
plt.scatter(uv_img_cords[0,:], uv_img_cords[1,:], s = 1, marker = '.' \
            ,edgecolors = 'none', c = colors/255.0)
plt.ylim(376,0)
plt.xlim(0,1241) 

plt.savefig("Velodyne_Projected.png", dpi= 2000)


plt.show()