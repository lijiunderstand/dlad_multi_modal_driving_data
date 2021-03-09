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

"""filter points"""
# Idea, just roughly filter points behind the car, dont mind points outside the field of view
# this should insure that we dont have wrong projections
#velodyne_filted = np.where(velodyne )
#velo_filtered = np.copy(velodyne)
#velo_filtered[velo_filtered[:,0]<0] = np.zeros((1,4))
#velo_filtered = velodyne[~np.all(velo_filtered == 0, axis = 1)]
velo_filtered = velodyne[velodyne[:,0]>0]
points_C = T_cam2_velo @ velodyne.T # @ is shorthand for np.matmult
uv_img_cords =  K_cam2 @ points_C[0:3,:] / points_C[2,:]