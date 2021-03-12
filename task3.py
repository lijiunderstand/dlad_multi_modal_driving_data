# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:39:05 2021

@author: Jason and Mirlan
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import data

"""load in Image 2"""

img_cam_2 = data["image_2"]

"""-----------------------------------
project 3D points on to Image Plane
-----------------------------------"""

velodyne = data["velodyne"]
K_cam2 = data["K_cam2"]
T_cam2_velo = data["T_cam2_velo"]
ang_res = 0.4 * (np.pi/180)

'''filter points'''
velo_filtered = velodyne[velodyne[:,0]>0]

'''to seperate the points, I will calculate their position into spherical coordinates (ISO convention)'''
r = np.sqrt(velo_filtered[:,0]**2 + velo_filtered[:,1]**2 + velo_filtered[:,2]**2)
theta = np.arccos(velo_filtered[:,2]/r)
phi = np.arctan2(velo_filtered[:,1], velo_filtered[:,0])

#velo_filtered_spherical = np.concatenate((r.reshape(-1,1), theta.reshape(-1,1), phi.reshape(-1,1)), axis=1)

theta_colors =np.floor(((theta- np.min(theta))/ (np.max(theta)- np.min(theta)))*63)%4

"""project points"""
points_2 = T_cam2_velo @ velo_filtered.T # @ is shorthand for np.matmult
uv_img_cords =  K_cam2 @ points_2[0:3,:] / points_2[2,:]

"""Scatter plot the points onto the raw Image"""
plt.imshow(img_cam_2)
plt.scatter(uv_img_cords[0,:], uv_img_cords[1,:], s = 1, marker = '.' \
            ,edgecolors = 'none', c = theta_colors/3 ,cmap = 'Accent')
plt.ylim(376,0)
plt.xlim(0,1241) 

plt.savefig("Velodyne_Projected_Laser_ID.png", dpi= 2000)