# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 17:15:00 2021

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

colors[:,[0,2]] = colors[:,[2,0]] #switch from BGR to RGB

"""project points"""
points_2 = T_cam2_velo @ velo_filtered.T # @ is shorthand for np.matmult
uv_img_cords =  K_cam2 @ points_2[0:3,:] / points_2[2,:]

"""Scatter plot the points onto the raw Image"""
plt.imshow(img_cam_2)
plt.scatter(uv_img_cords[0,:], uv_img_cords[1,:], s = 1, marker = '.' \
            ,edgecolors = 'none', c = colors/255.0)
plt.ylim(376,0)
plt.xlim(0,1241)

plt.savefig("Velodyne_Projected.png", dpi= 2000)


"""--------------------------
Do 3D bounding Boxes for Cars
--------------------------"""

objects = data["objects"] #this has to be one of the dumbest data structures...

for obj in objects:
    location3D_0 = obj[11:14] #3d location in frame 0
    obj_size = obj[8:11]
    rot_y = obj[14]

    """ find Location of corners """
    x = obj_size[2]/2
    y = obj_size[0]
    z = obj_size[1]/2
    # !!!! WEIRD ASS COORDINATES FOR CAM 0
    corners_car_frame = np.array([[x, -y, z],
                                  [-x, -y, z],
                                  [-x, -y, -z],
                                  [x, -y, -z],
                                  [x, 0, z],
                                  [-x, 0, z],
                                  [-x, 0, -z],
                                  [x, 0, -z]])
    rot = np.array([[np.cos(rot_y), 0, np.sin(rot_y)],
                    [0, 1, 0],
                    [-np.sin(rot_y), 0, np.cos(rot_y)]])
    corners_0 = (rot @ corners_car_frame.T).T + location3D_0


    """ Project corners and lines onto Image (with some nice boilerplate code) """
    color_box = np.array(list(np.random.choice(range(256), size=3))) /255.0
    corners_uv2_lambda = P_rect_20 @ np.concatenate((corners_0 , np.ones((8,1))), axis =1).T 
    corners_uv2 = corners_uv2_lambda / corners_uv2_lambda[2,:]
    plt.scatter(corners_uv2[0,:], corners_uv2[1,:], s=0.5, color = color_box)
    plt.plot(corners_uv2[0,0:4], corners_uv2[1,0:4] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[0,3]], corners_uv2[1,[0,3]] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,4:8], corners_uv2[1,4:8] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[4,7]], corners_uv2[1,[4,7]] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[0,4]], corners_uv2[1,[0,4]] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[1,5]], corners_uv2[1,[1,5]] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[2,6]], corners_uv2[1,[2,6]] , linewidth=0.5, color = color_box)
    plt.plot(corners_uv2[0,[3,7]], corners_uv2[1,[3,7]] , linewidth=0.5, color = color_box)


plt.savefig("Velodyne_Projected_3DBox.png", dpi= 2000)
plt.show()
