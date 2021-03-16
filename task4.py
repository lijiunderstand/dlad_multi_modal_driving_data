# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:10:06 2021

@author: Jason & Mirlan
"""
import data_utils as du
import cv2
import numpy as np


test_frame = 37
[R_velo2cam, t_velo2cam] = du.calib_velo2cam("data/problem_4/calib_velo_to_cam.txt")
cam0tocam2 = du.calib_cam2cam("data/problem_4/calib_cam_to_cam.txt", '02')

for ind in range(test_frame,test_frame+1): #replace with range(430) when done testing
    
    '''import all timestamps for this frame'''
    time_img2 = du.compute_timestamps("data/problem_4/image_02/timestamps.txt", ind)
    time_oxts = du.compute_timestamps("data/problem_4/oxts/timestamps.txt", ind)
    time_velo_start = du.compute_timestamps("data/problem_4/velodyne_points/timestamps_start.txt", ind)
    time_velo_end = du.compute_timestamps("data/problem_4/velodyne_points/timestamps_end.txt", ind)
    time_velo = du.compute_timestamps("data/problem_4/velodyne_points/timestamps.txt", ind)
    
    '''Data for this frame'''
    velodyne_points = du.load_from_bin("data/problem_4/velodyne_points/data/%010d.bin" % ind)
    img2 = cv2.imread("data/problem_4/image_02/data/%010d.png" % ind) 
    
    '''show unrectivied points'''
    velodyne_dist = velodyne_points[velodyne_points[:,0]>0]
    velodyne_dist_r = np.sqrt(velodyne_dist[:,0]**2 + velodyne_dist[:,1]**2 + velodyne_dist[:,2]**2)
    
    velodyne_dist_colors = du.depth_color(velodyne_dist_r)
    #velodyne_dist_colors = np.zeros(np.shape(velodyne_dist_colors)) #only for test purpose, to remove
    
    velodyne_dist_cam0 = R_velo2cam @ velodyne_dist.T + t_velo2cam  
    velodyne_dist_uv_lambda = cam0tocam2[:,:3] @ velodyne_dist_cam0
    velodyne_dist_uv = velodyne_dist_uv_lambda / velodyne_dist_uv_lambda[2,:]
    img_before = du.print_projection_plt(velodyne_dist_uv, velodyne_dist_colors, img2)
    cv2.imshow("Befor Correction",img_before)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
