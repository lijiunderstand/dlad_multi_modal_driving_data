# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 14:10:06 2021

@author: Jason & Mirlan
"""
import data_utils as du
import cv2
import numpy as np

'''Switch between test frame and all frames, for all frames you need a 'Result' folder for it to run properly'''
ONLY_DO_TEST_FRAME = True; 
 
f_velodyne = 10
test_frame = 37

[R_velo2cam, t_velo2cam] = du.calib_velo2cam("data/problem_4/calib_velo_to_cam.txt")
cam0tocam2 = du.calib_cam2cam("data/problem_4/calib_cam_to_cam.txt", '02')


if ONLY_DO_TEST_FRAME:
    range_use = [test_frame]#[test_frame,44,312,428]#range(test_frame,test_frame+1)
else:
    range_use = range(430)

'''they dont give us a function to get R and t imu2velo for some reason -> velo2cam should work though'''
[R_imu2velo, t_imu2velo] = du.calib_velo2cam("data/problem_4/calib_imu_to_velo.txt")

def project_points(velopoints, image):
    '''filter points'''
    points_filtered = velopoints[velopoints[:,0]>0]
    points_filtered_r = np.sqrt(points_filtered[:,0]**2 + points_filtered[:,1]**2 + points_filtered[:,2]**2)
    
    points_colors = du.depth_color(points_filtered_r)
    points_colors = points_colors * (127.5/np.max(points_colors))
    
    '''project velopoints onto image plane'''
    points_cam0 = R_velo2cam @ points_filtered.T + t_velo2cam  
    points_uv_lambda = cam0tocam2[:,:3] @ points_cam0
    points_uv = points_uv_lambda / points_uv_lambda[2,:]
    return du.print_projection_plt(points_uv, points_colors, image) 
    
for ind in range_use: 
    
    '''import all timestamps for this frame'''
    time_img2 = du.compute_timestamps("data/problem_4/image_02/timestamps.txt", ind)
    time_oxts = du.compute_timestamps("data/problem_4/oxts/timestamps.txt", ind)
    time_velo_start = du.compute_timestamps("data/problem_4/velodyne_points/timestamps_start.txt", ind)
    time_velo_end = du.compute_timestamps("data/problem_4/velodyne_points/timestamps_end.txt", ind)
    time_velo = du.compute_timestamps("data/problem_4/velodyne_points/timestamps.txt", ind)
    
    '''Data for this frame'''
    velodyne_points = du.load_from_bin("data/problem_4/velodyne_points/data/%010d.bin" % ind)
    img2 = cv2.imread("data/problem_4/image_02/data/%010d.png" % ind) 
    oxts_velocity = du.load_oxts_velocity("data/problem_4/oxts/data/%010d.txt" %ind)
    oxts_angular_rate = du.load_oxts_angular_rate("data/problem_4/oxts/data/%010d.txt" %ind)
    
    velo_velocity = R_imu2velo @ oxts_velocity  #can i do this? with ang rate independent?
    velo_angular_rate = R_imu2velo @ oxts_angular_rate
    
    '''project points for befor picture'''
    img_before = project_points(velodyne_points, img2)
      
    '''rectify the velodyne points
    Idea: for every slice of phi in the velo points we need to calculate when 
    it happend and then figure out what the R and t for those points are to transform them to the right point'''
    velodyne_point_phi = np.arctan2(velodyne_points[:,1], velodyne_points[:,0])
    f_velodyne = 1/(time_velo_end - time_velo_start)
    for p in range(velodyne_points.shape[0]):
        phi = velodyne_point_phi[p]
        
        #phi_vel_start = -(f_velodyne*(2*np.pi))*(time_img2 - time_velo_start) #angle when the velodyne starts recording
        #dt_since_start = (phi - phi_vel_start)/(f_velodyne*(2*np.pi)) #time it took velodyne to reach current point from start
        
        
        #dt_since_start = time_revolution * (phi/(2*np.pi)) #!!!! THE BUG IS HERE, this assumes that we started with the velo facing forward, that is wrong, need to figure out where we are by sorting
        #dt_camtrigger_to_pointrecorded = (time_img2 - time_velo_start - dt_since_start)
        phi_vel_start = (f_velodyne*(2*np.pi))*(time_velo - time_velo_start)
        if (phi_vel_start <= np.pi/2 or phi_vel_start >= (3/2) *np.pi):
            print('assumption doesnt hold')
            
        dt_camtrigger_to_pointrecorded = -(phi / (f_velodyne*(2*np.pi))) # - (time_img2 - time_velo)
        
        omega = velo_angular_rate[2] * dt_camtrigger_to_pointrecorded
        R_z = np.array([[np.cos(omega), -np.sin(omega), 0],
                        [np.sin(omega), np.cos(omega), 0],
                        [0,             0,             1]])
        point_rectified = R_z @ velodyne_points[p,:] + dt_camtrigger_to_pointrecorded * velo_velocity
        velodyne_points[p,:] = point_rectified
    
    img_after = project_points(velodyne_points, img2) 
    if ONLY_DO_TEST_FRAME:
        
        cv2.imwrite('Before_Correction_%010d.png' %ind, img_before)
        cv2.imshow("Before Correction",img_before)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        
        cv2.imwrite('After_Correction_%010d.png' %ind, img_after)
        cv2.imshow("After Correction",img_after)
        cv2.waitKey(2000) 
        cv2.destroyAllWindows()
    else:
        cv2.imwrite('Result/After_correction_%010d.png' %ind, img_after)
    
        
        
