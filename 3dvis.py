# Deep Learning for Autonomous Driving
# Material for Problem 2 of Project 1
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
import os
from load_data import load_data
vispy.use('PyQt5')

class Visualizer():
    def __init__(self):
        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.grid = self.canvas.central_widget.add_grid()
        self.view = vispy.scene.widgets.ViewBox(border_color='white',
                        parent=self.canvas.scene)
        self.grid.add_widget(self.view, 0, 0)

        # Point Cloud Visualizer
        self.sem_vis = visuals.Markers()
        self.view.camera = vispy.scene.cameras.TurntableCamera(up='z', azimuth=90)
        self.view.add(self.sem_vis)
        visuals.XYZAxis(parent=self.view.scene)
        
        # Object Detection Visualizer
        self.obj_vis = visuals.Line()
        self.view.add(self.obj_vis)
        self.connect = np.asarray([[0,1],[0,3],[0,4],
                                   [2,1],[2,3],[2,6],
                                   [5,1],[5,4],[5,6],
                                   [7,3],[7,4],[7,6]])

    def update(self, points, colors):
        '''
        :param points: point cloud data
                        shape (N, 3)          
        Task 2: Change this function such that each point
        is colored depending on its semantic label
        '''
        self.sem_vis.set_data(points, size=3, face_color=colors)
    
    def update_boxes(self, corners):
        '''
        :param corners: corners of the bounding boxes
                        shape (N, 8, 3) for N boxes
        (8, 3) array of vertices for the 3D box in
        following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        If you plan to use a different order, you can
        change self.connect accordingly.
        '''
        for i in range(corners.shape[0]):
            connect = np.concatenate((connect, self.connect+8*i), axis=0) \
                      if i>0 else self.connect
        self.obj_vis.set_data(corners.reshape(-1,3),
                              connect=connect,
                              width=2,
                              color=[0,1,0,1])

if __name__ == '__main__':
    data = load_data('data/demo.p') # Change to data.p for your final submission 
    visualizer = Visualizer()
    color_map = data["color_map"]
    sem_label = data["sem_label"]
    
    colors = np.zeros((sem_label.size,3))
    i = 0 
    for key in sem_label:
        colors[i,:] = color_map[int(key)]
        i = i + 1
    
    visualizer.update(data['velodyne'][:,:3], colors/255.0)
    '''
    Task 2: Compute all bounding box corners from given
    annotations. You can visualize the bounding boxes using
    visualizer.update_boxes(corners)
    '''
    
    objects = data["objects"] #this has to be one of the dumbest data structures...
    T_cam0_velo = data["T_cam0_velo"]
    
    '''invert T_cam0_velo to get T_velo_cam0'''
    C_0v = T_cam0_velo[:3,:3]
    t_0v = T_cam0_velo[:3,3]
    T_velo_cam0 = np.concatenate((np.concatenate((C_0v.T, (- C_0v.T @ t_0v).reshape(-1,1)), axis = 1), np.array([0,0,0,1]).reshape(-1,1).T ), axis=0)
    
    '''build array with all of the corners for the function'''
    num_obj = len(objects)
    corners_velo_all = np.zeros((num_obj,8,3))
    j=0
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
        
        '''Corners need to be transformed to velodyne coordinates'''
        corners_velo = (T_velo_cam0 @ np.concatenate((corners_0, np.ones((8,1))), axis =1).T).T
        corners_velo_all[j,:,:]= corners_velo[:,:3]
        j = j+1
    
    visualizer.update_boxes(corners_velo_all)
    vispy.app.run()




