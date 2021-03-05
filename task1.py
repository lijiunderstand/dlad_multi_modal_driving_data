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

plt.scatter(velodyne[:,0],velodyne[:,1],c =velodyne[:,3], s=0.1)

