#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import load_data
data_path = os.path.join('data','demo.p')
data = load_data(data_path)

velodyne = data["velodyne"]
reflectance = velodyne[:,3]


# In[13]:


#shift points to positive 
points = velodyne[:,0:3]
reflectance = velodyne[:,3]
points = points - points.min(axis = 0)

res = 0.2
range_velo = points[:,0:2].max(axis = 0)
grid = np.zeros((int(range_velo[0]/res) + 1, int(range_velo[1]/res) + 1))

points = (points/res).astype(int)

for i in range(len(points)):
    refl = reflectance[i]
    curr_bin = [points[i,0],points[i,1]]
    if refl >= grid[curr_bin[0],curr_bin[1]]:
            grid[curr_bin[0],curr_bin[1]] = refl
plt.imshow(grid, cmap='gray')
plt.show


# In[ ]:




