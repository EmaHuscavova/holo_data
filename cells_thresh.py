# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:26:35 2023

@author: emahu
"""

import numpy as np
from skimage import io as skio
import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature
from skimage import morphology

img = skio.imread("C:/Users/emahu/Documents/SKOLA/holoDATA/data/00001_PC3_img.tif")

img_gaus = filters.gaussian(img, sigma=3)
img_med = filters.median(img_gaus, footprint=np.ones([3,3]))
mask_max = feature.peak_local_max(img_med, min_distance=10, threshold_abs=0.1, indices = True)



w,h = img.shape
threshold = 0.1

res = img >= threshold


fig = plt.figure(figsize=(20, 15))

fig.add_subplot(1,2,1)
plt.imshow(morphology.remove_small_objects(res, min_size=200))
plt.axis('off')

fig.add_subplot(1,2,2)
plt.imshow(morphology.remove_small_objects(res, min_size=30))
plt.axis('off')



row, col = 2,3
fig = plt.figure(figsize=(17, 12))

fig.add_subplot(row,col,1)
  
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Img")
  
# Adds a subplot at the 2nd position
fig.add_subplot(row,col,2)
  
# showing image
plt.imshow(img_med)
plt.axis('off')
plt.title("Filter")
  
# Adds a subplot at the 3rd position
fig.add_subplot(row,col,3)
  
# showing image
plt.imshow(res)
plt.axis('off')
plt.title("Threshold")

fig.add_subplot(row,col,4)
  
# showing image
plt.imshow(feature.peak_local_max(img_med, min_distance=10, threshold_abs=0.1, indices = False))
plt.axis('off')
plt.title("max")

