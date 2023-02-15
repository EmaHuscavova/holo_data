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
from skimage import segmentation
from skimage import measure


img = skio.imread("C:/Users/emahu/Documents/SKOLA/holoDATA/data/00001_PC3_img.tif")

img_gaus = filters.gaussian(img, sigma=3)
img_med = filters.median(img_gaus, footprint=np.ones([3,3]))


############# threshold
w,h = img.shape
threshold = 0.1

img_thr = img >= threshold
img_so = morphology.remove_small_objects(img_thr, min_size=100)
img_sh = morphology.remove_small_holes(img_so, area_threshold=100)
img_morph = morphology.area_opening(img_sh, area_threshold=100)


fig = plt.figure(figsize=(17, 12))

fig.add_subplot(1,2,1)
                                                                                                                                                  
# showing image
plt.imshow(morphology.area_opening(img_sh, area_threshold=100))
plt.axis('off')
plt.title("area_opening")
  
# Adds a subplot at the 2nd position
fig.add_subplot(1,2,2)
  
# showing image
plt.imshow(morphology.opening(img_sh))
plt.axis('off')
plt.title("opening")

#############################

local_max = feature.peak_local_max(img_med, min_distance=10, threshold_abs=0.1, indices = False)

labels = measure.label(local_max) 
img_seg = segmentation.watershed(-img_med, markers=labels, mask=img_morph) # ako masku pouzivam binarny obraz vytvoreny thresholdom
plt.imshow(segmentation.mark_boundaries(img_med, img_seg))



h_max = morphology.h_maxima(img_med, h=0.1)

labels2 = measure.label(h_max) 
img_seg2 = segmentation.watershed(-img_med, markers=labels2, mask=img_morph)
plt.imshow(segmentation.mark_boundaries(img_med, img_seg2))

# kombinujem obe maxima aby som mohla nastavovat vsetky parametre: h, distance, threshold
labels_res = measure.label(local_max*h_max)
seg_res = segmentation.watershed(-img_med, markers=labels_res, mask=img_morph)
plt.imshow(segmentation.mark_boundaries(img_med, seg_res))


row, col = 1,3
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
plt.imshow(segmentation.mark_boundaries(img_med, seg_res))
plt.axis('off')
plt.title("Segmentation")


