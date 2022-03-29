#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:38:56 2022

@author: cyrilvallez
"""

from PIL import Image
import skimage.feature as feature
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np

image = Image.open('Datasets/BSDS500/Control/data221.jpg')
image = image.convert('L')

descs, descs_img = feature.daisy(image, step=180, radius=58, rings=2, histograms=6,
                                 orientations=8, visualize=True)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(descs_img)
descs_num = descs.shape[0] * descs.shape[1]
ax.set_title('%i DAISY descriptors extracted:' % descs_num)
plt.show()


#%%

image = Image.open('Datasets/BSDS500/Control/data221.jpg')
image = image.convert('L')
t0 = time.time()

for i in range(10):
    sift = feature.ORB()
    sift.detect_and_extract(image)
    descriptors_sk = sift.descriptors

dt_sk = (time.time() - t0)/10

t0 = time.time()
for i in range(10):
    sift = cv2.ORB_create()
    keypoints, descriptors = sift.detectAndCompute(np.array(image), None)
    
dt_cv = (time.time() - t0)/10