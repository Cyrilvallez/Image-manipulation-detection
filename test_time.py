#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 08:23:30 2022

@author: cyrilvallez
"""

import generator
import time
import os
from PIL import Image, ImageFilter
import numpy as np
import torch
import torchvision.transforms.functional as F
from skimage import util, transform, filters
import cv2
import scipy.ndimage as ndi

path = 'test_hashing/BSDS500/Control/'
images = os.listdir(path)

median_kernels=(3, 5, 7)
gaussian_kernels=(3, 5, 7)
rotation_angles=(5, 10, 20, 40, 60)

t0 = time.time()

for a in images[0:20]:
    
    image = Image.open(path+a)
    size = image.size
    
    #test = generator.text_attack(path+a, **generator.ATTACK_PARAMETERS)
    for angle in rotation_angles:
        rotated = F.rotate(image, angle=angle, interpolation=F.InterpolationMode.BICUBIC,
                 expand=True)
        rotated = rotated.resize(size, resample=Image.BICUBIC)
    
dt = time.time() - t0

#t0 = time.time()

#for a in range(10):
    
#    test2 = generator.cropping_attack_bis(path, cropping_percentages, resize_cropping=True)
    
#dt2 = time.time() - t0

#test = list(test.values())
#test2 = list(test2.values())



print(f'Time needed first : {dt/20} s')
#print(f'Time needed now : {dt2/10} s')