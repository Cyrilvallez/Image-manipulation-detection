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
from hashing import featurehash as fh




#%%

image1 = Image.open('Datasets/BSDS500/Control/data221.jpg')
image1 = image1.convert('L')

image2 = Image.open('Datasets/BSDS500/Control/data229.jpg')
image2 = image2.convert('L')

des1 = fh.ORB(image1, device='cpu')
des2 = fh.ORB(image2, device='cpu')

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

des1 = fh.ImageDescriptors(des1, matcher)
des2 = fh.ImageDescriptors(des2, matcher)


res = des1.matches(des2, threshold=0.3)
