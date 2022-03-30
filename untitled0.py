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

image1 = Image.open('Datasets/BSDS500/Control/data221.jpg')
image1 = image1.convert('L')

image2 = Image.open('Datasets/BSDS500/Control/data229.jpg')
image2 = image2.convert('L')


#%%

des1 = fh.SIFT(image1)
des2 = fh.SIFT(image2)


des1 = fh.ImageDescriptors(des1, 'L2')
des2 = fh.ImageDescriptors(des2, 'L2')


res = des1.matches(des2, threshold=0.3)

test = fh.MATCHERS[des1.matcher].match(des1.descriptors, des2.descriptors)

distances = []

for a in test:
    distances.append(a.distance)

#%%
from tqdm import tqdm
import os

img1 = np.array(image1)
img2 = np.array(image2)

detector = cv2.ORB_create(nfeatures=20000000)
extractor = cv2.xfeatures2d.DAISY_create()
path = 'Datasets/BSDS500/Control_attacks/'

imgs = [path + file for file in os.listdir(path)]
N = []

for img in tqdm(imgs):
    img = np.array(Image.open(img).convert('L'))
    kps = detector.detect(img)
    _, des = extractor.compute(img, kps)
    if des is not None:
        N.append(len(des))

#kp, descriptors = extractor.compute(img1, kps)

#kp, descriptors = tot.detectAndCompute(img, None)

