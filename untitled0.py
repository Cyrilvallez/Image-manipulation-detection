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




#%%

image = Image.open('Datasets/BSDS500/Control/data221.jpg')
image = image.convert('L')
t0 = time.time()

for i in range(10):
    orb = feature.ORB()
    orb.detect_and_extract(image)
    descriptors_sk = orb.descriptors

dt_sk = (time.time() - t0)/10

t0 = time.time()
for i in range(10):
    orb = cv2.ORB_create()
    _, descriptors_cv = orb.detectAndCompute(np.array(image), None)
    
dt_cv = (time.time() - t0)/10


#%%

test = cv2.imread('Datasets/BSDS500/Control/data221.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

#%%
import time
from tqdm import tqdm

def array_of_bytes_to_bits(array):
    
    out = []
    
    for byte in array:
        bits = [True if digit=='1' else False for digit in bin(byte)[2:]]
        out.extend(bits)
        
    return out
        
t0 = time.time()
N = 100000
for i in tqdm(range(N)):
    foo = array_of_bytes_to_bits(array)
    
dt = (time.time() - t0)/N

print(f'\n{dt:.3e} s')