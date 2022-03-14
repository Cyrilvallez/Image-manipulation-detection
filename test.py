#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from hashing import imagehash as ih
from hashing import neuralhash as nh
from hashing.SimCLR import resnet_wider
import generator
from skimage.transform import radon
import random
import pickle
import json
import string
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
random.seed(256)
np.random.seed(256)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.models import inception_v3
import cv2

#%%

path = 'test_hashing/BSDS500/Identification/data32.jpg'

image = Image.open(path)

kernels=(3, 5, 7)

gaussians = []
medians = []

t0 = time.time()
for size in kernels:
    array = np.array(image)
    gaussian = cv2.GaussianBlur(array, (size, size), sigmaX=0.6*((size-1)*0.5 - 1) + 0.8,
                                sigmaY=0.6*((size-1)*0.5 - 1) + 0.8)
    gaussians.append(Image.fromarray(gaussian))
    median = cv2.medianBlur(array, size)
    medians.append(Image.fromarray(median))
dt = time.time() - t0


gaussians[2].save('gaussian.png')
medians[2].save('median.png')
