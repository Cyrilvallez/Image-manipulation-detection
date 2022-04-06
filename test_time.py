#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:23:00 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import hashing
import hashing.neuralhash as nh
from torch.utils.data import Dataset, IterableDataset, DataLoader
import scipy.spatial.distance as distance

algo = hashing.FeatureAlgorithm('SIFT', batch_size=512)

path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
file = os.listdir(path_database)

image = Image.open(path_database + file[0]) 

names = [
    'Inception v3',
    'ResNet50 1x',
    'ResNet101 1x',
    'ResNet152 1x',
    'ResNet50 2x',
    'ResNet101 2x',
    'EfficientNet B7',
    'SimCLR v1 ResNet50 1x',
    'SimCLR v1 ResNet50 2x',
    'SimCLR v1 ResNet50 4x',
    'SimCLR v2 ResNet50 2x',
    'SimCLR v2 ResNet101 2x',
    'SimCLR v2 ResNet152 3x',
    ]

for name in names:
    model = nh.NEURAL_MODEL_LOADER[name]('cpu')
    img = nh.NEURAL_MODEL_TRANSFORMS[name](image).unsqueeze(dim=0)

    with torch.no_grad():
        out = model(img).squeeze().numpy()

    print(f' {name} : {min(out)}')




