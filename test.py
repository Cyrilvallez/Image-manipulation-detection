#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import generator
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
import hashing
import hashing.neuralhash as nh
from hashing.SimCLRv1 import resnet_wider
from hashing.SimCLRv2 import resnet as SIMv2
import scipy.spatial.distance as distance

#%%

image = Image.open('/Users/cyrilvallez/Desktop/Project/Datasets/BSDS500/Control/data221.jpg')

simclr = nh.load_simclr_v2(depth=101, width=2)(device='cpu')

transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])

tensor = torch.unsqueeze(transforms(image), dim=0)

out = simclr(tensor)


mem_params = sum([param.nelement()*param.element_size() for param in simclr.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in simclr.buffers()])
mem_ema = (mem_params + mem_bufs)/1e9 # in bytes

alloc = torch.cuda.max_memory_allocated()/1e9


#%%

path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
attacked = [path_experimental + file for file in os.listdir(path_experimental)[0:500]]

algo = [
        hashing.NeuralAlgorithm('ResNet50 1x', raw_features=True, batch_size=150,
                        device='cpu', distance='Jensen-Shannon')
        ]

db = hashing.create_databases(algo, path_database)
features = hashing.create_databases(algo, attacked)

db = list(db[0][0].values())
features = list(features[0][0].values())

#%%

N = 10

t0 = time.time()
for k in range(N):
    for i in db:
        for j in features:
            foo = nh.jensen_shannon_distance(i.features, j.features)
            
dt_loop = (time.time())/N

t0 = time.time()
for k in range(N):
    A = np.zeros((len(db), len(db[0].features)))
    B = np.zeros((len(features), len(features[0].features)))
    
    for i in db:
        A[i,:] = i.features[0:]
    for i in features:
        B[i,:] = i.features[0:]
        
    foo = distance.cdist(A,B, metric='jensenshannon', base=2)
    
dt_scipy = (time.time() - t0)/N