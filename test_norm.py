#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:36:54 2022

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
import torch.nn.functional as F
import os
import hashing
import hashing.neuralhash as nh
from hashing.SimCLRv1 import resnet_wider
from hashing.SimCLRv2 import resnet as SIMv2
import scipy.spatial.distance as distance


path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

attacked_imgs = np.unique([file.split('_', 1)[0] for file in os.listdir(path_experimental)])

database = [file for file in os.listdir(path_database) if file.split('.',1)[0] in attacked_imgs]
attacks = []
for img in database:
    attack = [path_experimental + file for file in os.listdir(path_experimental) \
              if file.split('_', 1)[0] == img.split('.',1)[0]]
    attacks.append(attack)
    
non_attacks = [path_control + file for file in os.listdir(path_control)]
non_attacks = np.reshape(non_attacks, (100, 58)).tolist()
    
model = nh.load_simclr_v2(50, 2)('cuda')

distances = []
L1 = nh.norm(1)
L2 = nh.norm(2)

transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])

ref_distances = []

for i in range(len(database)):
    ref_image = Image.open(path_database + database[i])
    ref_image = torch.unsqueeze(transforms(ref_image), dim=0)
    ref_features = model(ref_image)
    
    tensors = []
    for img in attacks[i]:
        img = Image.open(img)
        tensors.append(transforms(img))
        
    tensors = torch.stack(tensors, dim=0).to('cuda')
    
    attack_features = model(tensors).cpu().numpy()
    
    for feat in attack_features:
        ref_distances.append(L1(ref_features, feat))
        
print(f'Mean distance for same images : {np.mean(ref_distances):.2f}')

unknown_distances = []

for i in range(len(database)):
    ref_image = Image.open(path_database + database[i])
    ref_image = torch.unsqueeze(transforms(ref_image), dim=0)
    ref_features = model(ref_image)
    
    tensors = []
    for img in non_attacks[i]:
        img = Image.open(img)
        tensors.append(transforms(img))
        
    tensors = torch.stack(tensors, dim=0).to('cuda')
    
    non_attack_features = model(tensors).cpu().numpy()
    
    for feat in non_attack_features:
        unknown_distances.append(L1(ref_features, feat))
        
print(f'Mean distance for unknown images : {np.mean(unknown_distances):.2f}')
    
