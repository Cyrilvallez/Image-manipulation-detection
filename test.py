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

"""
class ImageIterableDataset(IterableDataset):
    
    def __init__(self, imgs_to_attack, img_names, transforms, device='cuda'):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack
        self.img_names = img_names
        self.transforms = transforms
        self.device = device

    def __len__(self):
        return len(self.imgs_to_attack)
    
    def __iter__(self):
        
        for img, img_name in zip(self.imgs_to_attack, self.img_names):
            
            attacks = generator.perform_all_attacks(img, **generator.ATTACK_PARAMETERS)
            for key in attacks.keys():
                image = attacks[key]
                attack_name = key
                image = image.convert('RGB')
                image = self.transforms(image)
                image = image.to(torch.device(self.device))
                
                yield (image, img_name, attack_name)

transforms = T.Compose([
     T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
     T.CenterCrop(224),
     T.ToTensor()
     ])

path = 'test_hashing/BSDS500/Identification/'
 
imgs_names = [file for file in os.listdir(path)[0:1]]
imgs = [Image.open(path+file) for file in imgs_names]

dataset = ImageIterableDataset(imgs, imgs_names, transforms=transforms, device='cpu')
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
"""


path = 'test_hashing/BSDS500/Identification/'
 
imgs_names = [file for file in os.listdir(path)[0:1]]
imgs = [Image.open(path+file) for file in imgs_names]

class CustomDataset(object):
    
    def __init__(self, imgs_to_attack, img_names):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack
        self.img_names = img_names

    def __len__(self):
        return len(self.imgs_to_attack)
    
    def __iter__(self):
        
        for img, img_name in zip(self.imgs_to_attack, self.img_names):
            
            attacks = generator.perform_all_attacks(img, **generator.ATTACK_PARAMETERS)
            for key in attacks.keys():
                image = attacks[key]
                attack_name = key
                image = image.convert('RGB')
                
                yield (image, img_name, attack_name)
                
test_dataset = CustomDataset(imgs, imgs_names)
                
                
class Loader(object):
    
    def __init__(self, dataset, batch_size=256):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __iter__(self):
        
        iterator = iter(self.dataset)
        
        images = []
        img_names = []
        attack_names = []
        count = 0
        
        while True:
            
            try:
                image, img_name, attack_name = next(iterator)
            except StopIteration:
                yield (images, img_names, attack_names)
                break
            
            images.append(image)
            img_names.append(img_name)
            attack_names.append(attack_name)
            count += 1
            
            if count == self.batch_size:
                
                count = 0
                yield (images, img_names, attack_names)
                images = []
                img_names = []
                attack_names = []
                
    
loader = Loader(test_dataset, batch_size=5)

count = 0

for a in loader:
    count += len(a[0])
    
print(count)

