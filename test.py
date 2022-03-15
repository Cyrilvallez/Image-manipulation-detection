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


class ImageIterableDataset(IterableDataset):
    
    def __init__(self, imgs_to_attack, img_names, device='cuda'):
        super(IterableDataset).__init__()
        self.imgs_to_attack = imgs_to_attack
        self.img_names = img_names
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
                
                yield (image, img_name, attack_name)
                
                

def collate(batch):
    imgs, names, attacks = zip(*batch)
    return (imgs, names, attacks)


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
                if len(images) > 0:
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
                
                
                
path = 'test_hashing/BSDS500/Identification/'
 
imgs_names = [file for file in os.listdir(path)[0:30]]
imgs = [Image.open(path+file) for file in imgs_names]
                
    
test_dataset = CustomDataset(imgs, imgs_names)
loader = Loader(test_dataset, batch_size=5)

dataset = ImageIterableDataset(imgs, imgs_names, device='cpu')
dataloader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=collate)

transforms = T.Compose([
     T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
     T.CenterCrop(224),
     T.ToTensor()
     ])

dt_custom = []
dt_torch = []
count_custom = []
count_torch = []

for i in range(10):

    count = 0

    t0 = time.time()

    for a, _, _ in loader:
        count += len(a)
    
        tensors = []
        for img in a:
            tensors.append(transforms(img))
        
        tot = torch.stack(tensors, dim=0).to('cpu')
    
    dt_custom.append(time.time() - t0)
    count_custom.append(count)


    count = 0

    t0 = time.time()

    for a, _, _ in dataloader:
        count += len(a)
    
        tensors = []
        for img in a:
            tensors.append(transforms(img))
        
        tot = torch.stack(tensors, dim=0).to('cpu')
    
    dt_torch.append(time.time() - t0)
    count_torch.append(count)

print(f'Mean over 10 runs custom : {np.mean(dt_custom):.2f}')
print(f'Mean over 10 runs torch : {np.mean(dt_torch):.2f}')

        
#%%

class Parent():
    
    def __init__(self, a):
        self.a = a
        self.b = 3
        
    def test(self):
        self.a += self.add()
        
    def add(self):
        return 4
    
class Child(Parent):
    
    def __init__(self, a, b):
        
        Parent.__init__(self, a)
        self.b = b
        
    def add(self):
        return 28
    
Test = Child(1)
Test.test()
print(Test.a)