#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
from torchvision.models import inception_v3

#%%
class ImageIterableDataset(IterableDataset):
    """
    Class representing a dataset of attacks on images. Convenient to use in conjunction
    with PyTorch DataLoader.
    """
    
    def __init__(self, path_to_imgs, attack_fraction, transforms, device='cuda',
                 seed=23):
        super(IterableDataset).__init__()
        if type(path_to_imgs) == str:
            self.img_paths = [path_to_imgs + name for name in os.listdir(path_to_imgs)]
        elif type(path_to_imgs) == list:
            self.img_paths = path_to_imgs
        self.transforms = transforms
        self.device = device
        self.fraction = attack_fraction
        self.seed = seed

    def __len__(self):
        return len(self.img_paths)
    
    def __iter__(self):
        # new numpy random generator
        rng = np.random.default_rng(seed=self.seed)
        imgs_to_attack = rng.choice(self.img_paths, size=round(len(self.img_paths)*self.fraction), 
                   replace=False)
        for img in imgs_to_attack:
            try:
                img_name = img.rsplit('/', 1)[1]
            except IndexError:
                img_name = img
            attacks = generator.perform_all_attacks(img, **generator.ATTACK_PARAMETERS)
            for key in attacks.keys():
                image = attacks[key]
                attack_name = key
                image = image.convert('RGB')
                image = self.transforms(image)
                image = image.to(torch.device(self.device))
                
                yield (image, img_name, attack_name)
                
path = 'test_hashing/BSDS500/Control/'
attack_fraction = 10/250
transforms = T.Compose([
    T.Resize((299,299), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
device = 'cpu'

dataset = ImageIterableDataset(path, attack_fraction, transforms, device=device)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

for a in dataloader:
    pass

#%%


