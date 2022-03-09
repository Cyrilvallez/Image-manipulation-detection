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
from generator import generate_attacks as ga
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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3

#%%

SIMCLR = None
device = 'cpu'

t0 = time.time()

# Load the model if it has not being loaded already
if SIMCLR is None:
    SIMCLR = resnet_wider.resnet50x2()
    try:
        CHECKPOINT = torch.load(os.path.expanduser('~/Project/hashing/SimCLR/resnet50-2x.pth'))
    except FileNotFoundError:
        CHECKPOINT = torch.load(os.path.expanduser('~/Desktop/Project/hashing/SimCLR/resnet50-2x.pth'))
    SIMCLR.load_state_dict(CHECKPOINT['state_dict'])
    SIMCLR.fc = nn.Identity()
    SIMCLR.eval()
    SIMCLR.to(torch.device(device))
    
dt = time.time() - t0
