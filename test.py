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

path = 'test_performances/BSDS500/Control/'

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        
test = find('resnet50-2x.pth', os.path.dirname(os.getcwd()))