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
from hashing import general_hash as gh
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
path_db = 'test_hashing/BSDS500/Identification/'

algos = [
    nh.NeuralAlgorithm('Inception v3', hash_size=8, batch_size=512, device='cpu'),
    nh.NeuralAlgorithm('Inception v3', raw_features=True, distance='Jensen-Shannon',
                       device='cpu')
    ]

databases,_ = gh.create_databases(algos, path_db)