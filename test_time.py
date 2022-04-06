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
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'

path_database = [path_database + file for file in os.listdir(path_database)][0:250]
path_experimental = [Image.open(path_experimental + file) for file in os.listdir(path_experimental)[0:1]]


database = algo.create_database(path_database, {})

img = algo.preprocess(path_experimental)
algo.load_model()
fingerprint = algo.process_batch(img)
algo.kill_model()

t0 = time.time()
distances = fingerprint[0].compute_distances(database)[0]
dt_test = time.time() - t0



print(f'time : {dt_test:.2e}')
