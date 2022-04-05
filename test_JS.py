#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:07:37 2022

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

algo = hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                        device='cuda', distance='Jensen-Shannon')

path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'

path_database = [path_database + file for file in os.listdir(path_database)][0:10000]
path_experimental = [Image.open(path_experimental + file) for file in os.listdir(path_experimental)[0:1]]


database_test = algo.create_database(path_database, {})
database_original = super(type(algo), algo).create_database(path_database, {})


img = algo.preprocess(path_experimental)
fingerprint = algo.process_batch(img)

t0 = time.time()
distances = fingerprint.compute_distances_torch(database_test)
dt_new = time.time() - t0

t0 = time.time()
distances2 = fingerprint.compute_distances(database_original)
dt_old = time.time() - t0

print(f'Same : {np.allclose(distances, distances2)}')
print(f'N > 1e-10 : {(abs(distances - distances2) > 1e-10).sum()}')
print(f'time new : {dt_new:.2e}')
print(f'time old : {dt_old:.2e}')

#%%
"""
from torch.utils.data import Dataset, IterableDataset, DataLoader
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = [path_experimental + file for file in os.listdir(path_experimental)][0:10]

dataset = hashing.create_dataset(path_experimental, fraction=1)
dataloader = DataLoader(dataset, batch_size=100, shuffle=False,
                        collate_fn=hashing.collate)

for images, image_names, attack_names in dataloader:
    pass
"""
