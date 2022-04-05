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

algo_ori = hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                        device='cpu', distance='Jensen-Shannon', numpy=True)
algo_test = hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                        device='cpu', distance='Test_torch', numpy=False)

path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'

path_database = [path_database + file for file in os.listdir(path_database)][0:10000]
path_experimental = [Image.open(path_experimental + file) for file in os.listdir(path_experimental)[10001:10002]]


database_test = algo_test.create_database(path_database, {})
database_original = super(type(algo_ori), algo_ori).create_database(path_database, {})


img_test = algo_test.preprocess(path_experimental)
algo_test.load_model()
fingerprint_test = algo_test.process_batch(img_test)
algo_test.kill_model()

img_ori = algo_ori.preprocess(path_experimental)
algo_ori.load_model()
fingerprint_ori = algo_ori.process_batch(img_ori)
algo_ori.kill_model()

t0 = time.time()
distances = fingerprint_test[0].compute_distances_torch(database_test)[0]
dt_test = time.time() - t0

t0 = time.time()
distances2 = fingerprint_ori[0].compute_distances(database_original)[0]
dt_ori = time.time() - t0

print(f'Same : {np.allclose(distances, distances2)}')
print(f'N > 1e-10 : {(abs(distances - distances2) > 1e-8).sum()}')
print(f'time test: {dt_test:.2e}')
print(f'time original : {dt_ori:.2e}')

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

#%%

"""
index = 0

a = fingerprint_ori[0].features
b = database_test[0][index].numpy()

m = (a+b)/2

res = a*(np.log(a/m))
"""