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
import scipy.spatial.distance as distance

algo_ori = hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                        device='cpu', distance='Jensen-Shannon', numpy=True)
algo_test = hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                        device='cpu', distance='Test_torch', numpy=False)

path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'

path_database = [path_database + file for file in os.listdir(path_database)][0:1000]
path_experimental = [Image.open(path_experimental + file) for file in os.listdir(path_experimental)[0:1]]


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
print(f'N > 1e-6 : {(abs(distances - distances2) > 1e-6).sum()} out of {len(distances)}')
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

out1 = np.empty(len(a))
out2 = np.empty(len(a))

for i in range(len(a)):
    
    if a[i] > 0 and m[i] > 0:
        out1[i] = a[i]*np.log(a[i]/m[i])
    elif a[i] == 0 and m[i] >= 0:
        out1[i] = 0
    else:
        out1[i] = float('inf')
        
    if b[i] > 0 and m[i] > 0:
        out2[i] = b[i]*np.log(b[i]/m[i])
    elif b[i] == 0 and m[i] >= 0:
        out2[i] = 0
    else:
        out2[i] = float('inf')

"""

#%%
"""
def jensen_distance_torch(a, B, base=2):
    
    a = a/torch.sum(a)
    B = B/torch.sum(B, axis=1)[:,None]
    
    A = torch.tile(a, (B.shape[0], 1))
    M = (A+B)/2
    
    X = torch.where((A>0) & (M>0), A*(torch.log2(A/M)), torch.tensor([0.], device=a.device))
    #X[(A==0) & (M>=0)] = float('inf')
    
    Y = torch.where((B>0) & (M>0), B*(torch.log2(B/M)), torch.tensor([0.], device=a.device))
    #Y[(B==0) & (M>=0)] = float('inf')
    
    out = torch.sqrt(1/2*(X + Y).sum(dim=1)).cpu().numpy()
    
    return out, X, Y, A, M

a = fingerprint_test[0].features
b = database_test[0]

res, X, Y, A, M = jensen_distance_torch(a, b)

X = X.numpy()
Y = Y.numpy()
"""


