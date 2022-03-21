#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:54:11 2022

@author: cyrilvallez
"""

import os
import hashing

path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_ct = 'Datasets/BSDS500/Control_attacks/'

images = [path_experimental + file for file in os.listdir(path_experimental)]
images = images[0:1500]
#%%
algos = [
    hashing.NeuralAlgorithm('Inception v3', hash_size=8, device='cuda', batch_size=1028),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', hash_size=8, device='cuda', batch_size=1028),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=1028)
    ]

dataset = hashing.create_dataset(images, existing_attacks=True)

databases, _ = hashing.create_databases(algos, path_database)

batches = [256, 512, 1028, 2048]
mean = [{str(algos[0]):0, str(algos[1]):0, str(algos[2]):0} for i in range(len(batches))]

N = 1

for i, batch in enumerate(batches):
    for j in range(N):
        _, _, _, time = hashing.hashing(algos, [0.2], databases, dataset, general_batch_size=batch)
        for key in time.keys():
            mean[i][key] += time[key]
            
for i, batch in enumerate(batches):
    for key in time.keys():
        mean[i][key] /= N

for i, batch in enumerate(batches):
    print(f'Batch {batch} : {mean[i]}')
    

