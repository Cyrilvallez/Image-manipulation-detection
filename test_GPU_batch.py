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
    #hashing.NeuralAlgorithm('Inception v3', hash_size=8, device='cuda'),
    #hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', hash_size=8, device='cuda'),
    #hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=1028)
    hashing.NeuralAlgorithm('SimCLR v2 ResNet152 3x', raw_features=True, device='cuda',
                            distance='cosine', batch_size=32),
    #hashing.NeuralAlgorithm('SimCLR v1 ResNet50 4x', raw_features=True, device='cuda',
    #                        distance='cosine', batch_size=32),
    ]

dataset = hashing.create_dataset(images, existing_attacks=True)

databases, _ = hashing.create_databases(algos, path_database)

batches = [100]



keys = [str(algo) for algo in algos]

mean = []
for i in range(len(batches)):
    mean.append({})
    for j in range(len(keys)):
        mean[i][keys[j]] = 0

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
    

