#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:54:42 2022

@author: cyrilvallez
"""

# =============================================================================
# Main experiment file
# =============================================================================

import numpy as np
import hashing 
from helpers import utils

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()


path_database = 'Datasets/Kaggle_memes/Templates/'
path_experimental = 'Datasets/Kaggle_memes/Memes/'

algos = [
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Whash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Crop resistant hash', hash_size=8, batch_size=16, cutoff=1),
    hashing.FeatureAlgorithm('SIFT', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=16, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0, 0.4, 20),
    np.linspace(0.1, 0.4, 20),
    np.linspace(0.05, 0.4, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(0, 300, 20),
    np.linspace(0, 0.3, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(0.1, 0.4, 20),
    np.linspace(0.15, 0.65, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.3, 0.8, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0.4, 0.9, 20),
    ]
    
dataset = hashing.create_dataset(path_experimental, existing_attacks=True)

databases, time = hashing.create_databases(algos, path_database)

digest = hashing.hashing(algos, thresholds, databases, dataset,
                         general_batch_size=16, artificial_attacks=False)

names = ['general', 'image_wise', 'time']

for i in range(len(digest)):
    utils.save_dictionary(digest[i], save_folder + names[i] + '.json')
    
utils.save_dictionary(time, save_folder + 'time_db.json')