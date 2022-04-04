#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 17:08:12 2022

@author: cyrilvallez
"""

# =============================================================================
# Main experiment file
# =============================================================================

import numpy as np
import hashing 
from helpers import utils
import os

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()


path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_control = 'Datasets/ILSVRC2012_img_val/Control/'

algos = [
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Whash', hash_size=8, batch_size=512),
    hashing.FeatureAlgorithm('SIFT', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=512, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0, 0.4, 20),
    np.linspace(0.1, 0.4, 20),
    np.linspace(0.05, 0.4, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(0, 300, 20),
    np.linspace(0, 0.3, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(0.1, 0.4, 20),
    np.linspace(0.15, 0.65, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.3, 0.8, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0.4, 0.9, 20),
    ]

path_database = [path_database + file for file in os.listdir(path_database)][0:5000]
path_experimental = [path_experimental + file for file in os.listdir(path_experimental)][0:5000]
path_control = [path_control + file for file in os.listdir(path_control)][0:5000]
    
positive_dataset = hashing.create_dataset(path_experimental, fraction=500/5000,
                                          existing_attacks=False)
negative_dataset = hashing.create_dataset(path_control, fraction=500/5000,
                                          existing_attacks=False)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=512)

utils.save_digest(digest, save_folder)
