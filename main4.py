#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:29:05 2022

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


path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

algos = [
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=128),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=128),
    hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=128),
    hashing.ClassicalAlgorithm('Whash', hash_size=8, batch_size=128),
    hashing.ClassicalAlgorithm('Crop resistant hash', hash_size=8, batch_size=128),
    hashing.FeatureAlgorithm('SIFT', batch_size=128, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=128, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=128, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=128, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=128,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=128,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=128,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=128,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=128,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=128,
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
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.3, 0.8, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0.4, 0.9, 20),
    ]

    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=128)

utils.save_digest(digest, save_folder)