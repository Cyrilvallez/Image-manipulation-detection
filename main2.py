#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:52:23 2022

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


path_database = 'Datasets/Kaggle_memes/Templates_experimental/'
path_experimental = 'Datasets/Kaggle_memes/Experimental/'
path_control = 'Datasets/Kaggle_memes/Control/'

algos = [
    hashing.FeatureAlgorithm('SIFT', batch_size=64, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=64, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=64, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=64, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0, 16, 100),
    np.linspace(0, 0.04, 100),
    np.linspace(0, 0.05, 100),
    np.linspace(0, 0.1, 100),
    np.linspace(0.25, 0.31, 100),
    np.linspace(0.33, 0.46, 100),
    np.linspace(0.26, 0.29, 100),
    np.linspace(0.24, 0.29, 100),
    np.linspace(0.32, 0.38, 100),
    np.linspace(0.42, 0.49, 100),
    np.linspace(0.42, 0.48, 100),
    ]

positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                                negative_dataset, general_batch_size=64,
                                artificial_attacks=False)

utils.save_digest(digest, save_folder)

