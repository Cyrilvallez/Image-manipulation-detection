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
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Whash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Crop resistant hash', hash_size=8, batch_size=512, cutoff=1),
    hashing.FeatureAlgorithm('SIFT', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=512, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=512, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    [0.052],
    [0.224],
    [0.159],
    [0.072],
    [0.069],
    [67.7778],
    [0.0906],
    [0.0414],
    [0.1606],
    [0.2611],
    [0.3683],
    [0.2996],
    [0.3168],
    [0.5197],
    [0.5133],
    [0.5208],
    ]

positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)

digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                                negative_dataset, general_batch_size=64,
                                artificial_attacks=True)
                                
utils.save_digest(digest, save_folder)
