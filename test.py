#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
import hashing 
from helpers import utils

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()


path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_test/'
path_control = 'Datasets/BSDS500/Control_test/'

algos = [
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=512),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0., 0.4, 20),
    np.linspace(0.2, 0.8, 20),
    np.linspace(0.2, 0.8, 20),
    ]
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)

params = (0.5, 1.5, 2, 3)
all_attack_names = [f'brightness_enhancement_{i}' for i in params] + \
    [f'color_enhancement_{i}' for i in params] + \
    [f'sharpness_enhancement_{i}' for i in params] + \
    [f'contrast_enhancement_{i}' for i in params]
    

digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=512,
                               all_attack_names=all_attack_names)

utils.save_digest(digest, save_folder)

