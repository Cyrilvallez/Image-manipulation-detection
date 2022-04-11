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

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()


path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

algos = [
    # hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=256,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=256,
                            device='cuda', distance='L1'),
    # hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='Jensen-Shannon'),
    # hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='cosine'),
    # hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='L2'),
    # hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='L1'),
    # hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=256,
                            # device='cuda', distance='Jensen-Shannon')
    ]

thresholds = [
    # np.linspace(0, 0.4, 20),
    np.linspace(8, 25, 20),
    np.linspace(200, 800, 20),
    # np.linspace(0.3, 0.9, 20),
    # np.linspace(0, 0.4, 20),
    # np.linspace(3, 12, 20),
    # np.linspace(80, 250, 20),
    # np.linspace(0.3, 0.9, 20),
    ]
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=256)

utils.save_digest(digest, save_folder)
