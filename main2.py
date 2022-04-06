#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:32:49 2022

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


path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_control = 'Datasets/ILSVRC2012_img_val/Control/'

algos = [
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 1x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 1x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet152 1x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 1x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 4x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet152 3x', raw_features=True, batch_size=64,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0.15, 0.65, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0.15, 0.6, 20),
    np.linspace(0.15, 0.6, 20),
    np.linspace(0.15, 0.6, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.3, 0.75, 20),
    np.linspace(0.3, 0.85, 20),
    np.linspace(0.4, 0.9, 20),
    np.linspace(0.4, 0.9, 20),
    np.linspace(0.4, 0.9, 20),
    np.linspace(0.4, 0.95, 20),
    ]

    
positive_dataset = hashing.create_dataset(path_experimental, fraction=1000/25000,
                                          existing_attacks=False)
negative_dataset = hashing.create_dataset(path_control, fraction=1000/25000,
                                          existing_attacks=False)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=64)

utils.save_digest(digest, save_folder)
