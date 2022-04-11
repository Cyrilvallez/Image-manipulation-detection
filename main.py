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
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=32,
                            device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=32,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=32,
                            device='cuda', distance='L1'),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=32,
                            device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=32,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=32,
                            device='cuda', distance='L1'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L1'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L1'),                        
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L2'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='L1'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=32,
                            device='cuda', distance='Jensen-Shannon')
    ]

thresholds = [
    np.linspace(0, 0.4, 20),
    np.linspace(4, 30, 20),
    np.linspace(180, 820, 20),
    np.linspace(0.15, 0.65, 20),
    np.linspace(0, 0.5, 20),
    np.linspace(0, 15, 20),
    np.linspace(0, 500, 20),
    np.linspace(0.3, 0.9, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(6, 30, 20),
    np.linspace(200, 800, 20),
    np.linspace(0.2, 0.6, 20),
    np.linspace(0.05, 0.45, 20),
    np.linspace(3, 20, 20),
    np.linspace(100, 350, 20),
    np.linspace(0.4, 0.85, 20),
    np.linspace(0, 0.4, 20),
    np.linspace(3, 12, 20),
    np.linspace(80, 250, 20),
    np.linspace(0.3, 0.9, 20),
    ]
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=32)

utils.save_digest(digest, save_folder)
