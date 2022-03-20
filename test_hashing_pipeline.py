#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:03:55 2022

@author: cyrilvallez
"""


# =============================================================================
# Test the hashing pipeline
# =============================================================================

import numpy as np
import hashing 
from helpers import create_plot as plot

path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

algos = [
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=512),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=512),
    hashing.NeuralAlgorithm('Inception v3', hash_size=8, batch_size=512, device='cpu')
    ]

thresholds = np.linspace(0, 0.4, 10)
#thresholds = [[0.1,0.2], [0.2], [0.2, 0.3, 0.4]]
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


res = hashing.total_hashing(algos, thresholds, path_database, positive_dataset, negative_dataset)


