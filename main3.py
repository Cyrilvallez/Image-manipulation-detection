#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:53:03 2022

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
    hashing.FeatureAlgorithm('ORB', batch_size=1000, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=1000, n_features=40, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=1000, n_features=50, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=1000, n_features=100, cutoff=1),
    ]

thresholds = np.linspace(0, 0.3, 20)
    
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=1000)

utils.save_digest(digest, save_folder)
