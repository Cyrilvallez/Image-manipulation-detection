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


path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_control = 'Datasets/ILSVRC2012_img_val/Control/'

algos = [
    hashing.FeatureAlgorithm('ORB', batch_size=1000, n_features=30, cutoff=1),
    ]

thresholds = [
    np.linspace(0, 0.3, 20),
    ]
    
positive_dataset = hashing.create_dataset(path_experimental, fraction=1000/25000,
                                          existing_attacks=False)
negative_dataset = hashing.create_dataset(path_control, fraction=1000/25000,
                                          existing_attacks=False)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=1000)

utils.save_digest(digest, save_folder)
