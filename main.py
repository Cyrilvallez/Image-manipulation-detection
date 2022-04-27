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
import os

# Force the use of a user input at run-time to specify the path 
# so that we do not mistakenly reuse the path from previous experiments
save_folder = utils.parse_input()


path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

algos = [
    hashing.NeuralAlgorithm('DINO ViT B8', raw_features=True, batch_size=256,
                            device='cuda', distance='cosine'),
    ]

thresholds = np.linspace(0, 1, 50)
    

positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                                negative_dataset, general_batch_size=16)
                               

utils.save_digest(digest, save_folder)

