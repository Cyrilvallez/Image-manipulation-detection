#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:41:34 2022

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


path_database = 'Datasets/Kaggle_memes/Templates/'
path_experimental = 'Datasets/Kaggle_memes/Memes/'

algos = [
    # hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=256),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    # [0.211],
    [0.458],
    # [0.453],
    [0.363],
    ]
    
dataset = hashing.create_dataset(path_experimental, existing_attacks=True)

databases, time = hashing.create_databases(algos, path_database)

digest = hashing.hashing_ensemble(algos, thresholds, databases, dataset,
                         general_batch_size=16)

names = ['general', 'image_wise']

# Make sure the path exists, and creates it if this is not the case
exist = os.path.exists(save_folder)
if not exist:
    os.makedirs(save_folder)

for i in range(len(digest)):
    utils.save_dictionary(digest[i], save_folder + '/' + names[i] + '.json')
    
