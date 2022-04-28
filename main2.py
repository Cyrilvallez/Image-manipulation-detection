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
    hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=16),
    hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=16),
    hashing.FeatureAlgorithm('SIFT', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('ORB', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + DAISY', batch_size=16, n_features=30, cutoff=1),
    hashing.FeatureAlgorithm('FAST + LATCH', batch_size=16, n_features=30, cutoff=1),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('EfficientNet B7', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('ResNet101 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=16,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    [0.0],
    [0.1947],
    [0.1789],
    [4.6869],
    [0.0198],
    [0.0237],
    [0.0596],
    [0.287],
    [0.4377],
    [0.2688],
    [0.2713],
    [0.3448],
    [0.4511],
    [0.4442],
    ]

experimental = [path_experimental + file for file in os.listdir(path_experimental)]
experimental += [path_control + file for file in os.listdir(path_control)]

database = [path_database + file for file in os.listdir(path_database)]
path_db_ct = 'Datasets/Kaggle_memes/Templates_control/'
database += [path_db_ct + file for file in os.listdir(path_db_ct)]

# positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
# negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)

dataset = hashing.create_dataset(path_experimental, existing_attacks=True)

databases, time_db = hashing.create_databases(algos, database)


# digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                                # negative_dataset, general_batch_size=64,
                                # artificial_attacks=False)
                                
digest = hashing.hashing(algos, thresholds, path_database, dataset, general_batch_size=16,
                                artificial_attacks=False)

digest = (*digest, time_db)

names = ['general.json', 'image_wise.json', 'time.json', 'time_db.json']
# utils.save_digest(digest, save_folder)

for dic, name in zip(digest, names):
    utils.save_dictionary(dic, save_folder + '/' + name)