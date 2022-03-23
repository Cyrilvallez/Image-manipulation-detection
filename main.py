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
import argparse
import os

# =============================================================================
# =============================================================================

# Force the use of a user input path where to save the experiment at run-time,
# so that we do not mistakenly reuse the path from previous experiments
parser = argparse.ArgumentParser(description='Hashing experiment')
parser.add_argument('experiment_folder', type=str, help='A name for the experiment')
args = parser.parse_args()
experiment_folder = args.experiment_folder

results_folder = 'Results/'

# Check that it does not already exist
if experiment_folder in os.listdir(results_folder):
    raise ValueError('This experiment name is already taken. Choose another one.')
    
save_folder = results_folder + experiment_folder

# =============================================================================
# =============================================================================

path_database = 'Datasets/BSDS500/Experimental/'
path_experimental = 'Datasets/BSDS500/Experimental_attacks/'
path_control = 'Datasets/BSDS500/Control_attacks/'

algos = [
    #hashing.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=256),
    #hashing.ClassicalAlgorithm('Phash', hash_size=8, batch_size=256),
    #hashing.ClassicalAlgorithm('Dhash', hash_size=8, batch_size=256),
    #hashing.ClassicalAlgorithm('Whash', hash_size=8, batch_size=256),
    #hashing.ClassicalAlgorithm('Crop resistant hash', hash_size=8, batch_size=256),
    #hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=256,
    #                        device='cuda', distance='cosine'),
    #hashing.NeuralAlgorithm('Inception v3', raw_features=True, batch_size=256,
    #                        device='cuda', distance='Jensen-Shannon'),
    #hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=256,
    #                        device='cuda', distance='cosine'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 1x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 2x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v1 ResNet50 4x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet101 2x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon'),
    hashing.NeuralAlgorithm('SimCLR v2 ResNet152 3x', raw_features=True, batch_size=100,
                            device='cuda', distance='Jensen-Shannon')
    ]

thresholds = np.linspace(0, 0.8, 20)
    
positive_dataset = hashing.create_dataset(path_experimental, existing_attacks=True)
negative_dataset = hashing.create_dataset(path_control, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=100)

utils.save_digest(digest, save_folder)
