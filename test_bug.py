#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 09:14:27 2022

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


path_database = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_experimental = 'Datasets/ILSVRC2012_img_val/Experimental/'
path_control = 'Datasets/ILSVRC2012_img_val/Control/'

path_database = [path_database + file for file in os.listdir(path_database)[0:250]]
path_experimental = [path_experimental + file for file in os.listdir(path_experimental)[0:250]]
path_control = [path_control + file for file in os.listdir(path_control)[0:250]]

algos = [
    hashing.NeuralAlgorithm('SimCLR v2 ResNet50 2x', raw_features=True, batch_size=512,
                            device='cuda', distance='Jensen-Shannon'),
    ]

thresholds = [
    np.linspace(0.3, 0.9, 20),
    ]

    
positive_dataset = hashing.create_dataset(path_experimental, fraction=100/250, 
                                          existing_attacks=False)
negative_dataset = hashing.create_dataset(path_control, fraction=100/250,
                                          existing_attacks=False)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=512)

utils.save_digest(digest, save_folder)
