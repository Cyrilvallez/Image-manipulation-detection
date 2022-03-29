#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:53:39 2022

@author: cyrilvallez
"""

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
    hashing.FeatureAlgorithm('ORB', batch_size=100, n_features=20),
    ]

thresholds = np.linspace(0, 0.8, 10)

pos_images = [path_experimental + file for file in os.listdir(path_experimental)[0:500] \
              if 'data' in file]
neg_images = [path_control + file for file in os.listdir(path_control)[0:500] \
              if 'data' in file]
    
positive_dataset = hashing.create_dataset(pos_images, existing_attacks=True)
negative_dataset = hashing.create_dataset(neg_images, existing_attacks=True)


digest = hashing.total_hashing(algos, thresholds, path_database, positive_dataset,
                               negative_dataset, general_batch_size=50)

utils.save_digest(digest, save_folder)

