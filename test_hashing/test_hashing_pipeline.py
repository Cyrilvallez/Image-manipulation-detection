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
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from hashing import imagehash as ih
from hashing import neuralhash as nh
from hashing import general_hash as gh
import Create_plot as plot

path_db = 'BSDS500/Identification/'
path_id = 'BSDS500/Identification_attacks/'
path_ct = 'BSDS500/Control_attacks/'

algos = [
    ih.ClassicalAlgorithm('Ahash', hash_size=8, batch_size=512),
    ih.ClassicalAlgorithm('Phash', hash_size=8, batch_size=512),
    nh.NeuralAlgorithm('Inception v3', hash_size=8, batch_size=512, device='cpu')
    ]

thresholds = np.linspace(0, 0.4, 10)
#thresholds = [[0.1,0.2], [0.2], [0.2, 0.3, 0.4]]
    
positive_dataset = gh.create_dataset(path_id, existing_attacks=True)
negative_dataset = gh.create_dataset(path_ct, existing_attacks=True)


res = gh.total_hashing(algos, thresholds, path_db, positive_dataset, negative_dataset)



