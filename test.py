#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import hashing
import generator

#%%
path_db = 'test_hashing/BSDS500/Identification/'

algos = [
    hashing.NeuralAlgorithm('Inception v3', hash_size=8, batch_size=512, device='cpu'),
    hashing.NeuralAlgorithm('Inception v3', raw_features=True, distance='Jensen-Shannon',
                       device='cpu')
    ]

databases,_ = hashing.create_databases(algos, path_db)