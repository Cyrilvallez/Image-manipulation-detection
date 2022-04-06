#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:56:42 2022

@author: cyrilvallez
"""

import os
import numpy as np

memes = 'Datasets/Kaggle_memes/Memes'
templates = 'Datasets/Kaggle_memes/Templates'

memes = [file.split('_', 1)[0] for file in os.listdir(memes)]
templates = [file.split('.', 1)[0] for file in os.listdir(templates)]

unique, counts = np.unique(memes, return_counts=True)