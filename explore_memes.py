#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:56:42 2022

@author: cyrilvallez
"""

import os
import numpy as np
from helpers import utils
import matplotlib.pyplot as plt

memes = 'Datasets/Kaggle_memes/Memes'
templates = 'Datasets/Kaggle_memes/Templates'

memes = [file.split('_', 1)[0] for file in os.listdir(memes)]
templates = [file.split('.', 1)[0] for file in os.listdir(templates)]

unique, counts = np.unique(memes, return_counts=True)

dic = utils.load_dictionary('Results/Benchmark_memes/image_wise.json')

dic2 = dic['SimCLR v2 ResNet101 2x raw features Jensen-Shannon']['Threshold 0.400']

count_algo = []

for img in unique:
    count_algo.append(dic2[img]['correct detection'])

plt.figure()
plt.bar(np.arange(len(unique)), counts, color='b')
plt.bar(np.arange(len(unique)), count_algo, color='r')
