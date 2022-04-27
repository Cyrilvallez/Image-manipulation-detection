#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
import hashing 
from helpers import utils
import os

path_experimental = 'Datasets/Kaggle_memes/Experimental/'
path_control = 'Datasets/Kaggle_memes/Control/'

exp = [name.split('_', 1)[0] for name in os.listdir(path_experimental)]
cont = [name.split('_', 1)[0] for name in os.listdir(path_control)]

exp = np.unique(exp)
cont = np.unique(cont)

a = np.isin(exp, cont)

print(f'Sum : {a.sum()}')