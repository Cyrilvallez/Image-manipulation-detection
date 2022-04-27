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
import shutil

path_experimental = 'Datasets/Kaggle_memes/Experimental/'
path_control = 'Datasets/Kaggle_memes/Control/'

des_experimental = 'Datasets/Kaggle_memes/Templates_experimental/'
des_control = 'Datasets/Kaggle_memes/Templates_control/'

exp = [name.split('_', 1)[0] for name in os.listdir(path_experimental)]
cont = [name.split('_', 1)[0] for name in os.listdir(path_control)]

exp = np.unique(exp)
cont = np.unique(cont)

for name in exp:
    
    templates = os.listdir('Datasets/Kaggle_memes/Templates/')
    
    for file in templates:
        
        if file.split('.', 1)[0] == name:
            
            shutil.move('Datasets/Kaggle_memes/Templates/' + file, des_experimental)
            
for name in cont:
    
    templates = os.listdir('Datasets/Kaggle_memes/Templates/')
    
    for file in templates:
        
        if file.split('.', 1)[0] == name:
            
            shutil.move('Datasets/Kaggle_memes/Templates/' + file, des_control)