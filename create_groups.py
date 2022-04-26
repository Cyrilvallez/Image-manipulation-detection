#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:04:33 2022

@author: cyrilvallez
"""

# =============================================================================
# Divides the datasets into two subfolders randomly
# =============================================================================

import random
# Set the seed
random.seed(256)
import shutil
import os

source_dir = 'Datasets/ILSVRC2012_img_val'
target_dir1 = 'Datasets/ILSVRC2012_img_val/Experimental'
target_dir2 = 'Datasets/ILSVRC2012_img_val/Control'
    
file_names = os.listdir(source_dir)
# Remove non image data files
file_names = [a for a in file_names if 'ILSVRC2012' in a]
random.shuffle(file_names)
group1 = file_names[0:len(file_names)//2]
group2 = file_names[len(file_names)//2:]

for file_name in group1:
    shutil.move(os.path.join(source_dir, file_name), target_dir1)
    
for file_name in group2:
    shutil.move(os.path.join(source_dir, file_name), target_dir2)

#%%
import numpy as np
import shutil
import os

rng = np.random.default_rng(31)

memes_path = 'Datasets/Kaggle_memes/Memes'
templates_path = 'Datasets/Kaggle_memes/Templates'
target_dir = 'Datasets/Kaggle_memes/Experimental'

memes = [file.split('_', 1)[0] for file in os.listdir(memes_path)]
templates = [file.split('.', 1)[0] for file in os.listdir(templates_path)]

tot = 0
bank = set(templates)

N = len(memes)//2

while tot < N-200:
    template = rng.choice(list(bank), size=1, replace=False)[0]
    bank.remove(template)
    tot += len([a for a in os.listdir(memes_path) if a.split('_', 1)[0] == template])
    
    for img in os.listdir(memes_path):
        
        if img.split('_', 1)[0] == template:
            shutil.move(os.path.join(memes_path, img), target_dir)

#%%

for file in os.listdir(memes_path):
    shutil.move(os.path.join(memes_path, file), 'Datasets/Kaggle_memes/Control')