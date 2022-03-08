#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:04:33 2022

@author: cyrilvallez
"""

# =============================================================================
# Divides the BSDS500 dataset into two subfolders randomly
# =============================================================================

import random
# Set the seed
random.seed(256)
import shutil
import os

source_dir = 'BSDS500'
target_dir1 = 'BSDS500/Identification'
target_dir2 = 'BSDS500/Control'
    
file_names = os.listdir(source_dir)
# Remove non image data files
file_names = [a for a in file_names if 'data' in a]
random.shuffle(file_names)
group1 = file_names[0:len(file_names)//2]
group2 = file_names[len(file_names)//2:]

for file_name in group1:
    shutil.move(os.path.join(source_dir, file_name), target_dir1)
    
for file_name in group2:
    shutil.move(os.path.join(source_dir, file_name), target_dir2)
    