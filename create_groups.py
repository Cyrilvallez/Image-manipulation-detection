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
    