#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:28:50 2022

@author: cyrilvallez
"""

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from generator import generate_attacks as ga
import random
# Set the seed
random.seed(256)

# Number of images on which to perform the attacks in both groups
N = 100

G_VAR = [0.01, 0.02, 0.05]
S_VAR = [0.01, 0.02, 0.05]
SP_AMOUNT = [0.05, 0.1, 0.15]
G_KERNEL = [1, 2, 3]
M_KERNEL = [3, 5, 7]
QUALITY_FACTORS = [10, 50, 90]
RATIOS = [0.4, 0.8, 1.2, 1.6]
PERCENTAGES = [5, 10, 20, 40, 60]
RESIZE_CROP = True
ANGLES_ROT = [5, 10, 20, 40, 60]
RESIZE_ROT = True
ANGLES_SHEAR = [1, 2, 5, 10, 20]
FACTORS_CONTRAST = [0.6, 0.8, 1.2, 1.4]
FACTORS_COLOR = [0.6, 0.8, 1.2, 1.4]
FACTORS_BRIGHT = [0.6, 0.8, 1.2, 1.4]
FACTORS_SHARP = [0.6, 0.8, 1.2, 1.4]
LENGTHS = [10, 20, 30, 40, 50]

params = {'g_var':G_VAR, 's_var':S_VAR, 'sp_amount':SP_AMOUNT, 'g_kernel':G_KERNEL,
          'm_kernel':M_KERNEL, 'quality_factors':QUALITY_FACTORS,
         'ratios':RATIOS, 'percentages':PERCENTAGES, 'resize_crop':RESIZE_CROP,
         'angles_rot':ANGLES_ROT, 'resize_rot':RESIZE_ROT, 'angles_shear':ANGLES_SHEAR,
         'factors_contrast':FACTORS_CONTRAST, 'factors_color':FACTORS_COLOR,
         'factors_bright':FACTORS_BRIGHT, 'factors_sharp':FACTORS_SHARP,
         'lengths':LENGTHS}


path1 = 'BSDS500/Identification/'
path2 = 'BSDS500/Control/'

dest1 = 'BSDS500/Identification_attacks/'
dest2 = 'BSDS500/Control_attacks/'

names_id = os.listdir(path1)
names_ct = os.listdir(path2)

random.shuffle(names_id)
random.shuffle(names_ct)

images_id = [path1 + name for name in names_id[0:N]]
images_ct = [path2 + name for name in names_ct[0:N]]

save_id = [dest1 + name.split('.')[0] for name in names_id[0:N]]
save_ct = [dest2 + name.split('.')[0] for name in names_ct[0:N]]

ga.perform_all_and_save_list(images_id, save_name_list=save_id,
                             extension='PNG', **params)
ga.perform_all_and_save_list(images_ct, save_name_list=save_ct,
                             extension='PNG', **params)