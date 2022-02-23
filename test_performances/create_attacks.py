#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 09:28:50 2022

@author: cyrilvallez
"""

# =============================================================================
# Performs the attacks in both groups
# =============================================================================

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from generator import generate_attacks as ga
import random
# Set the seed
random.seed(256)

# Parameters for the attacks
GAUSSIAN_VARIANCES = (0.01, 0.02, 0.05)
SPECKLE_VARIANCES = (0.01, 0.02, 0.05)
SALT_PEPPER_AMOUNTS = (0.05, 0.1, 0.15)
GAUSSIAN_KERNELS = (1, 2, 3)
MEDIAN_KERNELS = (3, 5, 7)
COMPRESSION_QUALITY_FACTORS = (10, 50, 90)
SCALING_RATIOS = (0.4, 0.8, 1.2, 1.6)
CROPPING_PERCENTAGES = (5, 10, 20, 40, 60)
RESIZE_CROPPING = True
ROTATION_ANGLES = (5, 10, 20, 40, 60)
RESIZE_ROTATION = True
SHEARING_ANGLES = (1, 2, 5, 10, 20)
CONTRAST_FACTORS = (0.6, 0.8, 1.2, 1.4)
COLOR_FACTORS = (0.6, 0.8, 1.2, 1.4)
BRIGHTNESS_FACTORS = (0.6, 0.8, 1.2, 1.4)
SHARPNESS_FACTORS = (0.6, 0.8, 1.2, 1.4)
TEXT_LENGTHS = (10, 20, 30, 40, 50)

params = {
    'gaussian_variances': GAUSSIAN_VARIANCES,
    'speckle_variances': SPECKLE_VARIANCES,
    'salt_pepper_amounts': SALT_PEPPER_AMOUNTS,
    'gaussian_kernels': GAUSSIAN_KERNELS,
    'median_kernels': MEDIAN_KERNELS,
    'compression_quality_factors': COMPRESSION_QUALITY_FACTORS,
    'scaling_ratios': SCALING_RATIOS,
    'cropping_percentages': CROPPING_PERCENTAGES,
    'resize_cropping': RESIZE_CROPPING,
    'rotation_angles': ROTATION_ANGLES,
    'resize_rotation': RESIZE_ROTATION,
    'shearing_angles': SHEARING_ANGLES,
    'contrast_factors': CONTRAST_FACTORS,
    'color_factors': COLOR_FACTORS,
    'brightness_factors': BRIGHTNESS_FACTORS,
    'sharpness_factors': SHARPNESS_FACTORS,
    'text_lengths': TEXT_LENGTHS
    }

# Number of images on which to perform the attacks in both groups
N = 100


path1 = 'BSDS500/Identification/'
path2 = 'BSDS500/Control/'

destination1 = 'BSDS500/Identification_attacks/'
destination2 = 'BSDS500/Control_attacks/'

names_identification = os.listdir(path1)
names_control = os.listdir(path2)

random.shuffle(names_identification)
random.shuffle(names_control)

images_identification = [path1 + name for name in names_identification[0:N]]
images_control = [path2 + name for name in names_control[0:N]]

save_identification = [destination1 + name.split('.')[0] for name \
                       in names_identification[0:N]]
save_control = [destination2 + name.split('.')[0] for name in names_control[0:N]]

ga.perform_all_and_save_list(images_identification, save_name_list=save_identification,
                             extension='PNG', **params)
ga.perform_all_and_save_list(images_control, save_name_list=save_control,
                             extension='PNG', **params)