#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:10:03 2022

@author: cyrilvallez
"""

from generator.generate_attacks import * 

# Parameters that were used for the attacks. This is needed to compute
# the ROC curves for each attack separately
GAUSSIAN_VARIANCES = (0.01, 0.02, 0.05)
SPECKLE_VARIANCES = (0.01, 0.02, 0.05)
SALT_PEPPER_AMOUNTS = (0.05, 0.1, 0.15)
GAUSSIAN_KERNELS = (3, 5, 7)
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

ATTACK_PARAMETERS = {
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