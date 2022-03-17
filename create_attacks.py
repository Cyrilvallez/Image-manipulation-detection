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
import generator
import random
# Set the seed
random.seed(256)



params = generator.ATTACK_PARAMETERS

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

generator.perform_all_and_save_list(images_identification, save_name_list=save_identification,
                             extension='PNG', **params)
generator.perform_all_and_save_list(images_control, save_name_list=save_control,
                             extension='PNG', **params)