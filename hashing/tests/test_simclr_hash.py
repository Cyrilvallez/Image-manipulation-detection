#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:07:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import hashing.neuralhash as nh
import time

HASH_SIZE = 8

path_ref = ['data/lena.png']
path_adv = ['data/peppers.png']
hash_ref = nh.simclr_hash(path_ref, hash_size=HASH_SIZE, device='cpu')[0]
hash_adv = nh.simclr_hash(path_adv, hash_size=HASH_SIZE, device='cpu')[0]

distances_ref = []
distances_adv = []

t0 = time.time()

paths = ['data/' + file for file in os.listdir('data') if (file.startswith('lena') \
                                                           and file != 'lena.png')]

hashes = nh.simclr_hash(paths, hash_size=HASH_SIZE, device='cpu')

for hash_ in hashes:
        distances_ref.append(hash_ref.BER(hash_))
        distances_adv.append(hash_adv.BER(hash_))
        
dt = time.time() - t0

print(f'Mean of distances to Lena: {np.mean(distances_ref)}')
print(f'Mean of distances to Peppers: {np.mean(distances_adv)}')
print(f'Time needed for 58 images : {dt:.2f} s')
