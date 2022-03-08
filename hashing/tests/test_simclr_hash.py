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
sys.path.append(os.path.dirname(os.getcwd()))
import neural as nh
import time

HASH_SIZE = 8

img_ref = Image.open('data/lena.png')
img_adv = Image.open('data/peppers.png').convert('RGB')
hash_ref = nh.simclr_hash(img_ref, hash_size=HASH_SIZE)
hash_adv = nh.simclr_hash(img_adv, hash_size=HASH_SIZE)

distances_ref = []
distances_adv = []
hashes = []
names = []

t0 = time.time()

for file in os.listdir('data'):
    if file.startswith('lena') and file != 'lena.png':
        img = Image.open('data/' + file)
        hash_ = nh.simclr_hash(img, hash_size=HASH_SIZE)
        names.append(file)
        hashes.append(hash_)
        distances_ref.append(hash_ref.BER(hash_))
        distances_adv.append(hash_adv.BER(hash_))
        
dt = time.time() - t0

print(f'Mean of distances to Lena: {np.mean(distances_ref)}')
print(f'Mean of distances to Peppers: {np.mean(distances_adv)}')
print(f'Time needed for 58 images : {dt:.2f} s')

