#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:09:45 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
import imagehash

img_ref = Image.open('data/lena.png')
img_adv = Image.open('data/peppers.png')
hash_ref = imagehash.hist_hash(img_ref)
hash_adv = imagehash.hist_hash(img_adv)

distances_ref = []
distances_adv = []
hashes = []
names = []

for file in os.listdir('data'):
    if file.startswith('lena') and file != 'lena.png':
        img = Image.open('data/' + file)
        hash_ = imagehash.hist_hash(img)
        names.append(file)
        hashes.append(hash_)
        distances_ref.append((hash_ref - hash_)/len(hash_ref))
        distances_adv.append((hash_adv - hash_)/len(hash_adv))
        
print(f'Mean of distances to Lena: {np.mean(distances_ref)}')
print(f'Mean of distances to Peppers: {np.mean(distances_adv)}')

