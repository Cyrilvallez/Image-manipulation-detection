#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import hashing
import generator
from tqdm import tqdm
import torch

#%%
path_db = 'Datasets/BSDS500/Experimental/'
image = path_db + 'data5.jpg'

print(f'Before : {torch.cuda.memory_allocated()/1e9:.2f} GB')
for a in range(5):
    a = torch.rand(512, 3, 224, 244).to('cuda')
print(f'After : {torch.cuda.memory_allocated()/1e9:.2f} GB')



    