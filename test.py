#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:22:24 2022

@author: cyrilvallez
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import generator
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T
#%%
import hashing.neuralhash as nh
from hashing.SimCLRv1 import resnet_wider
from hashing.SimCLRv2 import resnet as SIMv2



image = Image.open('/Users/cyrilvallez/Desktop/Project/Datasets/BSDS500/Control/data221.jpg')

simclr = nh.load_simclr_v2(depth=101, width=2)(device='cpu')

transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])

tensor = torch.unsqueeze(transforms(image), dim=0)

out = simclr(tensor)


mem_params = sum([param.nelement()*param.element_size() for param in simclr.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in simclr.buffers()])
mem_ema = (mem_params + mem_bufs)/1e9 # in bytes

alloc = torch.cuda.max_memory_allocated()/1e9
