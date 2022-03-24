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

#%%

from helpers import utils

dic = {'test': 4, 'test2':2}
foo = []

for i in range(6):
    foo.append(dic)
    
foo = tuple(foo)

experiment_folder = 'TEst/bbs/vvi'

utils.save_digest(foo, experiment_folder)
    
#%%

import torchvision.models as models

image = Image.open('/Users/cyrilvallez/Desktop/Project/Datasets/BSDS500/Control/data221.jpg')

transforms = T.Compose([
    T.Resize((256,256), interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

tensor = transforms(image).unsqueeze(dim=0)

net = models.resnet152(pretrained=True)
net.fc = nn.Identity()

out = net(tensor)

#%%
import numpy as np
import torch.nn as nn
import scipy.special as special
import time
import torch

a = np.random.rand(2300)
N = 10000


t0 = time.time()
for i in range(N):
    b = special.softmax(a)
dt_scipy = (time.time() - t0)/N

t0 = time.time()
for i in range(N):
    b = nn.functional.softmax(torch.from_numpy(a), dim=0).numpy()
dt_torch = (time.time() - t0)/N