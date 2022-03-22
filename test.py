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
from hashing.SimCLRv1 import resnet_wider
from hashing.SimCLRv2 import resnet as SIMv2

def load_simclr_v2(depth, width, ema=True):
    """
    Load the simclr v2 ResNet model with given depth and width.

    Parameters
    ----------
    depth : int
        Depth of the ResNet model
    width : int
        Width multiplier.
    selective_kernel : Boolean
        Whether to use a selective kernel.

    Returns
    -------
    load : function
        a Loader function for the simclr v2 ResNet model with given depth and width.

    """
    
    if ema:
        checkpoint_file = f'hashing/SimCLRv2/Pretrained/r{depth}_{width}x_sk1_ema.pth'
    else:
        checkpoint_file = f'hashing/SimCLRv2/Pretrained/r{depth}_{width}x_sk1.pth'
    
    def load(device='cuda'):
        
        assert (device=='cpu' or device=='cuda')
        
        # Load the model 
        simclr, _ = SIMv2.get_resnet(depth=depth, width_multiplier=width, sk_ratio=0.0625)
        checkpoint = torch.load(checkpoint_file)
        simclr.load_state_dict(checkpoint['resnet'])
        simclr.eval()
        simclr.to(torch.device(device))
    
        return simclr
    
    
    return load

image = Image.open('/Users/cyrilvallez/Desktop/Project/Datasets/BSDS500/Control/data221.jpg')

simclr_ema = load_simclr_v2(depth=152, width=3, ema=True)(device='cpu')
simclr_no_ema = load_simclr_v2(depth=152, width=3, ema=False)(device='cpu')

transforms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.LANCZOS),
    T.CenterCrop(224),
    T.ToTensor()
    ])

tensor = torch.unsqueeze(transforms(image), dim=0)

out1 = simclr_ema(tensor)
out2 = simclr_no_ema(tensor)

mem_params = sum([param.nelement()*param.element_size() for param in simclr_ema.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in simclr_ema.buffers()])
mem_ema = (mem_params + mem_bufs)/1e9 # in bytes

mem_params = sum([param.nelement()*param.element_size() for param in simclr_no_ema.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in simclr_no_ema.buffers()])
mem_no_ema = (mem_params + mem_bufs)/1e9 # in bytes

alloc = torch.cuda.max_memory_allocated()/1e9
